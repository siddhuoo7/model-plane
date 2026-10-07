"""Local ML recommender — predicts complexity tier from request features.

Phase 4: loaded from a pre-trained scikit-learn model (joblib file).
Shadow mode: recommendation is captured in RoutingContext but does not affect routing
until ml_routing_enabled=true and confidence >= ml_confidence_threshold.

Two output modes controlled by ML_OUTPUT_MODE:
  "tier"       (default) — model predicts ComplexityTier; pipeline calls TierResolver.
  "deployment" (rollback) — legacy model predicts deployment name directly.
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_plane.classifier.features import RequestFeatures
from model_plane.classifier.taxonomy import ComplexityTier, TaskType
from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.scorer.scorer import ScorerResult

log = get_logger(__name__)


@dataclass
class MLRecommendation:
    """Result returned by LocalMLRecommender.predict().

    In "tier" mode  (ml_output_mode="tier"):
        predicted_tier  — ComplexityTier enum value (the primary output)
        deployment_name — None (resolved later by TierResolver)

    In "deployment" mode  (ml_output_mode="deployment", legacy rollback):
        deployment_name — deployment alias string (as before)
        predicted_tier  — str derived from the chosen deployment's tier tag
    """
    predicted_tier: ComplexityTier | str   # ComplexityTier when mode=tier; str when mode=deployment
    confidence: float
    ml_output_mode: str = "tier"           # "tier" | "deployment"
    deployment_name: str | None = None     # None in tier mode; set in deployment mode
    source: str = "local_ml"


class LocalMLRecommender:
    """Wraps a scikit-learn classifier trained on historical routing data."""

    def __init__(self, model_path: Path) -> None:
        self._model: Any = None
        self._model_path = model_path
        self._loaded = False
        self._class_names: list[str] = []
        self._load()

    def _load(self) -> None:
        try:
            import joblib

            if self._model_path.exists():
                loaded = joblib.load(self._model_path)
                if isinstance(loaded, dict):
                    self._model = loaded.get("model")
                    self._class_names = list(loaded.get("class_names", []))
                else:
                    self._model = loaded
                    self._class_names = []
                self._loaded = self._model is not None
                log.info("ml_model_loaded", path=str(self._model_path))
            else:
                log.warning("ml_model_not_found", path=str(self._model_path))
        except Exception as exc:
            log.warning("ml_model_load_failed", error=str(exc))

    def predict(
        self,
        features: RequestFeatures,
        scorer_result: ScorerResult,
        task_type: TaskType | None = None,
        blended_tier: ComplexityTier | None = None,
        classifier_conf: float = 0.0,
    ) -> MLRecommendation:
        """Run inference and return an MLRecommendation.

        When ``settings.ml_output_mode == "tier"`` (default), the model's output
        class is interpreted as a ComplexityTier.  When ``"deployment"`` (rollback),
        the output class is treated as a deployment name string.

        Guard rails (tier mode only):
          skipMl          — skip inference entirely when classifier is decisive but
                            scorer is uncertain (avoids contradictory signals).
          upgrade_cap     — cap the predicted tier at MEDIUM when scorer says SIMPLE
                            and classifier is confident (prevents over-escalation).
          downgrade_veto  — refuse to downgrade below blended_tier when classifier
                            is confident (prevents contradictory decisions).

        Args:
            features:        Extracted request features.
            scorer_result:   14-dim scorer output (includes ``tier`` and ``confidence``).
            task_type:       Optional task type for 15-dim feature vector.
            blended_tier:    The blended tier from Stage 3.5 (max of scorer/classifier).
                             Used by upgrade_cap and downgrade_veto guards.
            classifier_conf: Confidence from the classifier step (for guard thresholds).
        """
        output_mode = getattr(settings, "ml_output_mode", "tier")
        _blended = blended_tier if blended_tier is not None else scorer_result.tier

        if not self._loaded or self._model is None:
            # Fallback: return scorer tier directly
            return MLRecommendation(
                predicted_tier=scorer_result.tier,   # ComplexityTier enum
                confidence=0.0,
                ml_output_mode=output_mode,
                deployment_name=None,
                source="scorer_fallback",
            )

        # ── skipMl guard ──────────────────────────────────────────────────────
        # Applied before inference: if classifier is very confident but scorer is
        # uncertain, skip ML and let the blended tier stand.
        if output_mode == "tier" and self._should_skip_ml(
            classifier_conf=classifier_conf,
            scorer_conf=scorer_result.confidence,
        ):
            log.debug(
                "ml_guard_skip_ml",
                classifier_conf=round(classifier_conf, 3),
                scorer_conf=round(scorer_result.confidence, 3),
                blended_tier=_blended.value,
            )
            return MLRecommendation(
                predicted_tier=_blended,
                confidence=0.0,
                ml_output_mode=output_mode,
                deployment_name=None,
                source="skip_ml_guard",
            )

        vector = features.as_vector(task_type=task_type)
        try:
            proba = self._model.predict_proba([vector])[0]
            predicted_idx = int(proba.argmax())
            confidence = float(proba[predicted_idx])
            raw_label = (
                self._class_names[predicted_idx]
                if self._class_names
                else scorer_result.tier.value
            )

            if output_mode == "deployment":
                # Legacy path: raw_label is a deployment name; no guards applied
                predicted_tier_str = self._deployment_to_tier(raw_label, scorer_result.tier.value)
                return MLRecommendation(
                    predicted_tier=predicted_tier_str,
                    confidence=confidence,
                    ml_output_mode="deployment",
                    deployment_name=raw_label,
                )
            else:
                # Tier mode (default): raw_label is a ComplexityTier value string
                try:
                    predicted_tier = ComplexityTier(raw_label)
                except ValueError:
                    log.warning("ml_unknown_tier_label", label=raw_label)
                    predicted_tier = scorer_result.tier

                # ── Post-inference guards ─────────────────────────────────────
                predicted_tier = self._apply_upgrade_cap(
                    ml_tier=predicted_tier,
                    scorer_tier=scorer_result.tier,
                    classifier_conf=classifier_conf,
                )
                predicted_tier = self._apply_downgrade_veto(
                    ml_tier=predicted_tier,
                    blended_tier=_blended,
                    classifier_conf=classifier_conf,
                )

                return MLRecommendation(
                    predicted_tier=predicted_tier,
                    confidence=confidence,
                    ml_output_mode="tier",
                    deployment_name=None,
                )
        except Exception as exc:
            log.warning("ml_predict_failed", error=str(exc))
            return MLRecommendation(
                predicted_tier=scorer_result.tier,
                confidence=0.0,
                ml_output_mode=output_mode,
                deployment_name=None,
                source="scorer_fallback",
            )

    # ── ML guard rail implementations ─────────────────────────────────────────

    def _load_guard_thresholds(self) -> dict:
        """Load ml_guards thresholds from routing.yaml (with safe defaults)."""
        routing_cfg = settings.load_routing_config()
        return routing_cfg.get("ml_guards", {})

    def _should_skip_ml(self, classifier_conf: float, scorer_conf: float) -> bool:
        """Return True when ML inference should be skipped entirely.

        Trigger: classifier_conf >= skip_ml_classifier_conf AND
                 scorer_conf     <  skip_ml_scorer_conf
        """
        g = self._load_guard_thresholds()
        clf_thresh   = float(g.get("skip_ml_classifier_conf", 0.90))
        score_thresh = float(g.get("skip_ml_scorer_conf",     0.45))
        return classifier_conf >= clf_thresh and scorer_conf < score_thresh

    def _apply_upgrade_cap(
        self,
        ml_tier: ComplexityTier,
        scorer_tier: ComplexityTier,
        classifier_conf: float,
    ) -> ComplexityTier:
        """Cap ml_tier at MEDIUM when scorer says SIMPLE and classifier is confident.

        Trigger: scorer_tier == SIMPLE AND classifier_conf >= upgrade_cap_conf
                 AND ml_tier > MEDIUM
        Action:  set ml_tier = MEDIUM
        """
        g = self._load_guard_thresholds()
        cap_conf = float(g.get("upgrade_cap_conf", 0.80))
        if (
            scorer_tier == ComplexityTier.SIMPLE
            and classifier_conf >= cap_conf
            and ml_tier > ComplexityTier.MEDIUM
        ):
            log.debug(
                "ml_guard_upgrade_cap",
                original_ml_tier=ml_tier.value,
                capped_to="medium",
                scorer_tier=scorer_tier.value,
                classifier_conf=round(classifier_conf, 3),
            )
            return ComplexityTier.MEDIUM
        return ml_tier

    def _apply_downgrade_veto(
        self,
        ml_tier: ComplexityTier,
        blended_tier: ComplexityTier,
        classifier_conf: float,
    ) -> ComplexityTier:
        """Refuse ML downgrade when classifier is confident and blended tier is above SIMPLE.

        Trigger: classifier_conf >= downgrade_veto_conf AND blended_tier > SIMPLE
                 AND ml_tier < blended_tier
        Action:  set ml_tier = blended_tier
        """
        g = self._load_guard_thresholds()
        veto_conf = float(g.get("downgrade_veto_conf", 0.75))
        if (
            classifier_conf >= veto_conf
            and blended_tier > ComplexityTier.SIMPLE
            and ml_tier < blended_tier
        ):
            log.debug(
                "ml_guard_downgrade_veto",
                original_ml_tier=ml_tier.value,
                vetoed_to=blended_tier.value,
                blended_tier=blended_tier.value,
                classifier_conf=round(classifier_conf, 3),
            )
            return blended_tier
        return ml_tier

    def _tier_to_default_deployment(self, tier: str) -> str:
        """Map tier name to a deployment alias, preferring the provider order
        from routing.yaml so watsonx is favoured over openai when both share
        a tier."""
        from model_plane.registry.catalog import get_catalog

        catalog = get_catalog()
        deps = catalog.by_tier(tier)
        if not deps:
            healthy = catalog.all_healthy()
            return healthy[0].name if healthy else settings.default_model

        # Load preferred_providers order from routing config
        routing_cfg = settings.load_routing_config()
        preferred = routing_cfg.get("preferred_providers", [])

        if preferred:
            for provider in preferred:
                for dep in deps:
                    if dep.provider == provider:
                        return dep.name

        return deps[0].name

    def _deployment_to_tier(self, deployment_name: str, fallback_tier: str) -> str:
        from model_plane.registry.catalog import get_catalog

        dep = get_catalog().get(deployment_name)
        return dep.tier if dep else fallback_tier

    def reload(self) -> None:
        self._loaded = False
        self._class_names = []
        self._load()


# ── training data recorder ────────────────────────────────────────────────────


class TrainingDataRecorder:
    """Appends routing outcome records to a CSV file for offline training."""

    _lock = threading.Lock()

    def __init__(self, output_path: Path = Path("model_plane/data/training_data.csv")) -> None:
        self._path = output_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self._path.exists():
            with open(self._path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "timestamp", "request_id", "task_type", "tier",
                    "reasoning_markers", "code_presence", "simple_indicators",
                    "multi_step_patterns", "technical_terms", "token_count_signal",
                    "creative_markers", "question_complexity", "constraint_count",
                    "imperative_verbs", "output_format", "domain_specificity",
                    "reference_complexity", "negation_complexity",
                    "selected_deployment", "latency_ms", "cost_usd",
                    "validation_result", "fallback_used", "final_success",
                ])

    def record(self, ctx: "Any") -> None:  # RoutingContext circular import guard
        try:
            f = ctx.features
            c = ctx.classification
            if not f or not c:
                return
            row = [
                datetime.now(tz=timezone.utc).isoformat(),
                ctx.request_id,
                c.task_type.value,
                ctx.scorer_result.tier.value if ctx.scorer_result else "",
                *f.as_vector(task_type=c.task_type),
                ctx.selected_deployment.name if ctx.selected_deployment else "",
                ctx.latency_ms,
                ctx.cost_usd,
                ctx.validation_result or "",
                ctx.fallback_used,
                ctx.final_success if ctx.final_success is not None else "",
            ]
            with self._lock:
                with open(self._path, "a", newline="") as fh:
                    csv.writer(fh).writerow(row)
        except Exception as exc:
            log.warning("training_record_failed", error=str(exc))


# ── singletons ────────────────────────────────────────────────────────────────

_recommender: LocalMLRecommender | None = None
_recorder: TrainingDataRecorder | None = None


def get_recommender() -> LocalMLRecommender | None:
    """Return the active recommender, or None when ml_routing_enabled=False.

    Always returns None when the flag is off, even if a singleton was previously
    loaded.  This prevents singleton leakage between test runs where a test
    temporarily enables ML routing via monkeypatch.
    """
    global _recommender
    if not settings.ml_routing_enabled:
        return None
    if _recommender is None:
        _recommender = LocalMLRecommender(settings.ml_model_path)
    return _recommender


def get_recorder() -> TrainingDataRecorder:
    global _recorder
    if _recorder is None:
        _recorder = TrainingDataRecorder()
    return _recorder
