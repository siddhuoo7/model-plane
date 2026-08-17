"""Local ML recommender — predicts best deployment tier from request features.

Phase 4: loaded from a pre-trained scikit-learn model (joblib file).
Shadow mode: recommendation is captured in RoutingContext but does not affect routing
until ml_routing_enabled=true and confidence >= ml_confidence_threshold.
"""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_plane.classifier.features import RequestFeatures
from model_plane.classifier.taxonomy import ComplexityTier
from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.scorer.scorer import ScorerResult

log = get_logger(__name__)

TIER_LABELS = [
    ComplexityTier.SIMPLE.value,
    ComplexityTier.MEDIUM.value,
    ComplexityTier.COMPLEX.value,
    ComplexityTier.REASONING.value,
]


@dataclass
class MLRecommendation:
    deployment_name: str  # logical alias
    confidence: float
    predicted_tier: str
    source: str = "local_ml"


class LocalMLRecommender:
    """Wraps a scikit-learn classifier trained on historical routing data."""

    def __init__(self, model_path: Path) -> None:
        self._model: Any = None
        self._model_path = model_path
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            import joblib

            if self._model_path.exists():
                self._model = joblib.load(self._model_path)
                self._loaded = True
                log.info("ml_model_loaded", path=str(self._model_path))
            else:
                log.warning("ml_model_not_found", path=str(self._model_path))
        except Exception as exc:
            log.warning("ml_model_load_failed", error=str(exc))

    def predict(self, features: RequestFeatures, scorer_result: ScorerResult) -> MLRecommendation:
        if not self._loaded or self._model is None:
            # Fall back to scorer tier
            return MLRecommendation(
                deployment_name=self._tier_to_default_deployment(scorer_result.tier.value),
                confidence=0.0,
                predicted_tier=scorer_result.tier.value,
                source="scorer_fallback",
            )

        vector = features.as_vector()
        try:
            proba = self._model.predict_proba([vector])[0]
            predicted_idx = int(proba.argmax())
            confidence = float(proba[predicted_idx])
            tier = TIER_LABELS[predicted_idx]
            return MLRecommendation(
                deployment_name=self._tier_to_default_deployment(tier),
                confidence=confidence,
                predicted_tier=tier,
            )
        except Exception as exc:
            log.warning("ml_predict_failed", error=str(exc))
            return MLRecommendation(
                deployment_name=self._tier_to_default_deployment(scorer_result.tier.value),
                confidence=0.0,
                predicted_tier=scorer_result.tier.value,
                source="scorer_fallback",
            )

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

    def reload(self) -> None:
        self._loaded = False
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
                *f.as_vector(),
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
    global _recommender
    if _recommender is None and settings.ml_routing_enabled:
        _recommender = LocalMLRecommender(settings.ml_model_path)
    return _recommender


def get_recorder() -> TrainingDataRecorder:
    global _recorder
    if _recorder is None:
        _recorder = TrainingDataRecorder()
    return _recorder
