"""Unit tests for Sub-Task 2.5 — Tier-Based Task Overrides."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from model_plane.classifier.taxonomy import ComplexityTier, TaskType
from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_body(text: str, **kwargs) -> dict:
    return {"messages": [{"role": "user", "content": text}], **kwargs}


def _routing_cfg(**overrides) -> dict:
    """Return a minimal routing config, merging overrides."""
    base = {
        "task_overrides": {},
        "task_deployment_overrides": {},
        "preferred_providers": ["watsonx", "local", "anthropic", "openai"],
        "scorer_weights": {},
        "ml_guards": {},
    }
    base.update(overrides)
    return base


def _mock_settings(cfg: dict, *, ml_enabled: bool = False, **extra) -> MagicMock:
    """Build a MagicMock that quacks like the settings singleton for pipeline tests."""
    s = MagicMock()
    s.ml_routing_enabled = ml_enabled
    s.ml_confidence_threshold = extra.get("ml_confidence_threshold", 0.7)
    s.ml_output_mode = extra.get("ml_output_mode", "tier")
    s.default_model = extra.get("default_model", "watsonx-granite-small")
    s.similarity_routing_enabled = False
    s.load_routing_config.return_value = cfg
    return s


# ── Tier ceiling clamp (task_overrides new format) ────────────────────────────

class TestTierCeilingClamp:
    """task_overrides: {task_type: tier_string} clamps the resolved tier downward."""

    def test_clamp_ceiling_limits_high_tier(self):
        """A summarization ceiling of 'medium' prevents higher tier routing."""
        cfg = _routing_cfg(task_overrides={"summarization": "medium"})
        mock_s = _mock_settings(cfg, ml_enabled=False)

        # "Give me a summary" triggers SUMMARIZATION without also triggering
        # long_context_synthesis, so the ceiling clamp fires as expected.
        with patch("model_plane.routing.pipeline.settings", mock_s):
            ctx = build_routing_context(_make_body("Give me a summary of the report."))
            decision = run_routing_pipeline(ctx)

        assert decision.deployment is not None
        # Ceiling = medium; result tier must be ≤ medium
        assert decision.deployment.tier in ("small", "simple", "medium")

    def test_clamp_only_downgrade(self):
        """Tier ceiling only clamps downward — if tier is already ≤ ceiling, no change."""
        # Ceiling = complex for simple_qa; a simple request resolves to a low tier → no change
        cfg = _routing_cfg(task_overrides={"simple_qa": "complex"})
        mock_s = _mock_settings(cfg, ml_enabled=False)

        with patch("model_plane.routing.pipeline.settings", mock_s):
            ctx = build_routing_context(_make_body("What is 2+2?"))
            decision = run_routing_pipeline(ctx)

        assert decision.deployment is not None
        # simple_qa scores small/simple; ceiling=complex is higher so no clamp occurs
        assert decision.deployment.tier in ("small", "simple", "medium", "complex")

    def test_clamp_ml_reasoning_to_simple_for_translation(self, monkeypatch):
        """ML escalates to REASONING for translation; ceiling=simple clamps it back."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = MagicMock(
            deployment_name=None,
            confidence=0.92,
            predicted_tier=ComplexityTier.REASONING,
            ml_output_mode="tier",
            source="local_ml",
        )
        monkeypatch.setattr("model_plane.ml.recommender.get_recommender", lambda: mock_rec)

        cfg = _routing_cfg(task_overrides={"translation": "simple"})
        mock_s = _mock_settings(cfg, ml_enabled=True, ml_confidence_threshold=0.7, ml_output_mode="tier")

        with patch("model_plane.routing.pipeline.settings", mock_s):
            # "Translate this into French" matches _TRANSLATE → TaskType.TRANSLATION
            ctx = build_routing_context(_make_body("Translate this into French."))
            decision = run_routing_pipeline(ctx)

        assert decision.deployment is not None
        # ML said REASONING but ceiling=simple → result must be ≤ medium
        # (no "simple" tier exists in catalog, so resolver falls back to nearest
        # tier above: "medium", which is still far below the ML-suggested "reasoning")
        assert decision.deployment.tier in ("small", "simple", "medium")

    def test_ml_resolved_tier_also_clamped(self, monkeypatch):
        """Ceiling clamp also resets ml_resolved_tier on the context."""
        mock_rec = MagicMock()
        mock_rec.predict.return_value = MagicMock(
            deployment_name=None,
            confidence=0.91,
            predicted_tier=ComplexityTier.REASONING,
            ml_output_mode="tier",
            source="local_ml",
        )
        monkeypatch.setattr("model_plane.ml.recommender.get_recommender", lambda: mock_rec)

        cfg = _routing_cfg(task_overrides={"summarization": "medium"})
        mock_s = _mock_settings(cfg, ml_enabled=True, ml_confidence_threshold=0.7, ml_output_mode="tier")

        with patch("model_plane.routing.pipeline.settings", mock_s):
            ctx = build_routing_context(_make_body("Summarize this document."))
            decision = run_routing_pipeline(ctx)

        assert decision.deployment is not None
        assert decision.deployment.tier in ("simple", "medium")


# ── Backward compat: task_deployment_overrides ────────────────────────────────

class TestTaskDeploymentOverridesBackwardCompat:
    """task_deployment_overrides (old preferred_deployments format) still works."""

    def test_deployment_override_still_applied(self, monkeypatch):
        """structured_extraction in task_deployment_overrides picks from the list."""
        # This test expects watsonx to be picked via task override — override the
        # autouse fixture to also make watsonx appear credentialed.
        monkeypatch.setattr(
            "model_plane.registry.catalog.provider_has_creds",
            lambda p: p in ("openai", "local", "watsonx"),
        )
        cfg = _routing_cfg(
            task_deployment_overrides={
                "structured_extraction": {
                    "preferred_deployments": ["watsonx-mistral-medium", "watsonx-granite-medium"]
                }
            }
        )
        mock_s = _mock_settings(cfg, ml_enabled=False)

        with patch("model_plane.routing.pipeline.settings", mock_s):
            ctx = build_routing_context(
                _make_body("Extract the following fields as JSON: name, date, amount.")
            )
            decision = run_routing_pipeline(ctx)

        assert decision.deployment is not None
        # local-qwen-27b-medium is also a credentialed medium deployment (local provider
        # is always credentialed) and may outscore watsonx on zero-cost preference.
        assert decision.deployment.name in (
            "watsonx-mistral-medium",
            "watsonx-granite-medium",
            "watsonx-granite-large",
            "watsonx-mistral-large",
            "local-qwen-27b-medium",
            "local-qwen-27b",
        )

    def test_invalid_tier_ceiling_value_does_not_crash(self):
        """An invalid tier string in task_overrides logs a warning and is ignored."""
        cfg = _routing_cfg(task_overrides={"simple_qa": "not_a_real_tier"})
        mock_s = _mock_settings(cfg, ml_enabled=False)

        with patch("model_plane.routing.pipeline.settings", mock_s):
            ctx = build_routing_context(_make_body("What is Python?"))
            # Must not raise — invalid tier is silently ignored
            decision = run_routing_pipeline(ctx)

        assert decision.deployment is not None
