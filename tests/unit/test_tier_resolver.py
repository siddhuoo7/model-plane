"""Unit tests for TierResolver — Sub-Task 2.3."""

from __future__ import annotations

import pytest

from model_plane.classifier.taxonomy import ComplexityTier
from model_plane.registry.catalog import DeploymentConfig
from model_plane.routing.tier_resolver import TierResolver


# ── helpers ────────────────────────────────────────────────────────────────────

def _dep(name: str, tier: str) -> DeploymentConfig:
    return DeploymentConfig(
        name=name,
        litellm_model=f"watsonx/{name}",
        provider="watsonx",
        tier=tier,
    )


# ── TierResolver.resolve() ─────────────────────────────────────────────────────

class TestTierResolverExactMatch:
    """Pass 1: exact tier match returns the first candidate matching the tier."""

    def test_exact_simple(self):
        candidates = [_dep("wx-small", "simple"), _dep("wx-medium", "medium")]
        dep = TierResolver().resolve(ComplexityTier.SIMPLE, candidates)
        assert dep is not None
        assert dep.name == "wx-small"

    def test_exact_medium(self):
        candidates = [
            _dep("wx-medium-a", "medium"),
            _dep("wx-medium-b", "medium"),
            _dep("wx-complex", "complex"),
        ]
        dep = TierResolver().resolve(ComplexityTier.MEDIUM, candidates)
        assert dep is not None
        assert dep.name == "wx-medium-a"  # first match wins

    def test_exact_complex(self):
        candidates = [_dep("wx-large", "complex"), _dep("wx-reason", "reasoning")]
        dep = TierResolver().resolve(ComplexityTier.COMPLEX, candidates)
        assert dep is not None
        assert dep.name == "wx-large"

    def test_exact_reasoning(self):
        candidates = [_dep("wx-reason", "reasoning"), _dep("wx-large", "complex")]
        dep = TierResolver().resolve(ComplexityTier.REASONING, candidates)
        assert dep is not None
        assert dep.name == "wx-reason"


class TestTierResolverTierUpFallback:
    """Pass 2: when no exact tier match, escalate to next tier up."""

    def test_no_simple_falls_back_to_medium(self):
        """No 'simple' deployment → should pick the 'medium' one."""
        candidates = [_dep("wx-medium", "medium"), _dep("wx-large", "complex")]
        dep = TierResolver().resolve(ComplexityTier.SIMPLE, candidates)
        assert dep is not None
        assert dep.name == "wx-medium"

    def test_no_medium_falls_back_to_complex(self):
        candidates = [_dep("wx-large", "complex"), _dep("wx-reason", "reasoning")]
        dep = TierResolver().resolve(ComplexityTier.MEDIUM, candidates)
        assert dep is not None
        assert dep.name == "wx-large"

    def test_no_complex_falls_back_to_reasoning(self):
        candidates = [_dep("wx-reason", "reasoning")]
        dep = TierResolver().resolve(ComplexityTier.COMPLEX, candidates)
        assert dep is not None
        assert dep.name == "wx-reason"

    def test_skips_multiple_tiers_to_find_match(self):
        """No 'simple', no 'medium' → should pick 'complex'."""
        candidates = [_dep("wx-large", "complex")]
        dep = TierResolver().resolve(ComplexityTier.SIMPLE, candidates)
        assert dep is not None
        assert dep.name == "wx-large"


class TestTierResolverTopRankedFallback:
    """Pass 3: when no tier-up match either, fall back to the top-ranked candidate."""

    def test_no_tier_match_returns_first_candidate(self):
        """No deployment matches 'reasoning' or any higher tier → returns first."""
        candidates = [_dep("wx-small", "small"), _dep("wx-medium", "medium")]
        dep = TierResolver().resolve(ComplexityTier.REASONING, candidates)
        assert dep is not None
        # no "reasoning" match and no tier above reasoning → top-ranked fallback
        assert dep.name == "wx-small"

    def test_single_candidate_always_returned(self):
        """Even if tier doesn't match, a single candidate is always returned."""
        candidates = [_dep("only-dep", "medium")]
        dep = TierResolver().resolve(ComplexityTier.REASONING, candidates)
        assert dep is not None
        assert dep.name == "only-dep"


class TestTierResolverEdgeCases:
    """Edge cases and invariants."""

    def test_empty_candidates_returns_none(self):
        dep = TierResolver().resolve(ComplexityTier.MEDIUM, [])
        assert dep is None

    def test_never_downgrades(self):
        """Resolver must never return a lower-tier deployment when an exact match exists."""
        candidates = [
            _dep("wx-small", "simple"),
            _dep("wx-medium", "medium"),
            _dep("wx-large", "complex"),
        ]
        dep = TierResolver().resolve(ComplexityTier.COMPLEX, candidates)
        assert dep is not None
        assert dep.name == "wx-large"  # exact complex, not simple or medium

    def test_ranked_order_respected(self):
        """When multiple exact-tier matches exist, first (best-ranked) is returned."""
        candidates = [
            _dep("preferred-complex", "complex"),
            _dep("fallback-complex", "complex"),
        ]
        dep = TierResolver().resolve(ComplexityTier.COMPLEX, candidates)
        assert dep is not None
        assert dep.name == "preferred-complex"


# ── Pipeline integration — TierResolver called from Stage 6c ──────────────────

class TestPipelineTierResolver:
    """Verify Stage 6c wires up TierResolver for ml_output_mode=tier."""

    def test_tier_mode_returns_matching_deployment(self, monkeypatch):
        from unittest.mock import MagicMock

        from model_plane.classifier.taxonomy import ComplexityTier
        from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline

        ctx = build_routing_context(
            {"messages": [{"role": "user", "content": "Solve the Riemann hypothesis"}]}
        )

        mock_rec = MagicMock()
        mock_rec.predict.return_value = MagicMock(
            deployment_name=None,
            confidence=0.91,
            predicted_tier=ComplexityTier.REASONING,
            ml_output_mode="tier",
        )

        monkeypatch.setattr("model_plane.ml.recommender.get_recommender", lambda: mock_rec)
        monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", True)
        monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_confidence_threshold", 0.7)
        monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_output_mode", "tier")

        decision = run_routing_pipeline(ctx)

        assert decision.source == "ml_active"
        assert decision.deployment is not None
        assert decision.deployment.tier in ("complex", "reasoning")
        assert decision.ml_tier is not None
        assert decision.resolved_tier is not None

    def test_tier_mode_ml_tier_fields_populated(self, monkeypatch):
        """RoutingDecision.ml_tier and resolved_tier are set when TierResolver fires."""
        from unittest.mock import MagicMock

        from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline

        ctx = build_routing_context(
            {"messages": [{"role": "user", "content": "compare LSM-tree vs B-tree tradeoffs"}]}
        )

        mock_rec = MagicMock()
        mock_rec.predict.return_value = MagicMock(
            deployment_name=None,
            confidence=0.88,
            predicted_tier=ComplexityTier.COMPLEX,
            ml_output_mode="tier",
        )

        monkeypatch.setattr("model_plane.ml.recommender.get_recommender", lambda: mock_rec)
        monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", True)
        monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_confidence_threshold", 0.7)
        monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_output_mode", "tier")

        decision = run_routing_pipeline(ctx)

        assert decision.ml_tier == ComplexityTier.COMPLEX
        assert decision.resolved_tier is not None
