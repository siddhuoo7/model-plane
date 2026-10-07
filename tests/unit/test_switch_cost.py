"""Unit tests for Sub-Task 4.1 — Cache Switch-Cost Economics."""

from __future__ import annotations

import pytest

from model_plane.cache.switch_cost import (
    SWITCH_PENALTY_CAP,
    cache_score_adjustment,
    compute_stay_cost,
    compute_switch_cost,
)
from model_plane.cache.session import SessionState


# ── compute_switch_cost ────────────────────────────────────────────────────────

class TestComputeSwitchCost:
    def test_same_model_zero_cost(self):
        cost = compute_switch_cost("model-a", "model-a", 10_000, 0.000001)
        assert cost == 0.0

    def test_different_model_positive_cost(self):
        cost = compute_switch_cost("model-a", "model-b", 10_000, 0.000001)
        # 10_000 tokens × 0.000001 $/tok × 0.7 fraction = 0.007 USD
        assert cost == pytest.approx(0.007, rel=1e-4)

    def test_zero_tokens_zero_cost(self):
        cost = compute_switch_cost("model-a", "model-b", 0, 0.000001)
        assert cost == 0.0

    def test_scales_with_token_count(self):
        cost_small = compute_switch_cost("a", "b", 1_000, 0.000001)
        cost_large = compute_switch_cost("a", "b", 100_000, 0.000001)
        assert cost_large == cost_small * 100


# ── compute_stay_cost ─────────────────────────────────────────────────────────

class TestComputeStayCost:
    def test_same_model_zero_cost(self):
        cost = compute_stay_cost("model-a", "model-a", 2, 0.000001)
        assert cost == 0.0

    def test_zero_tier_gap_zero_cost(self):
        # Candidate is in the same tier as warm model
        cost = compute_stay_cost("model-a", "model-b", 0, 0.000001)
        assert cost == 0.0

    def test_negative_tier_gap_zero_cost(self):
        # Candidate is *lower* tier — no incentive to switch
        cost = compute_stay_cost("model-a", "model-b", -1, 0.000001)
        assert cost == 0.0

    def test_positive_tier_gap_positive_cost(self):
        cost = compute_stay_cost("model-a", "model-b", 2, 0.000001)
        assert cost > 0.0

    def test_scales_with_gap(self):
        cost_1 = compute_stay_cost("a", "b", 1, 0.000001)
        cost_2 = compute_stay_cost("a", "b", 2, 0.000001)
        assert cost_2 == pytest.approx(cost_1 * 2, rel=1e-4)


# ── cache_score_adjustment ────────────────────────────────────────────────────

class TestCacheScoreAdjustment:
    def test_no_warm_model_zero(self):
        adj = cache_score_adjustment(None, "model-b", 10_000, 0.000001)
        assert adj == 0.0

    def test_warm_model_same_candidate_positive(self):
        # Candidate IS the warm model → switch_cost = 0, stay_cost = 0 → adj = 0
        adj = cache_score_adjustment("model-a", "model-a", 10_000, 0.000001, tier_gap=0)
        assert adj == 0.0

    def test_warm_model_switch_penalised(self):
        # Large accumulated tokens → high switch cost → negative adjustment (penalty)
        adj = cache_score_adjustment("model-a", "model-b", 1_000_000, 0.000001, tier_gap=0)
        assert adj < 0.0

    def test_penalty_capped(self):
        # Even with extreme token count the penalty stays within cap
        adj = cache_score_adjustment("model-a", "model-b", 999_999_999, 1.0, tier_gap=0)
        assert adj >= -SWITCH_PENALTY_CAP

    def test_quality_gain_can_overcome_penalty(self):
        # Large tier gap (big quality improvement) → adjustment close to +cap
        adj = cache_score_adjustment("model-a", "model-b", 0, 0.000001, tier_gap=3)
        # With zero accumulated tokens switch_cost=0, stay_cost>0 → positive adj
        assert adj > 0.0

    def test_result_within_cap_range(self):
        for tokens in [0, 100, 10_000, 1_000_000]:
            for gap in [-1, 0, 1, 2]:
                adj = cache_score_adjustment("warm", "other", tokens, 0.000001, tier_gap=gap)
                assert -SWITCH_PENALTY_CAP <= adj <= SWITCH_PENALTY_CAP


# ── SessionState warm tracking ────────────────────────────────────────────────

class TestSessionStateWarmTracking:
    def test_has_warm_model_ids_field(self):
        state = SessionState(
            session_id="s1",
            provider="openai",
            model="gpt-4o",
            prefix_hash="abc",
            context_tokens=100,
            warm_model_ids={"gpt-4o"},
            accumulated_tokens=100,
        )
        assert "gpt-4o" in state.warm_model_ids
        assert state.accumulated_tokens == 100

    def test_default_warm_model_ids_is_empty_set(self):
        state = SessionState(
            session_id="s2",
            provider="openai",
            model="gpt-4o",
            prefix_hash="def",
            context_tokens=0,
        )
        assert state.warm_model_ids == set()
        assert state.accumulated_tokens == 0
