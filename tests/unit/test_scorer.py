"""Unit tests for the 14-dimension scorer with cost decay and local bonus."""
import math

from model_plane.classifier.features import RequestFeatures
from model_plane.registry.catalog import DeploymentConfig
from model_plane.scorer.scorer import (
    ComplexityTier,
    DEFAULT_COST_INPUT_WEIGHT,
    DEFAULT_COST_OUTPUT_WEIGHT,
    INPUT_DECAY,
    LOCAL_BONUS_COMPLEX_REASONING,
    LOCAL_BONUS_SIMPLE_MEDIUM,
    OUTPUT_DECAY,
    _cost_score,
    _local_provider_bonus,
    score,
)


def _make_features(**kwargs) -> RequestFeatures:
    f = RequestFeatures()
    for k, v in kwargs.items():
        setattr(f, k, v)
    return f


def test_simple_request_scores_low():
    f = _make_features(simple_indicators=0.9, token_count_signal=0.02, reasoning_markers=0.0, code_presence=0.0)
    result = score(f)
    assert result.tier == ComplexityTier.SIMPLE
    assert result.raw_score < 0.30


def test_reasoning_request_scores_high():
    f = _make_features(
        reasoning_markers=0.9,
        code_presence=0.8,
        multi_step_patterns=0.8,
        technical_terms=0.7,
        token_count_signal=0.5,
        simple_indicators=0.0,
    )
    result = score(f)
    assert result.tier in (ComplexityTier.COMPLEX, ComplexityTier.REASONING)
    assert result.raw_score > 0.50


def test_custom_weights_applied():
    f = _make_features(reasoning_markers=1.0, code_presence=0.0)
    result_default = score(f)
    custom_weights = {k: 0.0 for k in ["reasoning_markers", "code_presence", "simple_indicators",
        "multi_step_patterns", "technical_terms", "token_count", "creative_markers",
        "question_complexity", "constraint_count", "imperative_verbs", "output_format",
        "domain_specificity", "reference_complexity", "negation_complexity"]}
    custom_weights["simple_indicators"] = 1.0
    result_custom = score(f, weights=custom_weights)
    # with simple_indicators=1.0 weight and feature=0 (so inverted=1), score is high
    assert result_custom.raw_score != result_default.raw_score


def test_confidence_is_bounded():
    f = _make_features()
    result = score(f)
    assert 0.0 <= result.confidence <= 1.0


# ── Sub-Task 1.4: exponential cost decay ─────────────────────────────────────

def _make_dep(cost_input: float, cost_output: float, provider: str = "openai") -> DeploymentConfig:
    return DeploymentConfig(
        name="test-dep",
        litellm_model="openai/gpt-test",
        provider=provider,
        cost_per_1k_input=cost_input,
        cost_per_1k_output=cost_output,
    )


def test_cost_score_zero_when_no_deployment():
    """score() without a deployment contributes 0 cost sub-score."""
    f = _make_features(simple_indicators=0.5, token_count_signal=0.1)
    result_no_dep = score(f)
    result_with_dep = score(f, deployment=_make_dep(0.0, 0.0))
    # Zero-cost deployment: exp(0) * weight for both → non-zero boost
    assert result_with_dep.raw_score >= result_no_dep.raw_score


def test_exponential_decay_is_continuous():
    """$1.50/1k and $4.00/1k should score differently (step-function would bucket them same)."""
    f = _make_features()
    cheap_dep = _make_dep(0.0015, 0.002)   # $1.50/MTok input ≈ $0.0015/1k
    expensive_dep = _make_dep(0.004, 0.01)  # $4.00/MTok input ≈ $0.004/1k
    score_cheap = score(f, deployment=cheap_dep).raw_score
    score_pricey = score(f, deployment=expensive_dep).raw_score
    assert score_cheap > score_pricey, "Cheaper deployment must score higher"


def test_cost_score_helper_formula():
    """Verify the formula: exp(-cost / λ) * weight."""
    dep = _make_dep(0.001, 0.005)
    expected = (
        math.exp(-0.001 / INPUT_DECAY) * DEFAULT_COST_INPUT_WEIGHT
        + math.exp(-0.005 / OUTPUT_DECAY) * DEFAULT_COST_OUTPUT_WEIGHT
    )
    assert abs(_cost_score(dep, DEFAULT_COST_INPUT_WEIGHT, DEFAULT_COST_OUTPUT_WEIGHT) - expected) < 1e-9


def test_score_capped_at_1():
    """score() must never exceed 1.0 even with local bonus + cost sub-score."""
    f = _make_features(
        reasoning_markers=1.0, code_presence=1.0, multi_step_patterns=1.0,
        technical_terms=1.0, token_count_signal=1.0, simple_indicators=0.0,
    )
    dep = _make_dep(0.0, 0.0, provider="local")
    result = score(f, deployment=dep, tier=ComplexityTier.SIMPLE)
    assert result.raw_score <= 1.0


# ── Sub-Task 1.5: tier-aware local/vLLM bonus ────────────────────────────────

def test_local_bonus_tier_aware():
    """Simple tier gets 0.20 bonus, complex gets 0.12."""
    local_dep = _make_dep(0.0, 0.0, provider="local")
    assert _local_provider_bonus(local_dep, ComplexityTier.SIMPLE)   == LOCAL_BONUS_SIMPLE_MEDIUM
    assert _local_provider_bonus(local_dep, ComplexityTier.MEDIUM)   == LOCAL_BONUS_SIMPLE_MEDIUM
    assert _local_provider_bonus(local_dep, ComplexityTier.COMPLEX)  == LOCAL_BONUS_COMPLEX_REASONING
    assert _local_provider_bonus(local_dep, ComplexityTier.REASONING) == LOCAL_BONUS_COMPLEX_REASONING


def test_no_local_bonus_for_cloud_provider():
    cloud_dep = _make_dep(0.005, 0.015, provider="openai")
    assert _local_provider_bonus(cloud_dep, ComplexityTier.SIMPLE) == 0.0


def test_no_local_bonus_without_deployment():
    assert _local_provider_bonus(None, ComplexityTier.SIMPLE) == 0.0


def test_local_dep_scores_higher_for_same_complexity():
    """A local deployment should score higher than an equal-complexity cloud dep."""
    f = _make_features(simple_indicators=0.5, token_count_signal=0.1)
    tier = ComplexityTier.SIMPLE
    local_dep = _make_dep(0.0, 0.0, provider="local")
    cloud_dep = _make_dep(0.0, 0.0, provider="openai")
    score_local = score(f, deployment=local_dep, tier=tier).raw_score
    score_cloud = score(f, deployment=cloud_dep, tier=tier).raw_score
    assert score_local > score_cloud
