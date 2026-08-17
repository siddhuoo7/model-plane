"""Unit tests for the 14-dimension scorer."""
from model_plane.classifier.features import RequestFeatures
from model_plane.scorer.scorer import ComplexityTier, score


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
