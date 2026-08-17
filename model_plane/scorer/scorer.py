"""14-dimension weighted scorer.

Produces a single float in [0, 1] representing request complexity.
Higher = more complex → maps to a higher routing tier.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_plane.classifier.features import RequestFeatures
from model_plane.classifier.taxonomy import ComplexityTier

# Default weights from architecture document
DEFAULT_WEIGHTS: dict[str, float] = {
    "reasoning_markers": 0.18,
    "code_presence": 0.15,
    "simple_indicators": 0.12,   # NOTE: inverted — high simple → lower score
    "multi_step_patterns": 0.12,
    "technical_terms": 0.10,
    "token_count": 0.08,
    "creative_markers": 0.05,
    "question_complexity": 0.05,
    "constraint_count": 0.04,
    "imperative_verbs": 0.03,
    "output_format": 0.03,
    "domain_specificity": 0.02,
    "reference_complexity": 0.02,
    "negation_complexity": 0.01,
}

# Thresholds for tier mapping
TIER_THRESHOLDS: dict[str, tuple[float, float]] = {
    "simple": (0.0, 0.25),
    "medium": (0.25, 0.55),
    "complex": (0.55, 0.75),
    "reasoning": (0.75, 1.0),
}


@dataclass
class ScorerResult:
    raw_score: float  # weighted sum in [0, 1]
    tier: ComplexityTier
    dimension_scores: dict[str, float]
    confidence: float  # distance from nearest tier boundary → routing certainty


def score(features: RequestFeatures, weights: dict[str, float] | None = None) -> ScorerResult:
    """Compute weighted complexity score from a RequestFeatures object."""
    w = weights or DEFAULT_WEIGHTS

    dims: dict[str, float] = {
        "reasoning_markers": features.reasoning_markers,
        "code_presence": features.code_presence,
        # simple_indicators is inverted: low simple → more complex
        "simple_indicators": 1.0 - features.simple_indicators,
        "multi_step_patterns": features.multi_step_patterns,
        "technical_terms": features.technical_terms,
        "token_count": features.token_count_signal,
        "creative_markers": features.creative_markers,
        "question_complexity": features.question_complexity,
        "constraint_count": features.constraint_count,
        "imperative_verbs": features.imperative_verbs,
        "output_format": features.output_format,
        "domain_specificity": features.domain_specificity,
        "reference_complexity": features.reference_complexity,
        "negation_complexity": features.negation_complexity,
    }

    total_weight = sum(w.values())
    raw = sum(w.get(k, 0.0) * v for k, v in dims.items()) / total_weight

    tier, confidence = _score_to_tier(raw)

    return ScorerResult(
        raw_score=raw,
        tier=tier,
        dimension_scores=dims,
        confidence=confidence,
    )


def _score_to_tier(raw: float) -> tuple[ComplexityTier, float]:
    """Map a raw score to a ComplexityTier and produce a confidence value."""
    tier_map = {
        "simple": ComplexityTier.SIMPLE,
        "medium": ComplexityTier.MEDIUM,
        "complex": ComplexityTier.COMPLEX,
        "reasoning": ComplexityTier.REASONING,
    }

    selected_tier = "medium"
    for name, (lo, hi) in TIER_THRESHOLDS.items():
        if lo <= raw < hi:
            selected_tier = name
            break

    lo, hi = TIER_THRESHOLDS[selected_tier]
    mid = (lo + hi) / 2.0
    # Confidence: 1.0 when score is at midpoint; drops as it nears a boundary
    half_width = (hi - lo) / 2.0
    distance_from_mid = abs(raw - mid)
    confidence = 1.0 - (distance_from_mid / half_width) if half_width > 0 else 1.0
    confidence = max(0.1, round(confidence, 4))

    return tier_map[selected_tier], confidence
