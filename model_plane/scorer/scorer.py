"""14-dimension weighted scorer with optional cost + local-provider bonuses.

Produces a single float in [0, 1] representing request complexity.
Higher = more complex → maps to a higher routing tier.

Cost sub-score (Sub-Task 1.4): exponential decay matching strata scorer.ts.
  score(features, deployment=dep) adds a continuous cost penalty based on the
  deployment's per-1k pricing.  Returns 0 cost contribution when no deployment
  is passed (tier-selection phase before a deployment is chosen).

Local provider bonus (Sub-Task 1.5): tier-aware bonus for local/vLLM providers.
  0.20 for simple/medium requests (local is fast and cheap), 0.12 for complex/
  reasoning (capable but not the only option).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from model_plane.classifier.features import RequestFeatures
from model_plane.classifier.taxonomy import ComplexityTier

if TYPE_CHECKING:
    from model_plane.registry.catalog import DeploymentConfig

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

# ── Cost scoring constants (strata scorer.ts values) ─────────────────────────
# λ for exponential decay: score = exp(-cost_per_1k / λ)
# Higher λ → gentler decay (less price-sensitive).
INPUT_DECAY  = 2.5   # λ for input cost (per 1k tokens, USD)
OUTPUT_DECAY = 8.0   # λ for output cost (per 1k tokens, USD)

# Default weights for cost sub-scores (strata defaults)
DEFAULT_COST_INPUT_WEIGHT  = 0.20
DEFAULT_COST_OUTPUT_WEIGHT = 0.08

# ── Local provider bonus constants (strata scorer.ts values) ─────────────────
LOCAL_BONUS_SIMPLE_MEDIUM       = 0.20   # strong preference for cheap simple tasks
LOCAL_BONUS_COMPLEX_REASONING   = 0.12  # softer preference — still prefer capable models

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


def score(
    features: RequestFeatures,
    weights: dict[str, float] | None = None,
    deployment: "DeploymentConfig | None" = None,
    tier: ComplexityTier | None = None,
) -> ScorerResult:
    """Compute weighted complexity score from a RequestFeatures object.

    Args:
        features:   Extracted request features (always required).
        weights:    Optional override for the 14-dim weights dict.  If the dict
                    contains ``cost_input_weight`` / ``cost_output_weight`` keys
                    those are used for the cost sub-score; otherwise the defaults
                    ``DEFAULT_COST_INPUT_WEIGHT`` / ``DEFAULT_COST_OUTPUT_WEIGHT``
                    apply.
        deployment: When provided, an exponential cost sub-score is added.
                    Pass the candidate deployment being evaluated for selection.
                    Omit during the initial tier-assessment phase.
        tier:       Current complexity tier estimate.  When provided along with
                    a local deployment, the tier-aware bonus is applied.
    """
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

    # Normalise over the 14-dim weights only (cost weights are separate)
    dim_weight_keys = set(DEFAULT_WEIGHTS.keys())
    total_weight = sum(v for k, v in w.items() if k in dim_weight_keys)
    raw = sum(w.get(k, 0.0) * v for k, v in dims.items()) / max(total_weight, 1e-9)

    # ── Cost sub-score (additive, deployment-dependent) ───────────────────────
    cost_input_w  = w.get("cost_input_weight",  DEFAULT_COST_INPUT_WEIGHT)
    cost_output_w = w.get("cost_output_weight", DEFAULT_COST_OUTPUT_WEIGHT)
    raw += _cost_score(deployment, cost_input_w, cost_output_w)

    # ── Local provider bonus (additive) ──────────────────────────────────────
    raw += _local_provider_bonus(deployment, tier)

    raw = min(raw, 1.0)

    tier_result, confidence = _score_to_tier(raw)

    return ScorerResult(
        raw_score=raw,
        tier=tier_result,
        dimension_scores=dims,
        confidence=confidence,
    )


def _cost_score(
    dep: "DeploymentConfig | None",
    input_weight: float,
    output_weight: float,
) -> float:
    """Return a cost sub-score using continuous exponential decay.

    Returns 0.0 when no deployment is provided (scores should be deployment-agnostic
    during the initial complexity-tier assessment phase).

    A cheaper deployment contributes a *higher* sub-score (exp → 1 as cost → 0),
    so low-cost deployments are preferred for equivalent complexity.
    """
    if dep is None:
        return 0.0
    input_score  = math.exp(-dep.cost_per_1k_input  / INPUT_DECAY)  * input_weight
    output_score = math.exp(-dep.cost_per_1k_output / OUTPUT_DECAY) * output_weight
    return input_score + output_score


def _local_provider_bonus(
    dep: "DeploymentConfig | None",
    tier: ComplexityTier | None,
) -> float:
    """Return a bonus score for local/vLLM providers, scaled by tier.

    Local models carry no external API cost and have predictable latency.
    The bonus is larger for simple/medium tasks (where local models are
    clearly the right choice) and smaller for complex/reasoning (where
    cloud models may still be better).
    """
    if dep is None or dep.provider != "local":
        return 0.0
    if tier in (ComplexityTier.SIMPLE, ComplexityTier.MEDIUM):
        return LOCAL_BONUS_SIMPLE_MEDIUM
    return LOCAL_BONUS_COMPLEX_REASONING


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
