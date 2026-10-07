"""Cache switch-cost economics — Sub-Task 4.1.

Replaces the simple soft-preference heuristic (stay if context_tokens > 2000)
with a USD switch-cost model.

When a session has accumulated tokens on a warm model, switching to a different
deployment incurs a KV-cache reprefill cost. The router only switches when the
quality gain justifies that cost.

Feature-flag: ``CACHE_MODE=switch_cost`` in ``AppSettings`` enables this module.
Default is ``soft_preference`` (existing behaviour, no change).
"""

from __future__ import annotations

# KV-cache reprefill is billed at the normal input-token rate because the model
# must re-process every token it previously cached. We estimate the penalty as:
#
#   reprefill_cost = accumulated_tokens × input_price_per_token × REPREFILL_FRACTION
#
# REPREFILL_FRACTION ≈ 1.0 when the whole context must be re-ingested; in
# practice providers partially cache the system prompt, so we use 0.7 as a
# conservative default.
REPREFILL_FRACTION: float = 0.7

# Maximum score adjustment from switch-cost penalty. Capped so that a very warm
# cache never completely blocks a highly-superior candidate.
SWITCH_PENALTY_CAP: float = 0.30


def compute_switch_cost(
    warm_deployment_name: str,
    candidate_name: str,
    accumulated_tokens: int,
    input_price_per_token: float,
) -> float:
    """Return the estimated USD reprefill cost for switching from the warm model.

    Returns 0.0 if the candidate is the same as the warm model (no switch).

    Args:
        warm_deployment_name: Name of the currently warm/cached deployment.
        candidate_name: Name of the deployment being considered.
        accumulated_tokens: Number of context tokens accumulated in the session.
        input_price_per_token: Input price in USD per *single* token
            (i.e. ``input_per_mtok_usd / 1_000_000``).

    Returns:
        Estimated USD reprefill cost (0.0 if no switch).
    """
    if candidate_name == warm_deployment_name:
        return 0.0
    return accumulated_tokens * input_price_per_token * REPREFILL_FRACTION


def compute_stay_cost(
    warm_deployment_name: str,
    candidate_name: str,
    tier_gap: int,
    input_price_per_token: float,
) -> float:
    """Return the opportunity cost of staying on the current model.

    If the warm model is in a lower tier than the requested tier, staying
    incurs a quality penalty. We model that as a fraction of the per-token
    cost multiplied by the tier gap.

    Args:
        warm_deployment_name: Name of the currently warm deployment.
        candidate_name: Name of the candidate proposed by the router.
        tier_gap: Integer gap in tiers (0 = same tier; positive = candidate is
            higher tier than warm model).
        input_price_per_token: Input price in USD per token for the candidate.

    Returns:
        Estimated USD opportunity cost of staying (0.0 if no gap or same model).
    """
    if candidate_name == warm_deployment_name or tier_gap <= 0:
        return 0.0
    # Opportunity cost grows linearly with tier gap; scale by request cost proxy
    # (1 synthetic request = 500 input tokens)
    SYNTHETIC_REQUEST_TOKENS = 500
    return tier_gap * SYNTHETIC_REQUEST_TOKENS * input_price_per_token * 0.5


def cache_score_adjustment(
    warm_deployment_name: str | None,
    candidate_name: str,
    accumulated_tokens: int,
    input_price_per_token: float,
    tier_gap: int = 0,
) -> float:
    """Compute the net score adjustment for a candidate given cache state.

    Returns a float in [-SWITCH_PENALTY_CAP, +SWITCH_PENALTY_CAP]:
    - Negative  → penalise switching (stay on warm model).
    - Positive  → incentivise switching (warm model is clearly inferior).

    The adjustment is normalised to [-1, 1] using the sum of both costs, then
    capped at SWITCH_PENALTY_CAP to prevent the cache from completely overriding
    the scorer.

    Args:
        warm_deployment_name: Name of the warm deployment (None if no warm cache).
        candidate_name: Name of the candidate deployment being scored.
        accumulated_tokens: Tokens accumulated in the session on the warm model.
        input_price_per_token: Candidate's input price in USD per token.
        tier_gap: Tier distance (candidate tier rank − warm model tier rank).
            Positive means the candidate is higher tier.

    Returns:
        Score delta to add to the candidate's score (range limited to ±cap).
    """
    if not warm_deployment_name:
        return 0.0

    switch_cost = compute_switch_cost(
        warm_deployment_name, candidate_name, accumulated_tokens, input_price_per_token
    )
    stay_cost = compute_stay_cost(
        warm_deployment_name, candidate_name, tier_gap, input_price_per_token
    )

    # If switching is cheaper than staying, the adjustment is positive.
    # If staying is cheaper, the adjustment is negative.
    total = switch_cost + stay_cost
    if total == 0.0:
        return 0.0

    net = (stay_cost - switch_cost) / total  # in [-1, 1]
    # Clamp to penalty cap
    return max(-SWITCH_PENALTY_CAP, min(SWITCH_PENALTY_CAP, net))
