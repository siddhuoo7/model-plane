"""Custom routing strategy — selects a deployment from the filtered candidate list.

This is the main scoring+selection logic that replaces LiteLLM's built-in
routing when routing_mode=custom_plugin.
"""

from __future__ import annotations

from model_plane.classifier.taxonomy import ComplexityTier, TaskType
from model_plane.logging_setup import get_logger
from model_plane.registry.catalog import DeploymentConfig
from model_plane.routing.context import RoutingContext, RoutingDecision

log = get_logger(__name__)

# Task-capability affinity: preferred capabilities per task type
TASK_CAPABILITY_AFFINITY: dict[str, list[str]] = {
    TaskType.CODE_GENERATION: ["code", "function-calling"],
    TaskType.CODE_DEBUGGING: ["code", "reasoning"],
    TaskType.CODE_EDITING: ["code"],
    TaskType.REPOSITORY_SEARCH: ["code", "long-context"],
    TaskType.MATHEMATICAL_REASONING: ["reasoning", "math"],
    TaskType.TECHNICAL_REASONING: ["reasoning"],
    TaskType.LONG_CONTEXT_SYNTHESIS: ["long-context"],
    TaskType.STRUCTURED_EXTRACTION: ["function-calling", "json-mode"],
    TaskType.TOOL_CALL_INTERPRETATION: ["function-calling"],
}


def _preferred_providers() -> list[str]:
    """Load provider preference order from routing config."""
    from model_plane.config import settings
    rc = settings.load_routing_config()
    return rc.get("preferred_providers", [])


def select_deployment(
    ctx: RoutingContext,
    candidates: list[DeploymentConfig],
) -> RoutingDecision | None:
    """Score each candidate and return the best deployment."""
    if not candidates:
        return None

    if len(candidates) == 1:
        return RoutingDecision(
            deployment=candidates[0],
            source="custom_plugin",
            confidence=ctx.routing_confidence,
        )

    task_type = ctx.classification.task_type if ctx.classification else TaskType.UNKNOWN
    tier = ctx.scorer_result.tier if ctx.scorer_result else ComplexityTier.MEDIUM
    provider_order = _preferred_providers()

    scores: list[tuple[float, int, DeploymentConfig]] = []
    for dep in candidates:
        s = _score_candidate(dep, task_type, tier, ctx)
        # Provider preference as tiebreaker: lower index = higher priority
        provider_rank = provider_order.index(dep.provider) if dep.provider in provider_order else len(provider_order)
        scores.append((s, provider_rank, dep))

    # Sort: highest score first; on tie, lowest provider_rank first
    scores.sort(key=lambda t: (-t[0], t[1]))
    best_score, _, best_dep = scores[0]

    # Normalise confidence from score spread
    score_spread = scores[0][0] - scores[-1][0] if len(scores) > 1 else 1.0
    confidence = min(0.5 + score_spread, 0.99)

    log.debug(
        "deployment_scores",
        request_id=ctx.request_id,
        scores=[(d.name, round(s, 3)) for s, _, d in scores],
        selected=best_dep.name,
    )

    return RoutingDecision(
        deployment=best_dep,
        source="custom_plugin",
        confidence=round(confidence, 4),
        reasoning=f"top_score={best_score:.3f}",
    )


def _score_candidate(
    dep: DeploymentConfig,
    task_type: TaskType,
    tier: ComplexityTier,
    ctx: RoutingContext,
) -> float:
    score = 0.0

    # Tier match — exact match is best
    tier_order = [ComplexityTier.SIMPLE, ComplexityTier.MEDIUM, ComplexityTier.COMPLEX, ComplexityTier.REASONING]
    dep_tier_idx = tier_order.index(ComplexityTier(dep.tier)) if dep.tier in [t.value for t in tier_order] else 1
    req_tier_idx = tier_order.index(tier)
    tier_distance = abs(dep_tier_idx - req_tier_idx)
    score += max(0.0, 1.0 - tier_distance * 0.25)

    # Capability affinity
    preferred = TASK_CAPABILITY_AFFINITY.get(task_type, [])
    if preferred:
        matches = sum(1 for cap in preferred if dep.supports(cap))
        score += 0.3 * (matches / len(preferred))

    # Cost preference (lower cost = better, up to 0.2 bonus)
    if dep.cost_per_1k_input < 0.01:
        score += 0.2
    elif dep.cost_per_1k_input < 0.05:
        score += 0.1

    # Context fit — penalise if request exceeds 80% of context window
    if ctx.features:
        token_ratio = ctx.features.total_tokens / max(dep.context_limit, 1)
        if token_ratio > 0.9:
            score -= 0.5  # hard penalise: might not fit
        elif token_ratio > 0.7:
            score -= 0.1

    # Cache warmth bonus
    if ctx.cache_warm and ctx.selected_deployment and ctx.selected_deployment.name == dep.name:
        score += 0.15

    # Local preference for non-sensitive tiers
    if dep.provider == "local" and tier in (ComplexityTier.SIMPLE, ComplexityTier.MEDIUM):
        score += 0.05

    return score
