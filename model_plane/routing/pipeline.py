"""Main routing pipeline — orchestrates all stages to produce a RoutingDecision."""

from __future__ import annotations

import uuid
from typing import Any

from model_plane.classifier.classifier import classify
from model_plane.classifier.features import extract_features
from model_plane.classifier.taxonomy import ComplexityTier
from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.policy.engine import PolicyEngine
from model_plane.registry.catalog import get_catalog
from model_plane.routing.context import RoutingContext, RoutingDecision
from model_plane.routing.strategy import select_deployment
from model_plane.scorer.scorer import score

# Tier numeric order for blending
_TIER_RANK: dict[str, int] = {
    ComplexityTier.SIMPLE.value: 0,
    ComplexityTier.MEDIUM.value: 1,
    ComplexityTier.COMPLEX.value: 2,
    ComplexityTier.REASONING.value: 3,
}
_RANK_TIER: dict[int, ComplexityTier] = {v: k for k, v in _TIER_RANK.items()}  # type: ignore[misc]

log = get_logger(__name__)

_policy_engine: PolicyEngine | None = None


def _get_policy_engine() -> PolicyEngine:
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine(policies=None)
    return _policy_engine


def build_routing_context(
    raw_request: dict[str, Any],
    *,
    request_id: str | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
    agent_type: str | None = None,
) -> RoutingContext:
    return RoutingContext(
        request_id=request_id or str(uuid.uuid4()),
        raw_request=raw_request,
        tenant_id=tenant_id,
        session_id=session_id,
        agent_type=agent_type,
    )


def run_routing_pipeline(ctx: RoutingContext) -> RoutingDecision:
    """
    Full routing pipeline:
      1.  Feature extraction
      2.  Task classification
      3.  14-dimension scoring
      3.5 Tier blending — max(scorer_tier, classifier_tier) fixes two-brain disagreement
      3.6 Active ML recommendation (if ml_routing_enabled and model loaded)
      4.  Build candidate pool
      5.  Policy enforcement
      6.  Tier filtering (uses blended tier)
      6b. Task override (highest priority)
      7.  Custom strategy selection
    """
    catalog = get_catalog()
    routing_config = settings.load_routing_config()

    # ── 1. Feature extraction ────────────────────────────────────────────────
    ctx.features = extract_features(ctx.raw_request)
    log.debug(
        "step1_features",
        request_id=ctx.request_id,
        tokens=ctx.features.total_tokens,
        reasoning_markers=round(ctx.features.reasoning_markers, 3),
        code_presence=round(ctx.features.code_presence, 3),
        simple_indicators=round(ctx.features.simple_indicators, 3),
        multi_step=round(ctx.features.multi_step_patterns, 3),
        technical_terms=round(ctx.features.technical_terms, 3),
        has_tools=ctx.features.has_tools,
        language_hint=ctx.features.language_hint,
    )

    # ── 2. Task classification ───────────────────────────────────────────────
    ctx.classification = classify(ctx.features)
    log.debug(
        "step2_classification",
        request_id=ctx.request_id,
        task=ctx.classification.task_type.value,
        tier=ctx.classification.complexity_tier.value,
        confidence=round(ctx.classification.confidence, 3),
        top_signals={k: round(v, 2) for k, v in ctx.classification.signals.items() if v > 0},
    )

    # ── 3. 14-dim scoring ────────────────────────────────────────────────────
    custom_weights = routing_config.get("scorer_weights")
    ctx.scorer_result = score(ctx.features, weights=custom_weights)
    log.debug(
        "step3_score",
        request_id=ctx.request_id,
        raw_score=round(ctx.scorer_result.raw_score, 4),
        tier=ctx.scorer_result.tier.value,
        score_confidence=round(ctx.scorer_result.confidence, 3),
        top_dimensions={
            k: round(v, 3)
            for k, v in sorted(ctx.scorer_result.dimension_scores.items(), key=lambda x: -x[1])
            if v > 0.01
        },
    )

    # ── 2.5 Similarity fast-path ─────────────────────────────────────────────
    # If the similarity index has enough history and returns a high-confidence
    # match, short-circuit to the matched deployment immediately.
    if getattr(settings, "similarity_routing_enabled", False):
        try:
            from model_plane.routing.similarity_router import get_similarity_router
            sim_router = get_similarity_router()
            if sim_router:
                last_user = ctx.features.last_user_text if ctx.features else ""
                sim_result = sim_router.predict(last_user)
                if sim_result:
                    sim_dep = get_catalog().get(sim_result.deployment_name)
                    if sim_dep and sim_dep.healthy:
                        ctx.selected_deployment = sim_dep
                        ctx.routing_source = "similarity"
                        ctx.routing_confidence = sim_result.confidence
                        log.info(
                            "routing_decision",
                            request_id=ctx.request_id,
                            deployment=sim_dep.name,
                            model=sim_dep.litellm_model,
                            task=ctx.classification.task_type.value,
                            tier="similarity_fast_path",
                            raw_score=0.0,
                            source="similarity",
                            confidence=sim_result.confidence,
                            task_override_used=False,
                            ml_recommended=None,
                            ml_confidence=0.0,
                        )
                        return RoutingDecision(
                            deployment=sim_dep,
                            source="similarity",
                            confidence=sim_result.confidence,
                            reasoning=f"similarity_top_cos={sim_result.confidence:.3f}",
                        )
        except Exception as exc:
            log.warning("similarity_routing_failed", request_id=ctx.request_id, error=str(exc))

    # ── 3.5 Tier blending — take the max of scorer and classifier tiers ──────
    # Fixes the "two-brain" problem: a short code request scores SIMPLE by the
    # 14-dim scorer (no code blocks) but classifies CODE_GENERATION (COMPLEX).
    # Taking the max prevents silent under-routing on keyword-sparse prompts.
    scorer_rank = _TIER_RANK.get(ctx.scorer_result.tier.value, 1)
    clf_rank = _TIER_RANK.get(ctx.classification.complexity_tier.value, 1)
    blended_rank = max(scorer_rank, clf_rank)
    blended_tier = ComplexityTier(_RANK_TIER[blended_rank])

    if blended_tier != ctx.scorer_result.tier:
        log.debug(
            "step3_5_tier_blended",
            request_id=ctx.request_id,
            scorer_tier=ctx.scorer_result.tier.value,
            classifier_tier=ctx.classification.complexity_tier.value,
            blended_tier=blended_tier.value,
        )
        ctx.scorer_result.tier = blended_tier  # mutate in-place so Step 6 uses it

    # ── 3.6 Active ML recommendation ────────────────────────────────────────
    # When ml_routing_enabled=true and a trained model exists, the ML prediction
    # can override the blended tier if its confidence exceeds the threshold.
    if settings.ml_routing_enabled:
        try:
            from model_plane.ml.recommender import get_recommender
            rec = get_recommender()
            if rec and ctx.features and ctx.scorer_result:
                ml_result = rec.predict(ctx.features, ctx.scorer_result)
                ctx.ml_recommended_deployment = ml_result.deployment_name
                ctx.ml_recommendation_confidence = ml_result.confidence
                if ml_result.confidence >= settings.ml_confidence_threshold:
                    ml_tier = ComplexityTier(ml_result.predicted_tier)
                    ml_rank = _TIER_RANK.get(ml_tier.value, blended_rank)
                    # ML can escalate but not downgrade (conservative)
                    if ml_rank > blended_rank:
                        log.debug(
                            "step3_6_ml_tier_escalation",
                            request_id=ctx.request_id,
                            blended_tier=blended_tier.value,
                            ml_tier=ml_tier.value,
                            ml_confidence=round(ml_result.confidence, 3),
                        )
                        ctx.scorer_result.tier = ml_tier
                    ctx.routing_source = "ml_active"
        except Exception as exc:
            log.warning("ml_active_failed", request_id=ctx.request_id, error=str(exc))

    # ── 4. Build candidate pool ──────────────────────────────────────────────
    all_healthy = catalog.all_healthy()
    log.debug(
        "step4_candidates",
        request_id=ctx.request_id,
        healthy_count=len(all_healthy),
        deployments=[d.name for d in all_healthy],
    )
    if not all_healthy:
        log.error("no_healthy_deployments", request_id=ctx.request_id)
        return _fallback_decision(ctx, catalog)

    # ── 5. Policy enforcement ────────────────────────────────────────────────
    policy_engine = _get_policy_engine()
    policy_decision = policy_engine.enforce(
        tenant_id=ctx.tenant_id,
        task_type=ctx.classification.task_type,
        candidates=all_healthy,
        estimated_tokens=ctx.features.total_tokens,
    )
    log.debug(
        "step5_policy",
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        allowed=policy_decision.allowed,
        reason=policy_decision.reason or "ok",
        after_filter=[d.name for d in policy_decision.filtered_deployments],
    )

    if not policy_decision.allowed:
        log.warning("routing_blocked_by_policy", request_id=ctx.request_id, reason=policy_decision.reason)
        return _fallback_decision(ctx, catalog)

    ctx.candidate_deployments = policy_decision.filtered_deployments

    # ── 6. Tier filtering (uses blended tier) ───────────────────────────────
    tier = ctx.scorer_result.tier
    tier_candidates = [d for d in ctx.candidate_deployments if d.tier == tier.value]
    if not tier_candidates:
        tier_candidates = ctx.candidate_deployments
        log.debug("step6_tier_broadened", request_id=ctx.request_id, tier=tier.value,
                  reason="no_deployments_in_tier", using=[d.name for d in tier_candidates])
    else:
        log.debug("step6_tier_filter", request_id=ctx.request_id, tier=tier.value,
                  candidates=[d.name for d in tier_candidates])

    # ── 6b. Task override check ──────────────────────────────────────────────
    task_overrides = routing_config.get("task_overrides", {})
    override_names: list[str] = task_overrides.get(
        ctx.classification.task_type.value, {}
    ).get("preferred_deployments", [])
    if override_names:
        override_candidates = [
            catalog.get(n) for n in override_names
            if catalog.get(n) and catalog.get(n).healthy
        ]
        if override_candidates:
            log.debug(
                "step6b_task_override_applied",
                request_id=ctx.request_id,
                task=ctx.classification.task_type.value,
                override_deployments=[d.name for d in override_candidates],
            )
            tier_candidates = override_candidates

    # ── 7. Custom strategy selection ────────────────────────────────────────
    decision = select_deployment(ctx, tier_candidates)
    if decision is None:
        return _fallback_decision(ctx, catalog)

    ctx.selected_deployment = decision.deployment
    if ctx.routing_source == "unset":
        ctx.routing_source = decision.source
    ctx.routing_confidence = decision.confidence

    log.info(
        "routing_decision",
        request_id=ctx.request_id,
        deployment=decision.deployment.name,
        model=decision.deployment.litellm_model,
        task=ctx.classification.task_type.value,
        tier=tier.value,
        raw_score=round(ctx.scorer_result.raw_score, 4),
        source=ctx.routing_source,
        confidence=decision.confidence,
        task_override_used=bool(override_names),
        ml_recommended=ctx.ml_recommended_deployment,
        ml_confidence=round(ctx.ml_recommendation_confidence, 3),
    )

    return decision


def _fallback_decision(ctx: RoutingContext, catalog) -> RoutingDecision:
    """Return the configured default deployment or first healthy one."""
    dep = catalog.get(settings.default_model) or (catalog.all_healthy() or [None])[0]
    if dep is None:
        raise RuntimeError("No deployments available — cannot route request")
    ctx.fallback_used = True
    ctx.routing_source = "default"
    return RoutingDecision(
        deployment=dep,
        source="default",
        confidence=0.5,
        reasoning="fallback_to_default",
    )


