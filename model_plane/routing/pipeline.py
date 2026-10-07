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
    ctx.classification = classify(ctx.features, owner_id=ctx.tenant_id)
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

    # ── Tier-based task overrides (ceiling clamp, new format) ────────────────
    # task_overrides maps task_type → tier string; clamp blended_tier + ML tier
    # to at most this ceiling so the right model tier is always used.
    _tier_overrides: dict[str, str] = routing_config.get("task_overrides", {})
    _task_tier_ceiling_str = _tier_overrides.get(ctx.classification.task_type.value)
    task_tier_ceiling: ComplexityTier | None = None
    if _task_tier_ceiling_str:
        try:
            task_tier_ceiling = ComplexityTier(_task_tier_ceiling_str)
        except ValueError:
            log.warning(
                "invalid_task_tier_ceiling",
                request_id=ctx.request_id,
                task=ctx.classification.task_type.value,
                value=_task_tier_ceiling_str,
            )

    # ── Deployment-list task overrides (legacy/backward-compat format) ────────
    # Reads from task_deployment_overrides; also falls back to old-style
    # task_overrides that contain preferred_deployments dicts (pre-migration).
    _dep_overrides = routing_config.get("task_deployment_overrides", {})
    # Backward compat: old configs may still use task_overrides with preferred_deployments
    if not _dep_overrides:
        _old_format = routing_config.get("task_overrides", {})
        if isinstance(next(iter(_old_format.values()), None), dict):
            _dep_overrides = _old_format
    override_names: list[str] = _dep_overrides.get(
        ctx.classification.task_type.value, {}
    ).get("preferred_deployments", [])

    # ── 3.6 Active ML recommendation ────────────────────────────────────────
    # When ml_routing_enabled=true and a trained model exists, the ML prediction
    # can override the blended tier if its confidence exceeds the threshold.
    # If classification or scoring is already extremely confident for an
    # override-backed task, keep the override path; otherwise let ML decide.
    #
    # Two modes (settings.ml_output_mode):
    #   "tier"       — model outputs ComplexityTier; we escalate/accept the tier
    #                  and let Steps 6-7 select a deployment within it.
    #   "deployment" — legacy: model outputs a deployment name directly (rollback).
    override_locked = bool(override_names) and (
        ctx.classification.confidence >= 0.98
        or ctx.scorer_result.confidence >= 0.98
    )
    if settings.ml_routing_enabled and not override_locked:
        try:
            from model_plane.ml.recommender import get_recommender
            rec = get_recommender()
            if rec and ctx.features and ctx.scorer_result:
                ml_result = rec.predict(
                    ctx.features,
                    ctx.scorer_result,
                    task_type=ctx.classification.task_type if ctx.classification else None,
                    blended_tier=blended_tier,
                    classifier_conf=ctx.classification.confidence if ctx.classification else 0.0,
                )
                ctx.ml_recommendation_confidence = ml_result.confidence

                if ml_result.confidence >= settings.ml_confidence_threshold:
                    ml_output_mode = getattr(settings, "ml_output_mode", "tier")

                    if ml_output_mode == "deployment" and ml_result.deployment_name:
                        # ── Legacy deployment mode ───────────────────────────
                        ctx.ml_recommended_deployment = ml_result.deployment_name
                        predicted_tier_val = (
                            ml_result.predicted_tier.value
                            if isinstance(ml_result.predicted_tier, ComplexityTier)
                            else str(ml_result.predicted_tier)
                        )
                        ml_tier = ComplexityTier(predicted_tier_val)
                        ml_rank = _TIER_RANK.get(ml_tier.value, blended_rank)
                        ml_dep = catalog.get(ml_result.deployment_name)
                        if ml_dep and ml_dep.healthy:
                            log.debug(
                                "step3_6_ml_deployment_selected",
                                request_id=ctx.request_id,
                                deployment=ml_dep.name,
                                ml_tier=ml_tier.value,
                                ml_confidence=round(ml_result.confidence, 3),
                            )
                            ctx.routing_source = "ml_active"
                            return RoutingDecision(
                                deployment=ml_dep,
                                source="ml_active",
                                confidence=round(ml_result.confidence, 4),
                                reasoning=f"ml_confidence={ml_result.confidence:.3f}",
                            )
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

                    else:
                        # ── Tier mode (default) ──────────────────────────────
                        # predicted_tier is a ComplexityTier enum. Store it on
                        # context so Step 6 can call TierResolver.resolve() with
                        # the policy-filtered candidate list.
                        ml_tier = (
                            ml_result.predicted_tier
                            if isinstance(ml_result.predicted_tier, ComplexityTier)
                            else ComplexityTier(str(ml_result.predicted_tier))
                        )
                        ctx.ml_recommended_deployment = ml_tier.value  # log tier name
                        ctx.ml_resolved_tier = ml_tier                  # picked up in Step 6
                        ml_rank = _TIER_RANK.get(ml_tier.value, blended_rank)
                        # Escalate the blended tier so Step 6 tier-filter also uses it
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

    # ── 3.7 Tier ceiling clamp (tier-based task_overrides) ───────────────────
    # If task_overrides maps this task to a tier ceiling, clamp the current
    # tier (after ML escalation) to at most that ceiling.  This is a downward
    # clamp only — if the resolved tier is already ≤ ceiling, no change.
    if task_tier_ceiling is not None and ctx.scorer_result.tier > task_tier_ceiling:
        log.debug(
            "step3_7_tier_ceiling_clamp",
            request_id=ctx.request_id,
            task=ctx.classification.task_type.value,
            pre_clamp_tier=ctx.scorer_result.tier.value,
            ceiling=task_tier_ceiling.value,
        )
        ctx.scorer_result.tier = task_tier_ceiling
        # Also reset the ml_resolved_tier so TierResolver uses the clamped tier
        if ctx.ml_resolved_tier is not None and ctx.ml_resolved_tier > task_tier_ceiling:
            ctx.ml_resolved_tier = task_tier_ceiling

    # ── L1. Session affinity fast-path ──────────────────────────────────────
    # Level 1: if the session has accumulated enough tokens and a warm deployment
    # is available and healthy, route directly to it without any scoring.
    # This is a *hard short-circuit* (like ML deployment mode) — it returns
    # immediately, skipping security/policy/tier steps.  The security guard
    # already ran above, so this cannot bypass PII-blocking.
    # Feature-flagged: settings.cache_session_affinity_enabled
    if (
        settings.cache_session_affinity_enabled
        and ctx.session_id
    ):
        try:
            from model_plane.cache.session import get_session_cache
            _sc = get_session_cache()
            _ss = _sc.get(ctx.session_id)
            if (
                _ss
                and not _ss.is_expired()
                and _ss.accumulated_tokens >= settings.cache_session_affinity_min_tokens
            ):
                _warm_dep = catalog.get(_ss.model)
                if _warm_dep and _warm_dep.healthy:
                    ctx.affinity_source = "session_affinity"
                    ctx.cache_warm = True
                    ctx.context_reuse_score = 1.0
                    log.info(
                        "cache_l1_session_affinity",
                        request_id=ctx.request_id,
                        session_id=ctx.session_id,
                        deployment=_warm_dep.name,
                        accumulated_tokens=_ss.accumulated_tokens,
                    )
                    return RoutingDecision(
                        deployment=_warm_dep,
                        source="cache_session_affinity",
                        confidence=0.95,
                        reasoning=(
                            f"L1_session_affinity:tokens={_ss.accumulated_tokens}"
                        ),
                    )
        except Exception as _exc:
            log.debug("cache_l1_session_affinity_skipped", error=str(_exc))

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

    # ── 0.5 Security guard — hard eligibility filter ─────────────────────────
    # Classifies data sensitivity of the request (PII, secrets, keywords) and
    # removes any candidates whose provider trust level is not permitted to
    # receive data at that sensitivity level.  This runs BEFORE policy
    # enforcement so the policy engine never even evaluates blocked providers.
    # Feature-flagged: set security_routing_enabled=false to disable entirely.
    # Check if security routing is enabled in in-memory settings OR in DB overrides
    is_sec_enabled = settings.security_routing_enabled
    if is_sec_enabled:
        try:
            import model_plane.db as _db
            for _k in ("__global__", ctx.tenant_id):
                if _k:
                    _sr = _db.load_routing_override("smart_routing", _k)
                    if "security_routing_enabled" in _sr:
                        is_sec_enabled = bool(_sr["security_routing_enabled"])
        except Exception:
            pass

    if is_sec_enabled:
        try:
            from model_plane.routing.security import (
                SecurityContext,
                get_security_guard,
            )
            routing_config = settings.load_routing_config()
            trust_overrides = routing_config.get("provider_trust_overrides", {})
            guard = get_security_guard(trust_overrides or None, owner_id=ctx.tenant_id)
            messages = ctx.raw_request.get("messages", [])
            sensitivity, pii_types, secret_types = guard.classify(messages)
            sec_ctx = SecurityContext(
                data_sensitivity=sensitivity,
                pii_types=pii_types,
                secret_types=secret_types,
            )
            all_healthy = guard.filter(all_healthy, sec_ctx)
            ctx.security_ctx = sec_ctx
            log.debug(
                "step0_5_security",
                request_id=ctx.request_id,
                sensitivity=sensitivity.value,
                pii_types=pii_types or "none",
                secret_types=secret_types or "none",
                blocked=list(sec_ctx.blocked_deployments.keys()) or "none",
                remaining=[d.name for d in all_healthy],
            )
            if not all_healthy:
                log.warning(
                    "all_candidates_blocked_by_security",
                    request_id=ctx.request_id,
                    sensitivity=sensitivity.value,
                    audit=sec_ctx.audit_summary,
                )
                ctx.security_blocked = True
                detected_items = []
                if sec_ctx.pii_types:
                    detected_items.append(f"PII ({', '.join(sec_ctx.pii_types)})")
                if sec_ctx.secret_types:
                    detected_items.append(f"secrets ({', '.join(sec_ctx.secret_types)})")
                detected_str = f" containing {', '.join(detected_items)}" if detected_items else ""
                ctx.security_refusal_message = (
                    f"Request blocked by security policy: {sensitivity.value.upper()} sensitivity data{detected_str} "
                    f"cannot be processed by available providers. No eligible on-premise deployments are currently available."
                )
                ctx.routing_source = "security_guard"
                return RoutingDecision(
                    deployment=None,
                    source="security_guard",
                    confidence=1.0,
                    reasoning=f"security_blocked: sensitivity={sensitivity.value}",
                )
        except Exception as exc:
            log.warning("security_guard_failed", request_id=ctx.request_id, error=str(exc))

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
    if override_names:
        from model_plane.provider_creds import provider_has_creds as _phc
        override_candidates = [
            catalog.get(n) for n in override_names
            if catalog.get(n) and catalog.get(n).healthy and _phc(catalog.get(n).provider)
        ]
        if override_candidates:
            log.debug(
                "step6b_task_override_applied",
                request_id=ctx.request_id,
                task=ctx.classification.task_type.value,
                override_deployments=[d.name for d in override_candidates],
            )
            tier_candidates = override_candidates

    # ── 6b2. Content-hash routing — Level 3 cache-aware routing ─────────────
    # For RAG / document workloads: if context.prefix_hash is set and another
    # deployment is already holding a cache for that hash (tracked via session),
    # prefer that deployment so the document prefix isn't split across endpoints.
    # This is a *soft re-sort* (not a hard short-circuit), ensuring security and
    # policy filters still apply.  Feature-flagged.
    if (
        settings.cache_content_hash_routing_enabled
        and ctx.prefix_hash
        and len(tier_candidates) > 1
    ):
        try:
            from model_plane.cache.session import get_session_cache as _gsc
            _all_sessions = list(getattr(_gsc(), "_sessions", {}).values())
            # Build a mapping: prefix_hash → {deployment_name: sticky_count}
            _hash_affinity: dict[str, int] = {}
            for _sess in _all_sessions:
                if (
                    getattr(_sess, "prefix_hash", None) == ctx.prefix_hash
                    and not _sess.is_expired()
                ):
                    _hash_affinity[_sess.model] = _hash_affinity.get(_sess.model, 0) + 1
            if _hash_affinity:
                _max_sticky = settings.cache_content_hash_max_sticky
                # Filter to candidates that are in the affinity map and below max_sticky cap
                _hash_preferred = [
                    d for d in tier_candidates
                    if _hash_affinity.get(d.name, 0) > 0
                    and _hash_affinity.get(d.name, 0) <= _max_sticky
                ]
                if _hash_preferred:
                    # Move hash-preferred candidates to the front; rest follow.
                    _rest = [d for d in tier_candidates if d not in _hash_preferred]
                    tier_candidates = _hash_preferred + _rest
                    ctx.affinity_source = "content_hash"
                    log.debug(
                        "cache_l3_content_hash_routing",
                        request_id=ctx.request_id,
                        prefix_hash=ctx.prefix_hash,
                        preferred=[d.name for d in _hash_preferred],
                        sticky_counts=_hash_affinity,
                    )
        except Exception as _exc:
            log.debug("cache_l3_content_hash_skipped", error=str(_exc))

    # ── 6c. Cache switch-cost re-ranking (CACHE_MODE=switch_cost) ───────────
    # Feature-flag: only active when settings.cache_mode == "switch_cost".
    # Adjusts candidate scores by the USD cost of switching away from the warm
    # session model. The candidate list is re-sorted; the cheapest-to-stay
    # candidate moves to the front so Step 7's strategy picks it.
    if settings.cache_mode == "switch_cost" and ctx.session_id and len(tier_candidates) > 1:
        try:
            from model_plane.cache.session import get_session_cache
            from model_plane.cache.switch_cost import cache_score_adjustment

            _TIER_RANK_LOCAL = {"simple": 0, "medium": 1, "complex": 2, "reasoning": 3}
            session_state = get_session_cache().get(ctx.session_id)
            if session_state and session_state.warm_model_ids:
                warm_model = session_state.model
                accum_tokens = session_state.accumulated_tokens
                warm_tier_rank = _TIER_RANK_LOCAL.get(
                    getattr(catalog.get(warm_model), "tier", "simple") if catalog.get(warm_model) else "simple", 0
                )

                def _switch_adj(dep: "Any") -> float:
                    cand_tier_rank = _TIER_RANK_LOCAL.get(dep.tier, 0)
                    tier_gap = cand_tier_rank - warm_tier_rank
                    price = getattr(dep, "input_per_mtok_usd", 0.0) / 1_000_000
                    return cache_score_adjustment(
                        warm_model, dep.name, accum_tokens, price, tier_gap
                    )

                # Sort: candidates with higher (less negative) adjustment first.
                # Tie-break preserves existing order.
                tier_candidates = sorted(
                    tier_candidates,
                    key=lambda d: -_switch_adj(d),
                )
                log.debug(
                    "step6c_switch_cost_rerank",
                    request_id=ctx.request_id,
                    warm_model=warm_model,
                    accumulated_tokens=accum_tokens,
                    ranked=[d.name for d in tier_candidates],
                )
        except Exception as _exc:
            log.debug("step6c_switch_cost_skipped", error=str(_exc))

    # ── 6e. Context-reuse re-rank ────────────────────────────────────────────
    # Boosts candidates that are likely to benefit from an existing KV / prefix
    # cache hit for this session.  The bonus is a soft signal — it adds to the
    # sort key but cannot override the security hard-filter above.
    # Feature-flagged: set context_reuse_enabled=false to disable.
    if settings.context_reuse_enabled and ctx.session_id and len(tier_candidates) > 1:
        try:
            from model_plane.cache.session import get_session_cache
            from model_plane.routing.context_reuse import score_context_reuse
            session_state = get_session_cache().get(ctx.session_id)
            if session_state:
                ctx.turn_index = getattr(session_state, "model_switches", 0)
                reuse_scores: list[tuple[float, Any]] = []
                for dep in tier_candidates:
                    rs = score_context_reuse(
                        session_state,
                        dep,
                        ctx,
                        token_threshold=settings.context_reuse_token_threshold,
                        ttl_seconds=settings.cache_ttl_seconds,
                    )
                    reuse_scores.append((rs, dep))
                # Re-sort: highest (reuse_weight * reuse_score) first, preserving
                # the existing relative order on ties via stable sort.
                w = settings.context_reuse_weight
                tier_candidates = [
                    dep for _, dep in sorted(
                        reuse_scores, key=lambda t: -(w * t[0])
                    )
                ]
                best_reuse = reuse_scores[0][0] if reuse_scores else 0.0
                ctx.context_reuse_score = best_reuse
                log.debug(
                    "step6e_context_reuse",
                    request_id=ctx.request_id,
                    session_id=ctx.session_id,
                    best_reuse_score=round(best_reuse, 3),
                    ranked=[d.name for d in tier_candidates],
                )
        except Exception as _exc:
            log.debug("step6e_context_reuse_skipped", error=str(_exc))

    # ── 6d. TierResolver — ML tier mode selects best deployment from candidates
    # When Stage 3.6 stored a predicted tier (ml_output_mode="tier"), the resolver
    # picks the best deployment from the already-filtered candidate list rather
    # than letting Step 7's general strategy handle it. This gives the ML model
    # explicit control over the tier while keeping policy and override rules intact.
    if ctx.ml_resolved_tier is not None and ctx.routing_source == "ml_active":
        from model_plane.routing.tier_resolver import TierResolver
        resolver = TierResolver()
        resolved_dep = resolver.resolve(ctx.ml_resolved_tier, tier_candidates)
        if resolved_dep is not None:
            ctx.selected_deployment = resolved_dep
            resolved_tier_used = ctx.ml_resolved_tier
            log.info(
                "routing_decision",
                request_id=ctx.request_id,
                deployment=resolved_dep.name,
                model=resolved_dep.litellm_model,
                task=ctx.classification.task_type.value,
                tier=tier.value,
                raw_score=round(ctx.scorer_result.raw_score, 4),
                source="ml_active",
                confidence=round(ctx.ml_recommendation_confidence, 4),
                task_override_used=bool(override_names),
                ml_recommended=ctx.ml_recommended_deployment,
                ml_confidence=round(ctx.ml_recommendation_confidence, 3),
                ml_tier=resolved_tier_used.value,
            )
            return RoutingDecision(
                deployment=resolved_dep,
                source="ml_active",
                confidence=round(ctx.ml_recommendation_confidence, 4),
                reasoning=f"tier_resolver={ctx.ml_resolved_tier.value}",
                ml_tier=ctx.ml_resolved_tier,
                resolved_tier=resolved_tier_used,
            )

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
        security_sensitivity=(
            ctx.security_ctx.data_sensitivity.value if ctx.security_ctx else "disabled"
        ),
        security_blocked=(
            list(ctx.security_ctx.blocked_deployments.keys()) if ctx.security_ctx else []
        ),
        context_reuse_score=round(ctx.context_reuse_score, 3),
        cache_affinity_source=ctx.affinity_source,
    )

    return decision


def _fallback_decision(ctx: RoutingContext, catalog) -> RoutingDecision:
    """Return a healthy, credentialed deployment dynamically chosen from the lowest cost tier."""
    from model_plane.provider_creds import provider_has_creds as _phc

    healthy_candidates = [
        d for d in catalog.all_healthy()
        if _phc(d.provider)
    ]
    if not healthy_candidates:
        raise RuntimeError("No credentialed and healthy deployments available — cannot route request")

    # If security context is active, filter candidates against security constraints
    if ctx.security_ctx:
        try:
            from model_plane.routing.security import get_security_guard
            guard = get_security_guard(owner_id=ctx.tenant_id)
            filtered = guard.filter(healthy_candidates, ctx.security_ctx)
            if filtered:
                healthy_candidates = filtered
        except Exception:
            pass

    # Tier hierarchy: prefer simple -> medium -> complex -> reasoning
    tier_priority = {"simple": 0, "small": 0, "medium": 1, "complex": 2, "large": 2, "reasoning": 3}

    # Sort dynamically by tier priority first, then lowest cost per 1k input
    healthy_candidates.sort(
        key=lambda d: (
            tier_priority.get(getattr(d, "tier", "medium"), 1),
            getattr(d, "cost_per_1k_input", 0.0),
        )
    )

    dep = healthy_candidates[0]
    ctx.fallback_used = True
    ctx.routing_source = "fallback_least_cost"
    return RoutingDecision(
        deployment=dep,
        source="fallback_least_cost",
        confidence=0.5,
        reasoning=f"fallback_to_{dep.name}_tier={getattr(dep, 'tier', 'simple')}",
    )


