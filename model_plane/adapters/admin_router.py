"""Admin and introspection endpoints.

GET  /v1/models              — list available deployments
GET  /v1/routing/status      — routing engine status
POST /v1/routing/explain     — dry-run routing trace (no LLM call)
POST /v1/routing/feedback    — submit user feedback to improve routing
GET  /v1/routing/metrics     — live JSON dashboard from training CSV
POST /v1/admin/reload        — reload model catalog
POST /v1/admin/retrain       — trigger offline ML retrain
GET  /healthz                — liveness probe
GET  /readyz                 — readiness probe
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from model_plane.adapters.auth import verify_api_key
from model_plane.config import settings
from model_plane.registry.catalog import get_catalog, reload_catalog
from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline
from model_plane.scorer import DEFAULT_WEIGHTS  # noqa: F401 – exposes scorer config

router = APIRouter(tags=["admin"])


class FeedbackRequest(BaseModel):
    """User feedback on a routing decision — drives both the similarity index
    and appends a labelled row to the ML training CSV."""
    request_id: str
    prompt: str                          # original user message text
    correct_deployment: str              # deployment name the user thinks was right
    routing_was_correct: bool            # was the router's choice acceptable?
    task_type: str | None = None         # optional hint
    quality_score: float | None = None   # 0.0–1.0, optional subjective score


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict:
    catalog = get_catalog()
    healthy = len(catalog.all_healthy())
    if healthy == 0:
        return Response(status_code=503, content="No healthy deployments")
    return {"status": "ok", "healthy_deployments": healthy}


@router.get("/v1/models")
async def list_models() -> dict:
    catalog = get_catalog()
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "owned_by": dep.provider,
                "tier": dep.tier,
                "capabilities": dep.capabilities,
                "context_limit": dep.context_limit,
                "healthy": dep.healthy,
            }
            for name, dep in catalog.deployments.items()
        ],
    }


@router.post("/v1/admin/reload")
async def reload() -> dict:
    cat = reload_catalog()
    return {"status": "ok", "deployments": len(cat.deployments)}


@router.get("/v1/routing/status")
async def routing_status() -> dict:
    catalog = get_catalog()
    return {
        "routing_mode": settings.routing_mode,
        "ml_enabled": settings.ml_routing_enabled,
        "compression_enabled": settings.compression_enabled,
        "total_deployments": len(catalog.deployments),
        "healthy_deployments": len(catalog.all_healthy()),
        "default_model": settings.default_model,
    }


@router.post("/v1/routing/explain")
async def routing_explain(
    request: Request,
    _auth: Annotated[str, Depends(verify_api_key)],
) -> dict[str, Any]:
    """Dry-run the routing pipeline and return the full decision trace.

    Accepts the same JSON body as ``POST /v1/chat/completions`` (or a subset
    with just ``messages``). No LLM call is made — the response shows exactly
    which deployment would be selected and why.

    Optional headers (same as the real endpoint):
      X-Tenant-Id, X-Session-Id, X-Agent-Type
    """
    import uuid

    body: dict = await request.json()
    if not body.get("messages"):
        raise HTTPException(status_code=422, detail="'messages' field is required")

    request_id = f"explain-{uuid.uuid4().hex[:8]}"

    ctx = build_routing_context(
        body,
        request_id=request_id,
        tenant_id=request.headers.get("X-Tenant-Id"),
        session_id=request.headers.get("X-Session-Id"),
        agent_type=request.headers.get("X-Agent-Type"),
    )

    try:
        decision = run_routing_pipeline(ctx)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Routing pipeline error: {exc}") from exc

    # Build structured trace —– all routing artefacts, no model response
    features = ctx.features
    classification = ctx.classification
    scorer = ctx.scorer_result

    return {
        "request_id": request_id,
        "dry_run": True,
        # ── selected deployment ──────────────────────────────────────────────
        "selected_deployment": decision.deployment.name,
        "litellm_model": decision.deployment.litellm_model,
        "provider": decision.deployment.provider,
        "tier": decision.deployment.tier,
        "routing_source": decision.source,
        "routing_confidence": round(decision.confidence, 4),
        "task_override_used": bool(
            (settings.load_routing_config().get("task_deployment_overrides", {}))
            .get(classification.task_type.value if classification else "", {})
            .get("preferred_deployments")
        ),
        "reasoning": decision.reasoning,
        # ── classification ───────────────────────────────────────────────────
        "classification": {
            "task_type": classification.task_type.value if classification else None,
            "complexity_tier": classification.complexity_tier.value if classification else None,
            "confidence": round(classification.confidence, 4) if classification else None,
            "top_signals": (
                {k: round(v, 3) for k, v in classification.signals.items() if v > 0}
                if classification else {}
            ),
        },
        # ── 14-dimension scorer ──────────────────────────────────────────────
        "scorer": {
            "raw_score": round(scorer.raw_score, 4) if scorer else None,
            "tier": scorer.tier.value if scorer else None,
            "confidence": round(scorer.confidence, 4) if scorer else None,
            "dimensions": (
                {k: round(v, 4) for k, v in scorer.dimension_scores.items()}
                if scorer else {}
            ),
        },
        # ── features ────────────────────────────────────────────────────────
        "features": {
            "total_tokens": features.total_tokens if features else None,
            "user_message_count": features.user_message_count if features else None,
            "has_tools": features.has_tools if features else None,
            "language_hint": features.language_hint if features else None,
            "reasoning_markers": round(features.reasoning_markers, 4) if features else None,
            "code_presence": round(features.code_presence, 4) if features else None,
            "multi_step_patterns": round(features.multi_step_patterns, 4) if features else None,
            "technical_terms": round(features.technical_terms, 4) if features else None,
        },
        # ── candidates that survived policy + tier filter ────────────────────
        "candidate_deployments": [d.name for d in ctx.candidate_deployments],
        # ── fallback info ────────────────────────────────────────────────────
        "fallback_chain": decision.deployment.fallback_to or [],
        "fallback_used": ctx.fallback_used,
        # ── cost estimate (no tokens consumed — uses per-1k rates × 0 tokens) ─
        # For explain we show the cost RATE so callers can estimate.
        "cost_rates": {
            "cost_per_1k_input_usd": decision.deployment.cost_per_1k_input,
            "cost_per_1k_output_usd": decision.deployment.cost_per_1k_output,
            "estimated_input_cost_usd": round(
                (features.total_tokens if features else 0) / 1000.0
                * decision.deployment.cost_per_1k_input, 8
            ),
        },
        # ── ml ───────────────────────────────────────────────────────────────
        "ml_recommendation": {
            "enabled": settings.ml_routing_enabled,
            "active": settings.ml_routing_enabled,  # now active, not shadow
            "recommended_deployment": ctx.ml_recommended_deployment,
            "confidence": ctx.ml_recommendation_confidence,
            "matches_selected": (
                ctx.ml_recommended_deployment == decision.deployment.name
                if ctx.ml_recommended_deployment else None
            ),
        },
        # ── similarity routing ───────────────────────────────────────────────
        "similarity_routing": {
            "enabled": settings.similarity_routing_enabled,
            "index_size": _similarity_index_size(),
        },
    }


def _similarity_index_size() -> int:
    try:
        from model_plane.routing.similarity_router import get_similarity_router
        r = get_similarity_router()
        return len(r._entries) if r else 0
    except Exception:
        return 0


@router.post("/v1/routing/feedback")
async def routing_feedback(
    feedback: FeedbackRequest,
    _auth: Annotated[str, Depends(verify_api_key)],
) -> dict[str, Any]:
    """Submit user feedback on a routing decision.

    This drives two learning mechanisms:
    1. Similarity index — adjusts future cosine-similarity votes
    2. Training CSV   — appends a labelled row for offline ML retraining

    Body::

        {
          "request_id": "mp-abc123",
          "prompt": "Write a Python class for a BST...",
          "correct_deployment": "watsonx-granite-large",
          "routing_was_correct": false,
          "task_type": "code_generation",
          "quality_score": 0.2
        }
    """
    catalog = get_catalog()

    # Validate deployment name
    dep = catalog.get(feedback.correct_deployment)
    if dep is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown deployment '{feedback.correct_deployment}'. "
                   f"Valid names: {list(catalog.deployments.keys())}",
        )

    results: dict[str, Any] = {
        "request_id": feedback.request_id,
        "correct_deployment": feedback.correct_deployment,
        "similarity_entries_updated": 0,
        "training_row_appended": False,
    }

    # ── 1. Update similarity index ────────────────────────────────────────────
    if settings.similarity_routing_enabled:
        try:
            from model_plane.routing.similarity_router import get_similarity_router
            sim_router = get_similarity_router()
            if sim_router and feedback.prompt:
                if feedback.routing_was_correct:
                    # Reinforce: add a new entry with full success score
                    sim_router.add_entry(
                        prompt_text=feedback.prompt,
                        deployment_name=feedback.correct_deployment,
                        task_type=feedback.task_type or "unknown",
                        success_score=1.0,
                    )
                    results["similarity_entries_updated"] = 1
                else:
                    # Correct: update the most similar entry with the right label
                    updated = sim_router.apply_feedback(
                        prompt_text=feedback.prompt,
                        correct_deployment=feedback.correct_deployment,
                    )
                    # Also add a new positive entry for the correct deployment
                    sim_router.add_entry(
                        prompt_text=feedback.prompt,
                        deployment_name=feedback.correct_deployment,
                        task_type=feedback.task_type or "unknown",
                        success_score=1.0,
                    )
                    results["similarity_entries_updated"] = updated + 1
        except Exception as exc:
            results["similarity_error"] = str(exc)

    # ── 2. Append to training CSV ─────────────────────────────────────────────
    try:
        import csv
        import threading
        from datetime import datetime, timezone
        from pathlib import Path as _Path

        csv_path = _Path("model_plane/data/training_data.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure header exists
        header = [
            "timestamp", "request_id", "task_type", "tier",
            "reasoning_markers", "code_presence", "simple_indicators",
            "multi_step_patterns", "technical_terms", "token_count_signal",
            "creative_markers", "question_complexity", "constraint_count",
            "imperative_verbs", "output_format", "domain_specificity",
            "reference_complexity", "negation_complexity",
            "selected_deployment", "latency_ms", "cost_usd",
            "validation_result", "fallback_used", "final_success",
            "feedback_source", "quality_score",
        ]
        write_header = not csv_path.exists()

        # Extract basic features from the prompt for the feature columns
        from model_plane.classifier.features import extract_features
        feats = extract_features({"messages": [{"role": "user", "content": feedback.prompt}]})

        row = [
            datetime.now(tz=timezone.utc).isoformat(),
            feedback.request_id,
            feedback.task_type or "unknown",
            dep.tier,
            feats.reasoning_markers, feats.code_presence, feats.simple_indicators,
            feats.multi_step_patterns, feats.technical_terms, feats.token_count_signal,
            feats.creative_markers, feats.question_complexity, feats.constraint_count,
            feats.imperative_verbs, feats.output_format, feats.domain_specificity,
            feats.reference_complexity, feats.negation_complexity,
            feedback.correct_deployment,
            0.0,   # latency unknown from feedback
            0.0,   # cost unknown from feedback
            "ok",
            False,
            1.0 if feedback.routing_was_correct else 0.0,  # final_success
            "user_feedback",
            feedback.quality_score if feedback.quality_score is not None else "",
        ]

        with threading.Lock():
            with open(csv_path, "a", newline="") as fh:
                w = csv.writer(fh)
                if write_header:
                    w.writerow(header)
                w.writerow(row)

        results["training_row_appended"] = True
    except Exception as exc:
        results["training_error"] = str(exc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Routing metrics dashboard
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/v1/routing/metrics")
async def routing_metrics() -> dict[str, Any]:
    """Live routing quality dashboard computed from the training CSV.

    Returns aggregated stats across all recorded requests:
    - request counts by deployment / task / tier / routing_source
    - mean latency and cost per deployment
    - validation failure rate
    - escalation rate
    - fallback rate

    No authentication required so this can be scraped by Prometheus sidecars.
    """
    import csv
    from collections import defaultdict
    from pathlib import Path as _Path

    csv_path = _Path("model_plane/data/training_data.csv")
    if not csv_path.exists():
        return {"status": "no_data", "message": "Training CSV not found"}

    rows: list[dict] = []
    try:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    total = len(rows)
    if total == 0:
        return {"status": "empty", "total_requests": 0}

    # ── per-deployment aggregation ─────────────────────────────────────────
    dep_latency: dict[str, list[float]] = defaultdict(list)
    dep_cost: dict[str, list[float]] = defaultdict(list)
    dep_count: dict[str, int] = defaultdict(int)
    task_count: dict[str, int] = defaultdict(int)
    tier_count: dict[str, int] = defaultdict(int)
    source_count: dict[str, int] = defaultdict(int)
    validation_fails = 0
    escalations = 0
    fallbacks = 0
    successes = 0

    for row in rows:
        dep = row.get("selected_deployment", "unknown")
        dep_count[dep] += 1

        task_count[row.get("task_type", "unknown")] += 1
        tier_count[row.get("tier", "unknown")] += 1
        source_count[row.get("routing_source", "unknown")] += 1

        try:
            lat = float(row.get("latency_ms") or 0)
            if lat > 0:
                dep_latency[dep].append(lat)
        except ValueError:
            pass

        try:
            cost = float(row.get("cost_usd") or 0)
            if cost > 0:
                dep_cost[dep].append(cost)
        except ValueError:
            pass

        val = row.get("validation_result", "ok")
        if val and val not in ("ok", "", "pass"):
            validation_fails += 1

        if str(row.get("fallback_used", "")).lower() in ("true", "1"):
            fallbacks += 1

        if str(row.get("escalated", "")).lower() in ("true", "1"):
            escalations += 1

        fs = row.get("final_success", "")
        if str(fs).lower() in ("true", "1", "1.0"):
            successes += 1

    # ── build response ────────────────────────────────────────────────────
    by_deployment = {}
    for dep in dep_count:
        latencies = dep_latency.get(dep, [])
        costs = dep_cost.get(dep, [])
        by_deployment[dep] = {
            "requests": dep_count[dep],
            "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "mean_cost_usd": round(sum(costs) / len(costs), 8) if costs else None,
            "total_cost_usd": round(sum(costs), 6) if costs else 0.0,
        }

    success_rows = [r for r in rows if r.get("final_success", "") not in ("", None)]
    success_rate = (
        successes / len(success_rows) if success_rows else None
    )

    return {
        "status": "ok",
        "total_requests": total,
        "success_rate": round(success_rate, 4) if success_rate is not None else None,
        "validation_failure_rate": round(validation_fails / total, 4),
        "escalation_rate": round(escalations / total, 4),
        "fallback_rate": round(fallbacks / total, 4),
        "by_deployment": by_deployment,
        "by_task_type": dict(task_count),
        "by_tier": dict(tier_count),
        "by_routing_source": dict(source_count),
        "csv_rows": total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Offline retrain trigger
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/v1/admin/retrain")
async def trigger_retrain(
    _auth: Annotated[str, Depends(verify_api_key)],
) -> dict[str, Any]:
    """Trigger an offline ML model retrain from the current training CSV.

    Runs in a background thread to avoid blocking the event loop.
    Returns immediately with a job ID; check /v1/routing/status for completion.
    """
    import threading
    import uuid as _uuid

    job_id = f"retrain-{_uuid.uuid4().hex[:8]}"

    def _retrain_worker() -> None:
        try:
            from pathlib import Path as _Path

            from model_plane.ml.trainer import train_model
            data_path = _Path("model_plane/data/training_data.csv")
            output_path = settings.ml_model_path

            if not data_path.exists():
                import logging
                logging.getLogger(__name__).warning(
                    "retrain_skipped: no training data at %s", data_path
                )
                return

            metrics = train_model(
                data_path=data_path,
                output_path=output_path,
            )

            # Reload the model in-process
            if settings.ml_routing_enabled:
                from model_plane.ml.recommender import get_recommender
                rec = get_recommender()
                if rec:
                    rec.reload()

            import structlog
            structlog.get_logger(__name__).info(
                "retrain_complete",
                job_id=job_id,
                metrics=metrics,
            )
        except Exception as exc:
            import structlog
            structlog.get_logger(__name__).error(
                "retrain_failed",
                job_id=job_id,
                error=str(exc),
            )

    # Run retrain in a daemon thread so the event loop stays free
    t = threading.Thread(target=_retrain_worker, daemon=True, name=f"retrain-{job_id}")
    t.start()

    return {
        "status": "started",
        "job_id": job_id,
        "message": (
            "Retrain running in background. "
            "The in-process ML model will be hot-swapped on completion."
        ),
        "data_path": "model_plane/data/training_data.csv",
        "output_path": str(settings.ml_model_path),
    }
