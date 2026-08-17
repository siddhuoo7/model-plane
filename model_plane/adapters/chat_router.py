"""OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, AsyncIterator

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_plane.adapters.auth import verify_api_key
from model_plane.adapters.executor import (
    aexecute,
    aexecute_with_fallback,
    maybe_fire_shadow,
    stream_response,
)
from model_plane.cache.session import get_cache_router
from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["completions"])


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    _auth: Annotated[str, Depends(verify_api_key)],
) -> Any:
    body: dict = await request.json()
    request_id = f"mp-{uuid.uuid4().hex[:12]}"

    # Headers → routing metadata
    tenant_id = request.headers.get("X-Tenant-Id")
    session_id = request.headers.get("X-Session-Id")
    agent_type = request.headers.get("X-Agent-Type")

    ctx = build_routing_context(
        body,
        request_id=request_id,
        tenant_id=tenant_id,
        session_id=session_id,
        agent_type=agent_type,
    )

    # ── routing ──────────────────────────────────────────────────────────────
    try:
        decision = run_routing_pipeline(ctx)
    except Exception as exc:
        log.error("routing_pipeline_failed", error=str(exc))
        raise HTTPException(status_code=503, detail="Routing pipeline failed") from exc

    # ── cache-aware routing adjustment ───────────────────────────────────────
    if session_id:
        messages = body.get("messages", [])
        anchor, cache_warm = get_cache_router().evaluate(session_id, messages, decision.deployment)
        ctx.cache_warm = cache_warm
        if anchor and anchor != decision.deployment.name:
            from model_plane.registry.catalog import get_catalog
            anchored_dep = get_catalog().get(anchor)
            if anchored_dep and anchored_dep.healthy:
                ctx.selected_deployment = anchored_dep
                decision = decision.__class__(
                    deployment=anchored_dep,
                    source="cache_anchor",
                    confidence=0.9,
                    reasoning="cache_warm_anchor",
                )

    # ── streaming vs non-streaming ──────────────────────────────────────────
    if body.get("stream", False):
        return StreamingResponse(
            _stream_chunks(ctx, request_id),
            media_type="text/event-stream",
            headers={"X-Request-Id": request_id, "X-Deployment": decision.deployment.name},
        )

    # ── non-streaming execution with validation + escalation ─────────────────
    routing_config = settings.load_routing_config()
    escalation_cfg = routing_config.get("escalation", {})
    fallback_chain = decision.deployment.fallback_to or []
    escalation_chain = escalation_cfg.get("fallback_chain", [])
    max_retries = int(escalation_cfg.get("max_retries", 2))
    escalate_on_schema = escalation_cfg.get("escalate_on_schema_fail", True)
    escalate_on_tool = escalation_cfg.get("escalate_on_tool_fail", True)

    expect_json = (body.get("response_format", {}) or {}).get("type") == "json_object"
    has_tools = bool(body.get("tools"))

    try:
        response = await aexecute_with_fallback(ctx, fallback_chain)
    except Exception as exc:
        log.error("execution_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=502, detail=f"Model execution failed: {exc}") from exc

    # ── validation + escalation loop ─────────────────────────────────────────
    if expect_json or has_tools:
        from model_plane.registry.catalog import get_catalog
        from model_plane.validation.validator import EscalationPolicy, validate_response

        policy = EscalationPolicy(
            max_retries=max_retries,
            escalate_on_schema_fail=escalate_on_schema,
            escalate_on_tool_fail=escalate_on_tool,
            fallback_deployments=escalation_chain,
        )
        catalog = get_catalog()

        for attempt in range(max_retries + 1):
            content = _extract_content(response)
            vr = validate_response(
                content,
                expect_json=expect_json,
                expect_tool_call=has_tools and not expect_json,
            )
            ctx.validation_result = vr.result_code

            if vr.passed:
                break

            next_dep_name = policy.next_fallback(attempt)
            if not next_dep_name or not policy.should_escalate(vr, attempt):
                log.warning(
                    "validation_failed_no_escalation",
                    request_id=request_id,
                    result_code=vr.result_code,
                    attempt=attempt,
                )
                break

            next_dep = catalog.get(next_dep_name)
            if not next_dep or not next_dep.healthy:
                break

            log.info(
                "escalating_to_fallback",
                request_id=request_id,
                validation_code=vr.result_code,
                from_dep=ctx.selected_deployment.name if ctx.selected_deployment else "?",
                to_dep=next_dep_name,
                attempt=attempt + 1,
            )
            ctx.selected_deployment = next_dep
            ctx.escalated = True
            try:
                response = await aexecute(ctx)
            except Exception as exc:
                log.warning("escalation_attempt_failed", dep=next_dep_name, error=str(exc))
                break

    # ── shadow traffic (fire-and-forget A/B call) ────────────────────────────
    maybe_fire_shadow(ctx)

    # ── update similarity index (fire-and-forget) ────────────────────────────
    _update_similarity_index(ctx)

    resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    return _inject_routing_headers(resp_dict, ctx)


async def _stream_chunks(ctx, request_id: str) -> AsyncIterator[bytes]:
    """Yield SSE chunks from the streaming response."""
    try:
        async for chunk in stream_response(ctx):
            data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
            yield b"data: " + orjson.dumps(data) + b"\n\n"
    except Exception as exc:
        log.error("stream_error", request_id=request_id, error=str(exc))
        yield b"data: [DONE]\n\n"
    else:
        yield b"data: [DONE]\n\n"


def _extract_content(response: Any) -> str | None:
    """Pull text content from a LiteLLM response object."""
    try:
        return response.choices[0].message.content
    except Exception:
        return None


def _update_similarity_index(ctx: Any) -> None:
    """Non-blocking: add this request to the similarity index for future routing."""
    try:
        if not settings.similarity_routing_enabled:
            return
        from model_plane.routing.similarity_router import get_similarity_router
        sim_router = get_similarity_router()
        if sim_router and ctx.features and ctx.selected_deployment:
            task = ctx.classification.task_type.value if ctx.classification else "unknown"
            # success_score: 1.0 if no escalation + validation ok, 0.5 otherwise
            success = 1.0 if (not ctx.escalated and ctx.validation_result in (None, "ok")) else 0.5
            sim_router.add_entry(
                prompt_text=ctx.features.last_user_text,
                deployment_name=ctx.selected_deployment.name,
                task_type=task,
                success_score=success,
            )
    except Exception as exc:
        log.debug("similarity_index_update_failed", error=str(exc))


def _inject_routing_headers(resp: dict, ctx: Any) -> JSONResponse:
    """Inject routing metadata + cost into response headers AND body."""
    dep = ctx.selected_deployment
    task = ctx.classification.task_type.value if ctx.classification else ""

    # ── cost calculation ─────────────────────────────────────────────────────
    # ctx.cost_usd is set by PostCallHook; if hook hasn't fired yet (rare),
    # compute it here from the response usage.
    cost_usd = ctx.cost_usd
    if cost_usd == 0.0 and dep:
        usage = resp.get("usage", {}) or {}
        input_tok = usage.get("prompt_tokens", 0)
        output_tok = usage.get("completion_tokens", 0)
        cost_usd = (
            input_tok / 1000.0 * dep.cost_per_1k_input
            + output_tok / 1000.0 * dep.cost_per_1k_output
        )

    headers = {
        "X-Request-Id": ctx.request_id,
        "X-Deployment": dep.name if dep else "",
        "X-Task-Type": task,
        "X-Routing-Source": ctx.routing_source,
        "X-Cost-USD": str(round(cost_usd, 8)),
        "X-Routing-Confidence": str(round(ctx.routing_confidence, 4)),
        "X-Cache-Warm": str(ctx.cache_warm).lower(),
        "X-Escalated": str(ctx.escalated).lower(),
    }

    # Also embed routing metadata inside the response body under "x_model_plane"
    resp["x_model_plane"] = {
        "request_id": ctx.request_id,
        "deployment": dep.name if dep else None,
        "litellm_model": dep.litellm_model if dep else None,
        "provider": dep.provider if dep else None,
        "task_type": task,
        "routing_source": ctx.routing_source,
        "routing_confidence": round(ctx.routing_confidence, 4),
        "cost_usd": round(cost_usd, 8),
        "cache_warm": ctx.cache_warm,
        "escalated": ctx.escalated,
        "fallback_used": ctx.fallback_used,
        "validation_result": ctx.validation_result,
        "ml_recommendation": ctx.ml_recommended_deployment,
        "tokens_saved_by_compression": max(
            0, ctx.tokens_before_compression - ctx.tokens_after_compression
        ),
    }

    return JSONResponse(content=resp, headers=headers)
