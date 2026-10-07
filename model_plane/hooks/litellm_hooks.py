"""LiteLLM pre-call and post-call hooks.

Pre-call hooks:
  - Feature extraction
  - Task classification + scoring
  - Policy enforcement
  - Cache-aware routing
  - Compression

Post-call hooks:
  - Usage/cost/latency extraction
  - Training data recording
  - Metrics emission
"""

from __future__ import annotations

from typing import Any

import litellm

from model_plane.cache.session import get_cache_router
from model_plane.classifier import extract_features
from model_plane.compression.processor import compress
from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.ml.recommender import get_recorder
from model_plane.observability.cost_accumulator import get_cost_accumulator
from model_plane.observability.metrics import record_request
from model_plane.observability.request_buffer import get_request_buffer
from model_plane.routing.context import RoutingContext

log = get_logger(__name__)


class PreCallHook(litellm.CustomLogger):  # type: ignore[misc]
    """
    LiteLLM pre-call hook:
    - injects routing metadata into kwargs
    - applies compression if enabled and threshold exceeded
    """

    def log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        ctx: RoutingContext | None = (
            kwargs.get("metadata", {}).get("__routing_ctx")
            or kwargs.get("__routing_ctx")  # backwards compat
        )
        if ctx is None:
            return

        # Feature extraction (if not already done by pipeline)
        if ctx.features is None:
            ctx.features = extract_features({"messages": messages})

        # Compression
        if (
            settings.compression_enabled
            and ctx.features.total_tokens >= settings.compression_token_threshold
        ):
            task_type_val = ctx.classification.task_type.value if ctx.classification else None
            profile = ctx.compression_profile
            if profile == "passthrough":
                profile = "auto"

            result = compress(
                messages,
                profile=profile,
                task_type=task_type_val,
                context_budget=kwargs.get("max_tokens", 4000),
            )
            ctx.tokens_before_compression = result.tokens_before
            ctx.tokens_after_compression = result.tokens_after
            ctx.compression_profile = result.profile_used

            if result.savings > 0:
                # Mutate messages in-place (LiteLLM passes the list by ref)
                messages.clear()
                messages.extend(result.messages)
                log.info(
                    "compression_applied",
                    request_id=ctx.request_id,
                    profile=result.profile_used,
                    tokens_saved=result.savings,
                    ratio=round(result.compression_ratio, 3),
                )

    async def async_log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        self.log_pre_api_call(model, messages, kwargs)


class PostCallHook(litellm.CustomLogger):  # type: ignore[misc]
    """
    LiteLLM post-call hook:
    - records cost, latency, tokens
    - appends training data
    - updates session cache
    - emits Prometheus metrics
    """

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        ctx: RoutingContext | None = (
            kwargs.get("__routing_ctx")
            or kwargs.get("litellm_params", {}).get("metadata", {}).get("__routing_ctx")
            or kwargs.get("metadata", {}).get("__routing_ctx")
        )
        if ctx is None:
            return

        # LiteLLM passes datetime objects, not floats — handle both
        try:
            import datetime as _dt
            if isinstance(end_time, _dt.datetime) and isinstance(start_time, _dt.datetime):
                ctx.latency_ms = (end_time - start_time).total_seconds() * 1000
            else:
                ctx.latency_ms = (end_time - start_time) * 1000
        except Exception:
            ctx.latency_ms = 0.0

        usage = getattr(response_obj, "usage", None)
        if usage:
            ctx.input_tokens = getattr(usage, "prompt_tokens", 0)
            ctx.output_tokens = getattr(usage, "completion_tokens", 0)

        if ctx.selected_deployment:
            dep = ctx.selected_deployment
            ctx.cost_usd = (
                ctx.input_tokens / 1000.0 * dep.cost_per_1k_input
                + ctx.output_tokens / 1000.0 * dep.cost_per_1k_output
            )

        # Update session cache
        if ctx.session_id and ctx.selected_deployment:
            messages = ctx.raw_request.get("messages", [])
            get_cache_router().update_session(
                session_id=ctx.session_id,
                deployment_name=ctx.selected_deployment.name,
                provider=ctx.selected_deployment.provider,
                messages=messages,
                context_tokens=ctx.input_tokens,
            )

        # Record training data
        get_recorder().record(ctx)

        # Metrics
        record_request(ctx, success=True)

        # ── Admin buffer + cost accumulator (Sub-Task 3.1) ────────────────────
        try:
            dep = ctx.selected_deployment
            task_type_val = (
                ctx.classification.task_type.value
                if ctx.classification else "unknown"
            )
            tier_val = dep.tier if dep else "unknown"
            provider_val = dep.provider if dep else "unknown"
            tenant_val = ctx.tenant_id or ""

            record: dict = {
                "request_id": ctx.request_id,
                "timestamp": __import__("time").time(),
                "deployment": dep.name if dep else None,
                "provider": provider_val,
                "tier": tier_val,
                "task_type": task_type_val,
                "tenant_id": tenant_val,
                "routing_source": ctx.routing_source,
                "routing_confidence": round(ctx.routing_confidence or 0.0, 4),
                "input_tokens": ctx.input_tokens,
                "output_tokens": ctx.output_tokens,
                "latency_ms": round(ctx.latency_ms, 1),
                "cost_usd": round(ctx.cost_usd, 8),
                "ml_recommended": ctx.ml_recommended_deployment,
                "ml_confidence": round(ctx.ml_recommendation_confidence or 0.0, 4),
                "security_sensitivity": (
                    ctx.security_ctx.data_sensitivity.value
                    if ctx.security_ctx else "disabled"
                ),
                "security_blocked": (
                    list(ctx.security_ctx.blocked_deployments.keys())
                    if ctx.security_ctx else []
                ),
                "pii_types": list(ctx.security_ctx.pii_types) if ctx.security_ctx else [],
                "secret_types": list(ctx.security_ctx.secret_types) if ctx.security_ctx else [],
                "context_reuse_score": round(ctx.context_reuse_score, 3),
            }
            get_request_buffer().push(record)
            get_cost_accumulator().record(
                provider=provider_val,
                tier=tier_val,
                task_type=task_type_val,
                tenant_id=tenant_val,
                cost_usd=ctx.cost_usd,
                input_tokens=ctx.input_tokens,
                output_tokens=ctx.output_tokens,
            )
        except Exception as exc:
            log.warning("admin_buffer_push_failed", error=str(exc))

        log.info(
            "request_complete",
            request_id=ctx.request_id,
            deployment=ctx.selected_deployment.name if ctx.selected_deployment else "unknown",
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            latency_ms=round(ctx.latency_ms, 1),
            cost_usd=round(ctx.cost_usd, 6),
        )

    def log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        ctx: RoutingContext | None = (
            kwargs.get("__routing_ctx")
            or kwargs.get("litellm_params", {}).get("metadata", {}).get("__routing_ctx")
            or kwargs.get("metadata", {}).get("__routing_ctx")
        )
        if ctx is None:
            return
        try:
            import datetime as _dt
            if isinstance(end_time, _dt.datetime) and isinstance(start_time, _dt.datetime):
                ctx.latency_ms = (end_time - start_time).total_seconds() * 1000
            else:
                ctx.latency_ms = (end_time - start_time) * 1000
        except Exception:
            ctx.latency_ms = 0.0
        record_request(ctx, success=False)
        # LiteLLM passes the exception in kwargs["exception"]; response_obj is None
        # on failure events.
        exc = kwargs.get("exception") or response_obj
        log.error(
            "request_failed",
            request_id=ctx.request_id,
            latency_ms=round(ctx.latency_ms, 1),
            error=str(exc) if exc is not None else "unknown",
        )

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: float, end_time: float
    ) -> None:
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: float, end_time: float
    ) -> None:
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


# ── register hooks with LiteLLM ────────────────────────────────────────────

_hooks_registered = False


def register_hooks() -> None:
    global _hooks_registered
    if _hooks_registered:
        return
    litellm.callbacks = [PreCallHook(), PostCallHook()]
    _hooks_registered = True
    log.info("litellm_hooks_registered")
