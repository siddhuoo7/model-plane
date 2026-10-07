"""LiteLLM execution layer — wraps litellm.completion / acompletion.

Responsibilities:
- Translate RoutingContext into a litellm call.
- Inject __routing_ctx so hooks can access it.
- Handle retries / fallback on exception.
- Shadow traffic: fire-and-forget parallel call to a shadow deployment.
- Return the raw LiteLLM response.
"""

from __future__ import annotations

import asyncio
import os
import random
import ssl
from typing import Any, AsyncIterator

import httpx
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from model_plane.adapters.providers import get_adapter
from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.routing.context import RoutingContext

log = get_logger(__name__)


# Providers whose model strings must carry a "provider/" prefix for LiteLLM
# to resolve the correct backend.  Bare model names like "gpt-4o" or "o1-mini"
# are rejected with "LLM Provider NOT provided" unless prefixed.
_PROVIDERS_NEEDING_PREFIX: frozenset[str] = frozenset({"openai", "anthropic"})


def _resolve_litellm_model(dep: "DeploymentConfig") -> str:  # noqa: F821
    """Return the model string to pass to LiteLLM.

    Bare model names (e.g. ``gpt-4o``, ``o1-mini``) are ambiguous to LiteLLM;
    they must be prefixed with ``<provider>/`` so LiteLLM can select the right
    integration. Models that already carry a slash prefix (e.g.
    ``anthropic/claude-3-haiku-…``, ``watsonx/ibm/…``) are passed through
    unchanged.
    """
    model = dep.litellm_model
    if dep.provider in _PROVIDERS_NEEDING_PREFIX and "/" not in model:
        model = f"{dep.provider}/{model}"
    return model


def _build_litellm_kwargs(ctx: RoutingContext, extra: dict) -> dict:
    dep = ctx.selected_deployment
    assert dep is not None, "No deployment selected before execution"

    # Use compression-modified messages if available
    messages = ctx.compressed_request.get("messages") if ctx.compressed_request else None
    if messages is None:
        messages = ctx.raw_request.get("messages", [])

    kwargs: dict[str, Any] = {
        "model": _resolve_litellm_model(dep),
        "messages": messages,
        # Store the RoutingContext in LiteLLM's metadata dict — it is passed
        # through to hooks as kwargs["litellm_params"]["metadata"] and is
        # never serialized into the provider request payload.
        "metadata": {"__routing_ctx": ctx},
    }

    # Forward request-level params
    for param in ("temperature", "max_tokens", "stream", "tools", "tool_choice",
                  "response_format", "stop", "top_p", "frequency_penalty",
                  "presence_penalty", "seed", "user"):
        if param in ctx.raw_request:
            kwargs[param] = ctx.raw_request[param]

    kwargs.update(extra)

    # Provider-specific API configuration
    # NOTE: Do NOT set litellm.ssl_verify or litellm.ssl_certificate here.
    # - litellm.ssl_certificate is a CLIENT cert (mTLS) — setting it to a CA file breaks httpx.
    # - litellm.ssl_verify with a custom CA replaces certifi entirely, breaking all public
    #   providers (watsonx, OpenAI, Anthropic, Bedrock) whose certs aren't in that CA file.
    # The OpenShift CA (REQUESTS_CA_BUNDLE) is applied only to local vLLM calls via the
    # dedicated httpx.AsyncClient in _aexecute_local_vllm (verify=ssl_ca_bundle below).
    if dep.api_base:
        kwargs["api_base"] = dep.api_base
    if dep.api_key_env:
        api_key = os.environ.get(dep.api_key_env)
        if api_key:
            kwargs["api_key"] = api_key

    # Delegate provider-specific auth/param injection to the adapter registry.
    # Each adapter returns a dict that is merged into kwargs; unknown providers
    # return {} via DefaultAdapter so no if/elif chain is needed here.
    provider_kwargs = get_adapter(dep.provider).build_kwargs(dep)
    # _vllm_headers is an internal sentinel used by _aexecute_local_vllm only;
    # it must not be forwarded to LiteLLM.
    provider_kwargs.pop("_vllm_headers", None)
    kwargs.update(provider_kwargs)

    return kwargs


async def _aexecute_local_vllm(ctx: RoutingContext, **extra: Any) -> Any:
    dep = ctx.selected_deployment
    assert dep is not None, "No deployment selected before execution"

    messages = ctx.compressed_request.get("messages") if ctx.compressed_request else None
    if messages is None:
        messages = ctx.raw_request.get("messages", [])

    payload: dict[str, Any] = {
        "model": dep.litellm_model.split("/", 1)[1],
        "messages": messages,
    }

    for param in ("temperature", "max_tokens", "stream", "tools", "tool_choice",
                  "response_format", "stop", "top_p", "frequency_penalty",
                  "presence_penalty", "seed", "user"):
        if param in ctx.raw_request:
            payload[param] = ctx.raw_request[param]
    payload.update(extra)

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if dep.api_key_env:
        api_key = settings.local_vllm_api_key or os.environ.get(dep.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    local_vllm_cookie = os.environ.get("LOCAL_VLLM_COOKIE")
    if local_vllm_cookie:
        headers["Cookie"] = local_vllm_cookie

    insecure_skip_verify = settings.local_vllm_insecure_skip_verify
    ssl_cert_file = settings.ssl_cert_file or settings.requests_ca_bundle
    verify: str | bool = False if insecure_skip_verify else True
    if not insecure_skip_verify and ssl_cert_file and os.path.exists(ssl_cert_file):
        verify = ssl_cert_file

    log.debug(
        "local_vllm_tls_config",
        deployment=dep.name,
        insecure_skip_verify=insecure_skip_verify,
        ssl_cert_file=ssl_cert_file,
        ssl_cert_exists=bool(ssl_cert_file and os.path.exists(ssl_cert_file)),
        requests_ca_bundle=settings.requests_ca_bundle,
        ssl_cert_file_env=settings.ssl_cert_file,
        verify_mode=("disabled" if verify is False else verify),
    )

    if not dep.api_base:
        raise RuntimeError("Local vLLM deployment missing api_base")

    async with httpx.AsyncClient(verify=verify, timeout=600.0) as client:
        response = await client.post(f"{dep.api_base}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def _push_local_vllm_to_buffer(ctx: RoutingContext, response: Any, start_time: float, end_time: float) -> None:
    """Manually push a local-vLLM call into the admin buffer and cost accumulator.

    Local vLLM calls bypass LiteLLM, so the PostCallHook never fires for them.
    This function replicates the same record that the hook would produce.
    """
    try:
        import time as _time
        from model_plane.observability.cost_accumulator import get_cost_accumulator
        from model_plane.observability.request_buffer import get_request_buffer

        dep = ctx.selected_deployment
        ctx.latency_ms = (end_time - start_time) * 1000

        # Extract token usage from the raw dict response
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        ctx.input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        ctx.output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        if dep:
            ctx.cost_usd = (
                ctx.input_tokens / 1000.0 * dep.cost_per_1k_input
                + ctx.output_tokens / 1000.0 * dep.cost_per_1k_output
            )

        task_type_val = ctx.classification.task_type.value if ctx.classification else "unknown"
        tier_val = dep.tier if dep else "unknown"
        provider_val = dep.provider if dep else "local"
        tenant_val = ctx.tenant_id or ""

        record: dict = {
            "request_id": ctx.request_id,
            "timestamp": _time.time(),
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
        log.info(
            "request_complete",
            request_id=ctx.request_id,
            deployment=dep.name if dep else "unknown",
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            latency_ms=round(ctx.latency_ms, 1),
            cost_usd=round(ctx.cost_usd, 6),
        )
    except Exception as exc:
        log.warning("local_vllm_buffer_push_failed", error=str(exc))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
)
async def aexecute(ctx: RoutingContext, **extra: Any) -> Any:
    """Async LiteLLM call with retry."""
    import time as _time

    dep = ctx.selected_deployment
    assert dep is not None, "No deployment selected before execution"

    if dep.provider == "local":
        log.debug("local_vllm_aexecute", request_id=ctx.request_id, deployment=dep.name)
        t0 = _time.monotonic()
        try:
            result = await _aexecute_local_vllm(ctx, **extra)
            _push_local_vllm_to_buffer(ctx, result, t0, _time.monotonic())
            return result
        except httpx.HTTPStatusError as exc:
            log.error("local_vllm_http_error", deployment=dep.name, status_code=exc.response.status_code, error=exc.response.text)
            raise
        except Exception as exc:
            log.warning("local_vllm_transient_error", deployment=dep.name, error=str(exc))
            raise

    kwargs = _build_litellm_kwargs(ctx, extra)
    stream = kwargs.get("stream", False)

    log.debug("litellm_aexecute", request_id=ctx.request_id, model=kwargs["model"], stream=stream)

    try:
        if stream:
            return await litellm.acompletion(**kwargs)
        return await litellm.acompletion(**kwargs)
    except litellm.exceptions.AuthenticationError as exc:
        log.error("litellm_auth_error", model=kwargs["model"], error=str(exc))
        raise
    except litellm.exceptions.BadRequestError as exc:
        log.error("litellm_bad_request", model=kwargs["model"], error=str(exc))
        raise
    except Exception as exc:
        log.warning("litellm_transient_error", model=kwargs["model"], error=str(exc))
        raise


async def aexecute_with_fallback(ctx: RoutingContext, fallback_names: list[str], **extra: Any) -> Any:
    """Try primary deployment, then each fallback in order."""
    from model_plane.registry.catalog import get_catalog

    catalog = get_catalog()
    original_dep = ctx.selected_deployment

    try:
        return await aexecute(ctx, **extra)
    except Exception as primary_exc:
        log.warning("primary_deployment_failed", deployment=original_dep.name if original_dep else "?", error=str(primary_exc))

    for name in fallback_names:
        dep = catalog.get(name)
        if dep and dep.healthy:
            from model_plane.provider_creds import provider_has_creds
            if not provider_has_creds(dep.provider):
                log.debug("fallback_skipped_no_creds", deployment=name, provider=dep.provider)
                continue
            ctx.selected_deployment = dep
            ctx.fallback_used = True
            log.info("trying_fallback_deployment", deployment=name)
            try:
                return await aexecute(ctx, **extra)
            except Exception as fb_exc:
                log.warning("fallback_deployment_failed", deployment=name, error=str(fb_exc))
                catalog.mark_unhealthy(name)

    raise RuntimeError("All deployments exhausted during fallback chain")


async def _shadow_call(ctx: RoutingContext, shadow_dep_name: str) -> None:
    """Fire-and-forget shadow execution for A/B data collection.

    Runs completely in the background; any exception is swallowed and logged.
    The shadow response is NOT returned to the caller.
    """
    from model_plane.registry.catalog import get_catalog

    catalog = get_catalog()
    shadow_dep = catalog.get(shadow_dep_name)
    if not shadow_dep or not shadow_dep.healthy:
        return

    import copy
    import uuid

    shadow_ctx = copy.copy(ctx)
    shadow_ctx.request_id = f"shadow-{uuid.uuid4().hex[:8]}"
    shadow_ctx.selected_deployment = shadow_dep

    try:
        response = await aexecute(shadow_ctx)
        usage = getattr(response, "usage", None)
        log.debug(
            "shadow_call_complete",
            shadow_id=shadow_ctx.request_id,
            shadow_deployment=shadow_dep_name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )
    except Exception as exc:
        log.debug("shadow_call_failed", shadow_deployment=shadow_dep_name, error=str(exc))


def maybe_fire_shadow(ctx: RoutingContext) -> None:
    """Probabilistically launch a shadow call based on shadow_traffic_fraction.

    Call this AFTER the primary response is received so the shadow runs
    concurrently with response serialisation, not on the critical path.
    """
    fraction = settings.shadow_traffic_fraction
    if fraction <= 0.0:
        return

    if random.random() > fraction:
        return

    from model_plane.registry.catalog import get_catalog

    catalog = get_catalog()
    # Pick a random healthy deployment that is NOT the currently selected one
    current_name = ctx.selected_deployment.name if ctx.selected_deployment else ""
    candidates = [d for d in catalog.all_healthy() if d.name != current_name]
    if not candidates:
        return

    shadow_dep = random.choice(candidates)
    log.debug(
        "shadow_traffic_fired",
        request_id=ctx.request_id,
        primary=current_name,
        shadow=shadow_dep.name,
        fraction=fraction,
    )
    asyncio.ensure_future(_shadow_call(ctx, shadow_dep.name))


async def stream_response(ctx: RoutingContext, **extra: Any) -> AsyncIterator:
    """Yield chunks from a streaming LiteLLM response."""
    kwargs = _build_litellm_kwargs(ctx, {**extra, "stream": True})
    response = await litellm.acompletion(**kwargs)
    async for chunk in response:
        yield chunk
