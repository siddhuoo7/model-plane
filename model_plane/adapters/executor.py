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
from typing import Any, AsyncIterator

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.routing.context import RoutingContext

log = get_logger(__name__)


def _build_litellm_kwargs(ctx: RoutingContext, extra: dict) -> dict:
    dep = ctx.selected_deployment
    assert dep is not None, "No deployment selected before execution"

    # Use compression-modified messages if available
    messages = ctx.compressed_request.get("messages") if ctx.compressed_request else None
    if messages is None:
        messages = ctx.raw_request.get("messages", [])

    kwargs: dict[str, Any] = {
        "model": dep.litellm_model,
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
    if dep.api_base:
        kwargs["api_base"] = dep.api_base
    if dep.api_key_env:
        api_key = os.environ.get(dep.api_key_env)
        if api_key:
            kwargs["api_key"] = api_key

    # watsonx auth — per https://docs.litellm.ai/docs/providers/watsonx
    # Required env vars: WATSONX_URL, WATSONX_APIKEY (note: no underscore before KEY)
    # project_id is passed as a direct param to completion(), not via api_key.
    if dep.provider == "watsonx":
        # 1. project_id — passed as completion() param AND set in os.environ
        project_id = (
            settings.watsonx_project_id
            or os.environ.get("WATSONX_PROJECT_ID")
            or os.environ.get("WX_PROJECT_ID")
        )
        if project_id:
            kwargs["project_id"] = project_id          # direct completion() param
            os.environ["WATSONX_PROJECT_ID"] = project_id   # LiteLLM env fallback

        # 2. URL — required
        wx_url = (
            settings.watsonx_url
            or os.environ.get("WATSONX_URL")
            or "https://us-south.ml.cloud.ibm.com"
        )
        kwargs["api_base"] = wx_url
        os.environ["WATSONX_URL"] = wx_url

        # 3. API key — LiteLLM reads WATSONX_APIKEY from env automatically;
        #    we ensure it is in os.environ using the correct name.
        api_key = (
            settings.watsonx_api_key                    # read from WATSONX_APIKEY or WATSONX_API_KEY
            or os.environ.get("WATSONX_APIKEY")
            or os.environ.get("WATSONX_API_KEY")
        )
        if api_key:
            os.environ["WATSONX_APIKEY"] = api_key      # canonical name LiteLLM reads

    return kwargs


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    reraise=True,
)
async def aexecute(ctx: RoutingContext, **extra: Any) -> Any:
    """Async LiteLLM call with retry."""
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
