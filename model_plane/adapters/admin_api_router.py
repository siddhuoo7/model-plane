"""Admin API router — Sub-Tasks 3.1, 3.2, and 3.3.

Mounted at /admin/api/ by app.py.

Sub-Task 3.1 endpoints:
  GET  /admin/api/health
  GET  /admin/api/traffic
  GET  /admin/api/requests
  GET  /admin/api/cost

Sub-Task 3.2 endpoints:
  GET    /admin/api/catalog
  POST   /admin/api/catalog
  PUT    /admin/api/catalog/{name}
  DELETE /admin/api/catalog/{name}
  POST   /admin/api/catalog/reload
  GET    /admin/api/routing/config
  PUT    /admin/api/routing/config
  POST   /admin/api/routing/preview
  GET    /admin/api/providers
  POST   /admin/api/providers/{name}/test
  GET    /admin/api/providers/{provider}/available
  POST   /admin/api/simulate

Sub-Task 3.3 endpoints:
  GET    /admin/api/ml/status
  POST   /admin/api/ml/retrain
  GET    /admin/api/ml/retrain/{job_id}/stream  (SSE)
  POST   /admin/api/ml/activate
  GET/POST/PUT/DELETE /admin/api/tenants
  GET/POST/PUT/DELETE /admin/api/alerts

Auth:
  All endpoints require a ``Bearer <ADMIN_API_KEY>`` header when
  ``settings.admin_api_key`` is set.  If the key is unset or empty, all
  requests are allowed (development mode).
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Annotated, Any

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.observability.cost_accumulator import get_cost_accumulator
from model_plane.observability.request_buffer import get_request_buffer
from model_plane.registry.catalog import (
    DeploymentConfig,
    get_catalog,
    load_prices_for_provider,
    reload_catalog,
)
from model_plane.provider_creds import provider_has_creds as _provider_has_creds

log = get_logger(__name__)

router = APIRouter(prefix="/admin/api", tags=["admin-api"])

# ── Auth dependency ───────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _require_admin_key(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Verify the Bearer token matches ADMIN_API_KEY.

    Skips auth entirely when ``settings.admin_api_key`` is unset/empty
    (local development mode).
    """
    required = settings.admin_api_key
    if not required:
        return  # auth disabled
    if creds is None or creds.credentials != required:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")


AdminAuth = Annotated[None, Depends(_require_admin_key)]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def admin_health(_auth: AdminAuth) -> dict[str, Any]:
    """Basic health check and routing status."""
    catalog = get_catalog()
    buf = get_request_buffer()
    return {
        "status": "ok",
        "timestamp": time.time(),
        "routing_mode": settings.routing_mode,
        "ml_routing_enabled": settings.ml_routing_enabled,
        "total_deployments": len(catalog.deployments),
        "healthy_deployments": len(catalog.all_healthy()),
        "buffer_size": len(buf),
        "buffer_capacity": settings.admin_buffer_size,
    }


@router.get("/traffic")
async def admin_traffic(
    _auth: AdminAuth,
    window_hours: int = Query(default=24, ge=1, le=720, description="Lookback window in hours"),
) -> dict[str, Any]:
    """Summarised traffic metrics for the requested time window.

    Derived from the cost accumulator bucketed data (hourly granularity).
    Returns request count, estimated cost, and token totals.
    """
    acc = get_cost_accumulator()
    summary = acc.summary(hours=window_hours)

    total_requests = sum(r["requests"] for r in summary)
    total_cost = sum(r["cost_usd"] for r in summary)
    total_input = sum(r["input_tokens"] for r in summary)
    total_output = sum(r["output_tokens"] for r in summary)

    # Provider breakdown
    by_provider: dict[str, dict] = {}
    for row in summary:
        p = row["provider"]
        if p not in by_provider:
            by_provider[p] = {"requests": 0, "cost_usd": 0.0,
                               "input_tokens": 0, "output_tokens": 0}
        by_provider[p]["requests"] += row["requests"]
        by_provider[p]["cost_usd"] = round(by_provider[p]["cost_usd"] + row["cost_usd"], 6)
        by_provider[p]["input_tokens"] += row["input_tokens"]
        by_provider[p]["output_tokens"] += row["output_tokens"]

    # Tier breakdown
    by_tier: dict[str, dict] = {}
    for row in summary:
        t = row["tier"]
        if t not in by_tier:
            by_tier[t] = {"requests": 0, "cost_usd": 0.0}
        by_tier[t]["requests"] += row["requests"]
        by_tier[t]["cost_usd"] = round(by_tier[t]["cost_usd"] + row["cost_usd"], 6)

    return {
        "window_hours": window_hours,
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "by_provider": by_provider,
        "by_tier": by_tier,
    }


@router.get("/requests")
async def admin_requests(
    _auth: AdminAuth,
    limit: int = Query(default=100, ge=1, le=1000),
    provider: str | None = Query(default=None),
    tier: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    tenant_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return recent routing decisions from the in-memory ring buffer.

    Results are ordered newest-first (default limit 100).
    """
    buf = get_request_buffer()
    records = buf.snapshot(
        limit=limit,
        provider=provider,
        tier=tier,
        task_type=task_type,
        tenant_id=tenant_id,
    )
    return {
        "count": len(records),
        "total_buffered": len(buf),
        "records": records,
    }


@router.get("/cost")
async def admin_cost(
    _auth: AdminAuth,
    window_hours: int = Query(default=24, ge=1, le=720),
) -> dict[str, Any]:
    """Per-provider / per-tier / per-task-type cost breakdown."""
    acc = get_cost_accumulator()
    rows = acc.summary(hours=window_hours)

    # Task-type breakdown
    by_task: dict[str, dict] = {}
    for row in rows:
        tt = row["task_type"]
        if tt not in by_task:
            by_task[tt] = {"requests": 0, "cost_usd": 0.0}
        by_task[tt]["requests"] += row["requests"]
        by_task[tt]["cost_usd"] = round(by_task[tt]["cost_usd"] + row["cost_usd"], 6)

    return {
        "window_hours": window_hours,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
        "rows": rows,
        "by_task_type": by_task,
    }


@router.delete("/observability/flush", status_code=200)
async def flush_observability(_auth: AdminAuth) -> dict[str, Any]:
    """Wipe all in-memory cost + request data AND clear the persisted DB tables.

    Use this to discard test / mock data without restarting the server.
    """
    import model_plane.observability.cost_accumulator as _acc_mod
    import model_plane.observability.request_buffer as _buf_mod
    from model_plane.db import prune_cost_buckets, prune_request_log

    # Reset in-memory cost accumulator
    with _acc_mod._acc_lock:
        _acc_mod._accumulator = None

    # Reset in-memory request buffer
    with _buf_mod._buffer_lock:
        _buf_mod._buffer = None

    # Wipe DB tables (cutoff far in the future removes everything)
    far_future = int(time.time()) + 10 * 365 * 24 * 3600
    prune_cost_buckets(far_future)
    prune_request_log(0)

    log.info("observability_flushed")
    return {"flushed": True}


@router.get("/providers/{provider}/available")
async def provider_available_models(
    provider: str,
    _auth: AdminAuth,
) -> dict[str, Any]:
    """List all models known for *provider* from ``config/prices.yaml``.

    Used by the "Add deployment" wizard in Sub-Task 3.5 to show pre-priced
    models for a selected provider without requiring a live API call.

    Returns entries in the original USD/MTok price format from prices.yaml.
    """
    entries = load_prices_for_provider(provider)
    if not entries:
        log.debug("no_prices_entries_for_provider", provider=provider)
    return {
        "provider": provider,
        "count": len(entries),
        "models": entries,
    }


# ── Catalog CRUD (Sub-Task 3.2) ───────────────────────────────────────────────

def _dep_to_dict(dep: DeploymentConfig) -> dict[str, Any]:
    """Serialise a DeploymentConfig to the canonical catalog API schema."""
    return {
        "name": dep.name,
        "litellm_model": dep.litellm_model,
        "provider": dep.provider,
        "tier": dep.tier,
        "healthy": dep.healthy,
        "context_limit": dep.context_limit,
        "capabilities": dep.capabilities,
        "tags": dep.tags,
        "cost_per_1k_input": dep.cost_per_1k_input,
        "cost_per_1k_output": dep.cost_per_1k_output,
        "input_per_mtok_usd": dep.input_per_mtok_usd,
        "output_per_mtok_usd": dep.output_per_mtok_usd,
        "cache_read_per_1k": dep.cache_read_per_1k,
        "cache_write_per_1k": dep.cache_write_per_1k,
        "fallback_to": dep.fallback_to,
        "max_parallel_requests": dep.max_parallel_requests,
        "rpm": dep.rpm,
        "api_base": dep.api_base,
        # Security / cache fields (Phase 7)
        "provider_trust": getattr(dep, "provider_trust", None),
        "cache_capability": getattr(dep, "cache_capability", None),
    }


class DeploymentCreateRequest(BaseModel):
    name: str
    litellm_model: str = ""
    provider: str = "openai"
    tier: str = "medium"
    context_limit: int = 8192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: list[str] = []
    tags: list[str] = []
    fallback_to: list[str] = []
    max_parallel_requests: int = 50
    rpm: int | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    # Explicit health flag — None means "don't change existing value" (patch semantics)
    healthy: bool | None = None



def _get_user_catalog(request: Request) -> "tuple[Any, str]":
    """Return (user_catalog_view, owner_id) for the calling user.

    Reads the session JWT from the Authorization header (if present) to look up
    this user's catalog overrides in SQLite, then applies them on top of the
    global YAML-base catalog.

    If no session JWT is present (admin-key-only call) owner_id is '' and the
    plain YAML catalog is returned.
    """
    import model_plane.db as _db
    from model_plane.adapters.settings_router import _jwt_verify
    owner_id = ""
    try:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            session = _jwt_verify(token)
            owner_id = session.get("id", "")
    except Exception:
        pass
    base = get_catalog()
    if not owner_id:
        return base, owner_id
    overrides = _db.load_catalog_overrides(owner_id)
    return base.apply_user_overrides(overrides), owner_id


@router.get("/catalog")
async def catalog_list(
    request: Request,
    _auth: AdminAuth,
    all: bool = Query(default=False, description="Set true to include uncredentialed providers"),
) -> dict[str, Any]:
    """Return deployments from models.yaml + user overrides.

    models.yaml is the immutable base; each user's enable/disable/delete/add
    actions are stored in SQLite and layered on top at read time.  One user's
    changes do not affect other users or the global routing pool.
    """
    catalog, _owner = _get_user_catalog(request)
    result = []
    uncredentialed_count = 0
    for d in catalog.deployments.values():
        cred_ok = _provider_has_creds(d.provider)
        if not cred_ok:
            uncredentialed_count += 1
        if not all and not cred_ok:
            continue
        dep_dict = _dep_to_dict(d)
        dep_dict["credential_ok"] = cred_ok
        result.append(dep_dict)

    return {
        "count": len(result),
        "total_in_yaml": len(get_catalog().deployments),
        "uncredentialed_hidden": uncredentialed_count if not all else 0,
        "deployments": result,
    }


@router.post("/catalog", status_code=201)
async def catalog_create(
    request: Request,
    body: DeploymentCreateRequest,
    _auth: AdminAuth,
) -> dict[str, Any]:
    """Add a new deployment for the calling user.

    Stored as an 'added' override in SQLite — models.yaml is not modified.
    Other users will not see this deployment unless they add it themselves.
    """
    import model_plane.db as _db
    _, owner_id = _get_user_catalog(request)
    base = get_catalog()

    # Check uniqueness across base YAML and user's existing overrides
    existing_overrides = _db.load_catalog_overrides(owner_id) if owner_id else {}
    user_catalog = base.apply_user_overrides(existing_overrides)
    if body.name in user_catalog.deployments:
        raise HTTPException(status_code=409, detail=f"Deployment '{body.name}' already exists")

    dep_data: dict[str, Any] = {
        "name": body.name,
        "litellm_model": body.litellm_model,
        "provider": body.provider,
        "tier": body.tier,
        "context_limit": body.context_limit,
        "cost_per_1k_input": body.cost_per_1k_input,
        "cost_per_1k_output": body.cost_per_1k_output,
        "capabilities": list(body.capabilities),
        "tags": list(body.tags),
        "fallback_to": list(body.fallback_to),
        "max_parallel_requests": body.max_parallel_requests,
        "rpm": body.rpm,
        "api_base": body.api_base,
        "api_key_env": body.api_key_env,
        "healthy": True,
    }
    from model_plane.registry.catalog import DeploymentConfig
    dep = DeploymentConfig(
        name=body.name, litellm_model=body.litellm_model,
        provider=body.provider, tier=body.tier,
        context_limit=body.context_limit,
        cost_per_1k_input=body.cost_per_1k_input,
        cost_per_1k_output=body.cost_per_1k_output,
        capabilities=list(body.capabilities), tags=list(body.tags),
        fallback_to=list(body.fallback_to),
        max_parallel_requests=body.max_parallel_requests,
        rpm=body.rpm, api_base=body.api_base, api_key_env=body.api_key_env,
    )
    # Always add to global routing catalog so the deployment is immediately routable
    base.deployments[dep.name] = dep
    base.tier_map.setdefault(dep.tier, []).append(dep.name)
    if owner_id:
        # Persist as user override so it survives restart and stays user-scoped
        _db.save_catalog_override(body.name, {"added": dep_data}, owner_id)
    log.info("catalog_deployment_added", name=body.name, owner=owner_id)
    return dep_data


@router.put("/catalog/{name}")
async def catalog_update(
    name: str,
    request: Request,
    body: DeploymentCreateRequest = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Update a deployment for the calling user.

    Changes are stored as field overrides in SQLite.  models.yaml is not modified.
    """
    import model_plane.db as _db
    catalog, owner_id = _get_user_catalog(request)
    dep = catalog.get(name)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"Deployment '{name}' not found")

    fields: dict[str, Any] = {}
    if body.litellm_model:
        fields["litellm_model"] = body.litellm_model
    if body.provider:
        fields["provider"] = body.provider
    fields["tier"] = body.tier
    fields["context_limit"] = body.context_limit
    fields["cost_per_1k_input"] = body.cost_per_1k_input
    fields["cost_per_1k_output"] = body.cost_per_1k_output
    if body.capabilities:
        fields["capabilities"] = list(body.capabilities)
    if body.tags:
        fields["tags"] = list(body.tags)
    if body.fallback_to:
        fields["fallback_to"] = list(body.fallback_to)
    fields["max_parallel_requests"] = body.max_parallel_requests
    if body.rpm is not None:
        fields["rpm"] = body.rpm
    if body.api_base is not None:
        fields["api_base"] = body.api_base
    if body.api_key_env is not None:
        fields["api_key_env"] = body.api_key_env
    if body.healthy is not None:
        fields["healthy"] = body.healthy

    if owner_id:
        patch: dict[str, Any] = {"fields": fields}
        if body.healthy is not None:
            # healthy/disabled is a first-class flag, not just a field override
            patch["disabled"] = not body.healthy
        _db.save_catalog_override(name, patch, owner_id)

    # Also patch the global in-memory catalog so routing picks it up immediately
    base_dep = get_catalog().get(name)
    if base_dep:
        for k, v in fields.items():
            if hasattr(base_dep, k):
                setattr(base_dep, k, v)

    log.info("catalog_deployment_updated", name=name, owner=owner_id)
    # Re-read the updated dep from the user's view
    updated_catalog, _ = _get_user_catalog(request)
    updated = updated_catalog.get(name)
    return _dep_to_dict(updated or dep)


@router.patch("/catalog/{name}/healthy")
async def catalog_set_healthy(
    name: str,
    request: Request,
    body: dict[str, Any] = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Toggle enable/disable for the calling user only.

    Stored as a 'disabled' override in SQLite.  Other users are unaffected.
    """
    import model_plane.db as _db
    catalog, owner_id = _get_user_catalog(request)
    dep = catalog.get(name)
    if dep is None:
        # Also check base YAML — deployment may be deleted in this user's view
        dep = get_catalog().get(name)
        if dep is None:
            raise HTTPException(status_code=404, detail=f"Deployment '{name}' not found")
    healthy = body.get("healthy")
    if healthy is None or not isinstance(healthy, bool):
        raise HTTPException(status_code=422, detail='Body must be {"healthy": true|false}')

    if owner_id:
        _db.save_catalog_override(name, {"disabled": not healthy, "fields": {"healthy": healthy}}, owner_id)

    # Also update the global in-memory dep so routing reflects the change
    base_dep = get_catalog().get(name)
    if base_dep:
        base_dep.healthy = healthy

    log.info("catalog_deployment_toggled", name=name, healthy=healthy, owner=owner_id)
    return {**_dep_to_dict(dep), "healthy": healthy}


@router.delete("/catalog/{name}", status_code=204)
async def catalog_delete(
    name: str,
    request: Request,
    _auth: AdminAuth = None,
) -> None:
    """Mark a deployment as deleted for the calling user.

    Stored as a 'deleted' override in SQLite.  models.yaml is not modified.
    Other users still see the deployment.
    """
    import model_plane.db as _db
    catalog, owner_id = _get_user_catalog(request)
    # Allow deleting user-added deployments too
    in_base = name in get_catalog().deployments
    in_user = name in catalog.deployments
    if not in_base and not in_user:
        raise HTTPException(status_code=404, detail=f"Deployment '{name}' not found")

    if owner_id:
        if in_base:
            # Mark as deleted in user's overlay
            _db.save_catalog_override(name, {"deleted": True}, owner_id)
        else:
            # User-created deployment — remove the added override entirely
            _db.delete_catalog_override(name, owner_id)

    # Remove from global routing catalog immediately
    base = get_catalog()
    if name in base.deployments:
        del base.deployments[name]
        base.tier_map = {}
        for d in base.deployments.values():
            base.tier_map.setdefault(d.tier, []).append(d.name)

    log.info("catalog_deployment_deleted", name=name, owner=owner_id)


@router.post("/catalog/reload")
async def catalog_reload(_auth: AdminAuth) -> dict[str, Any]:
    """Hot-reload the model catalog from disk (discards all in-memory changes)."""
    cat = reload_catalog()
    return {"status": "ok", "deployments": len(cat.deployments)}


# ── Routing config (Sub-Task 3.2) ─────────────────────────────────────────────

def _get_routing_owner(request: Request) -> str:
    """Extract owner_id from the session JWT in the request, or '' if none."""
    from model_plane.adapters.settings_router import _jwt_verify
    try:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            session = _jwt_verify(auth[7:])
            return session.get("id", "")
    except Exception:
        pass
    return ""


@router.get("/routing/config")
async def routing_config_get(request: Request, _auth: AdminAuth) -> dict[str, Any]:
    """Return the routing config merged with the calling user's DB overrides.

    routing.yaml is the immutable base; the user's routing_rules override in
    SQLite is layered on top so each user has their own routing preferences.
    """
    import model_plane.db as _db
    base = settings.load_routing_config()
    owner_id = _get_routing_owner(request)
    if owner_id:
        user_rules = _db.load_routing_override("routing_rules", owner_id)
        if user_rules:
            base = {**base, **user_rules}
    return {"config": base}


@router.put("/routing/config")
async def routing_config_put(
    request: Request,
    body: dict[str, Any] = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Save routing config overrides to SQLite for the calling user.

    routing.yaml is NEVER modified. Changes are user-scoped and survive restart
    via bootstrap step 4. Other users are unaffected.
    """
    import model_plane.db as _db
    owner_id = _get_routing_owner(request)
    if owner_id:
        _db.save_routing_override("routing_rules", body, owner_id)
        log.info("routing_config_updated_in_db", owner=owner_id)
        return {"status": "ok", "persisted": "db"}
    # Fallback for unauthenticated/admin-key calls: write to routing.yaml
    import pathlib
    path = pathlib.Path(settings.routing_config_path)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.dump(body, default_flow_style=False, sort_keys=False))
    tmp.replace(path)
    log.info("routing_config_updated_yaml", path=str(path))
    return {"status": "ok", "path": str(path)}


@router.post("/routing/preview")
async def routing_preview(
    body: dict[str, Any] = Body(...),
    limit: int = Query(default=100, ge=1, le=500),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Re-simulate the last *limit* buffered requests with a pending config delta.

    Body: the pending routing config dict (same schema as routing.yaml).
    Returns a before/after tier-distribution diff so operators can validate
    a config change before writing it to disk.
    """
    from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline

    buf = get_request_buffer()
    recent = buf.snapshot(limit=limit)
    if not recent:
        return {"status": "no_data", "message": "Ring buffer is empty — no requests to replay"}

    before: dict[str, int] = {}
    after: dict[str, int] = {}

    for rec in recent:
        before_tier = rec.get("tier", "unknown")
        before[before_tier] = before.get(before_tier, 0) + 1

        # Re-route with pending config patched onto settings.load_routing_config
        try:
            from unittest.mock import patch as _patch
            raw_req = {"messages": [{"role": "user", "content": rec.get("request_id", "")}]}
            with _patch.object(settings, "load_routing_config", return_value=body):
                ctx = build_routing_context(raw_req)
                decision = run_routing_pipeline(ctx)
            after_tier = decision.deployment.tier if decision.deployment else "unknown"
        except Exception:
            after_tier = before_tier  # on error, assume no change

        after[after_tier] = after.get(after_tier, 0) + 1

    return {
        "replayed": len(recent),
        "tier_distribution_before": before,
        "tier_distribution_after": after,
        "diff": {
            tier: after.get(tier, 0) - before.get(tier, 0)
            for tier in set(list(before) + list(after))
        },
    }


# ── Provider management (Sub-Task 3.2) ───────────────────────────────────────

@router.get("/providers")
async def providers_list(_auth: AdminAuth) -> dict[str, Any]:
    """List all registered providers with credential status and catalog model count."""
    from model_plane.adapters.providers import _REGISTRY

    catalog = get_catalog()
    # Count deployments per provider from the catalog
    deps_by_provider: dict[str, int] = {}
    for dep in catalog.deployments.values():
        deps_by_provider[dep.provider] = deps_by_provider.get(dep.provider, 0) + 1

    provider_status = []
    for name in _REGISTRY:
        has_creds = False
        endpoint = None
        cred_source = None  # which env var drives this provider
        if name == "openai":
            has_creds = bool(settings.openai_api_key)
            cred_source = "OPENAI_API_KEY"
        elif name == "anthropic":
            has_creds = bool(settings.anthropic_api_key)
            cred_source = "ANTHROPIC_API_KEY"
        elif name == "watsonx":
            has_creds = bool(settings.watsonx_api_key)
            endpoint = settings.watsonx_url
            cred_source = "WATSONX_APIKEY"
        elif name in ("local", "vllm"):
            # Credentialed only when the base URL is present in os.environ.
            # provider_creds_clear pops it, so the UI reflects "not set" immediately.
            import os as _os
            has_creds = bool(_os.environ.get("LOCAL_VLLM_API_BASE"))
            cred_source = "LOCAL_VLLM_API_BASE"
        elif name == "bedrock":
            # Prefer API-key mode; fall back to IAM credential check
            has_creds = bool(
                settings.aws_bearer_token_bedrock
                or __import__("os").environ.get("AWS_BEARER_TOKEN_BEDROCK")
                or settings.aws_access_key_id
            )
            endpoint = settings.aws_region_name
            if settings.aws_bearer_token_bedrock or __import__("os").environ.get("AWS_BEARER_TOKEN_BEDROCK"):
                cred_source = "AWS_BEARER_TOKEN_BEDROCK (API key mode)"
            else:
                cred_source = "AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY"
        elif name == "azure":
            has_creds = bool(settings.azure_api_key)
            endpoint = settings.azure_api_base
            cred_source = "AZURE_API_KEY + AZURE_API_BASE"
        elif name == "vertex_ai":
            has_creds = bool(settings.vertex_project)
            endpoint = settings.vertex_location
            cred_source = "VERTEXAI_PROJECT + VERTEXAI_LOCATION"
        elif name == "cohere":
            has_creds = bool(settings.cohere_api_key)
            cred_source = "COHERE_API_KEY"
        elif name == "mistral":
            has_creds = bool(settings.mistral_api_key)
            cred_source = "MISTRAL_API_KEY"

        provider_status.append({
            "name": name,           # UI reads `name`
            "provider": name,       # backward compat
            "configured": has_creds,
            "endpoint": endpoint,
            "cred_source": cred_source,
            "available_models": deps_by_provider.get(name, 0),
        })

    return {"providers": provider_status}


@router.post("/providers/{name}/test")
async def provider_test(
    name: str = Path(...),
    body: dict[str, Any] = Body(default={}),
    request: Request = None,
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Validate provider credentials by calling the provider's model-listing API.

    1. Calls list_models() on the provider adapter — a real network round-trip.
    2. Cross-references the returned live model IDs against the deployments
       configured in models.yaml for this provider.
    3. Marks yaml deployments whose litellm_model ID appears in the live list
       as healthy=True; marks those absent as healthy=False.
    4. Returns the full cross-reference so the UI can surface which yaml models
       are available vs. not found on the provider side.

    Accepts either an admin API key OR a UI session JWT (from the Provider
    Management modal) so SQLite-stored credentials are applied before the test.
    """
    from model_plane.adapters.providers import get_adapter
    from model_plane.adapters.settings_router import (
        _apply_creds_to_process, _jwt_verify,
    )
    import model_plane.db as _db

    # Priority (lowest → highest):
    #   1. Saved SQLite credentials for the calling user (baseline from previous saves)
    #   2. Form values typed in the current modal (overwrite SQLite so unsaved edits are tested)
    #
    # This means: if the user types a new (possibly wrong) API key in the modal and
    # clicks "Test connection" WITHOUT saving, the modal values win, so an invalid
    # key actually fails the test instead of silently using the old saved key.

    # Step 1 — apply saved SQLite creds as the baseline
    if request is not None:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                session = _jwt_verify(token)
                owner_id = session.get("id", "")
                if owner_id:
                    user_creds = _db.load_provider_creds(owner_id)
                    if name in user_creds and user_creds[name]:
                        _apply_creds_to_process(name, user_creds[name])
            except HTTPException:
                pass  # invalid/expired JWT — fall through
            except Exception:
                pass  # DB or other non-auth error — ignore

    # Step 2 — overwrite with current form values so the test reflects what
    #           the user has typed RIGHT NOW, not what was previously saved.
    if body:
        _apply_creds_to_process(name, {k: v for k, v in body.items() if isinstance(v, str) and v})

    # Fast-fail: check credentials are present before making network call
    if not _provider_has_creds(name):
        return {
            "provider": name,
            "healthy": False,
            "error": f"Credentials not configured for provider '{name}'. "
                     "Add credentials via the Provider Management page.",
            "live_models": [],
            "yaml_models": [],
            "available": [],
            "missing": [],
        }

    adapter = get_adapter(name)

    # Call the real model-listing API in a thread pool (blocking I/O)
    probe_error: str | None = None
    live_models: list[str] = []
    try:
        live_models = await asyncio.to_thread(adapter.list_models)
    except Exception as exc:
        probe_error = str(exc)

    ok = len(live_models) > 0 and probe_error is None

    # Cross-reference against yaml deployments for this provider
    catalog = get_catalog()
    yaml_deps = [d for d in catalog.deployments.values() if d.provider == name]

    # Build a normalised set of live model IDs for fuzzy matching
    live_set = {m.lower().strip() for m in live_models}

    def _model_in_live(dep_litellm: str) -> bool:
        """Check if a yaml deployment's litellm_model appears in the live list.

        Tries exact match first, then strips known prefixes
        (bedrock/, anthropic/, openai/, etc.) to get the bare model ID.
        """
        bare = dep_litellm.lower().strip()
        if bare in live_set:
            return True
        # Strip provider prefix  e.g. "bedrock/anthropic.claude-3-haiku-..." -> "anthropic.claude-3-haiku-..."
        for sep in ("/",):
            if sep in bare:
                bare_stripped = bare.split(sep, 1)[-1]
                if bare_stripped in live_set:
                    return True
                # Also try matching just the last segment for deeply nested IDs
                last = bare_stripped.split(sep)[-1] if sep in bare_stripped else bare_stripped
                if any(last in m for m in live_set):
                    return True
        # Reverse: check if any live model contains the bare dep ID as substring
        if any(bare in m or m in bare for m in live_set):
            return True
        return False

    available_deps: list[str] = []
    missing_deps: list[str] = []
    for d in yaml_deps:
        if ok and _model_in_live(d.litellm_model):
            d.healthy = True
            available_deps.append(d.name)
        elif ok:
            # Live listing succeeded but this model wasn't in the list
            d.healthy = False
            missing_deps.append(d.name)
        else:
            # Listing failed — don't change existing health status
            pass

    result: dict[str, Any] = {
        "provider": name,
        "healthy": ok,
        "live_model_count": len(live_models),
        "yaml_models": [d.name for d in yaml_deps],
        "available": available_deps,
        "missing": missing_deps,
    }
    if probe_error:
        result["error"] = probe_error
    return result


# ── Simulate (Sub-Task 3.2) ───────────────────────────────────────────────────

@router.post("/simulate")
async def simulate(
    body: dict[str, Any] = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Dry-run a chat request through the full routing pipeline.

    Accepts the same body as ``POST /v1/chat/completions`` (needs ``messages``).
    No LLM call is made. Returns a structured 7-step pipeline trace matching
    the strata simulator output format, used by the Simulator UI panel.
    """
    from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline

    if not body.get("messages"):
        raise HTTPException(status_code=422, detail="'messages' field is required")

    request_id = f"sim-{uuid.uuid4().hex[:8]}"
    ctx = build_routing_context(body, request_id=request_id)

    try:
        decision = run_routing_pipeline(ctx)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pipeline error: {exc}") from exc

    f = ctx.features
    clf = ctx.classification
    sc = ctx.scorer_result

    # Detect classifier fallback: configured classifier is non-regex but result
    # source is regex.  Two distinct cases:
    #   (a) library not installed → warn with install hint
    #   (b) library ran but confidence was below threshold → informational only (no warning)
    from model_plane.classifier.factory import check_classifier_availability
    configured_clf = _classifier_overrides.get("classifier_model", settings.classifier_model)
    actual_source_float = float(clf.signals.get("_source", 0.0)) if clf else 0.0
    actual_source = {0.0: "regex", 1.0: "mbert", 2.0: "laya"}.get(actual_source_float, "regex")
    classifier_warning: str | None = None
    if configured_clf != "regex" and actual_source == "regex":
        availability = check_classifier_availability()
        clf_available = availability.get(configured_clf, False)
        if not clf_available:
            _hints = {
                "laya": "Run: pip install laya",
                "mbert": "Run: pip install 'transformers>=4.35' torch",
            }
            classifier_warning = (
                f"Active classifier is '{configured_clf}' but it is unavailable — "
                f"result is from regex fallback. "
                + _hints.get(configured_clf, "Check server logs for details.")
            )
        # else: classifier ran but fell back (below confidence threshold) — not an error

    # ── Security trace (Stage 0.5) ────────────────────────────────────────────
    sec = ctx.security_ctx
    security_trace: dict[str, Any] = {
        "enabled": settings.security_routing_enabled,
        "sensitivity": sec.data_sensitivity.value if sec else "public",
        "pii_types": sec.pii_types if sec else [],
        "secret_types": sec.secret_types if sec else [],
        "blocked_providers": list(sec.blocked_deployments.keys()) if sec else [],
        "audit_summary": sec.audit_summary if sec else "",
        "candidates_removed": len(sec.blocked_deployments) if sec else 0,
    }

    # ── Cache / context-reuse trace (Stage 6e) ────────────────────────────────
    reuse_score = ctx.context_reuse_score
    session_id_used = ctx.session_id
    cache_routing_trace: dict[str, Any] = {
        "enabled": settings.context_reuse_enabled,
        "session_id": session_id_used,
        "context_reuse_score": round(reuse_score, 4),
        "turn_index": ctx.turn_index,
        "cache_warm": ctx.cache_warm,
        "prefix_hash": ctx.prefix_hash,
        "routing_reason": _cache_routing_reason(ctx, decision),
    }

    # ── Compression trace ─────────────────────────────────────────────────────
    tokens_before = ctx.tokens_before_compression or (f.total_tokens if f else 0)
    tokens_after = ctx.tokens_after_compression or tokens_before
    savings_pct = round((1 - tokens_after / max(tokens_before, 1)) * 100, 1)
    compression_trace: dict[str, Any] = {
        "profile": ctx.compression_profile,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "savings_pct": savings_pct,
        "cache_aware_skip": (
            ctx.compression_profile == "passthrough"
            and reuse_score >= 0.60
        ),
        "cache_aware_threshold": 0.60,
    }

    return {
        "dry_run": True,
        "classifier_warning": classifier_warning,
        "router": "smart",
        "request_id": request_id,
        "selected_model": decision.deployment.litellm_model if decision.deployment else None,
        "tier": decision.deployment.tier if decision.deployment else None,
        "task_type": clf.task_type.value if clf else None,
        "routing_stage": decision.source,
        "score": round(sc.raw_score, 4) if sc else None,
        "pipeline_steps": {
            "features": {
                "total_tokens": f.total_tokens if f else 0,
                "user_messages": f.user_message_count if f else 0,
                "has_tools": f.has_tools if f else False,
                "has_images": getattr(f, "has_images", False) if f else False,
                "language_hint": f.language_hint if f else None,
                "dimensions": {
                    k: round(v, 4)
                    for k, v in (sc.dimension_scores.items() if sc else {})
                },
            },
            "classifier": {
                "task_type": clf.task_type.value if clf else None,
                "classifier_tier": clf.complexity_tier.value if clf else None,
                "confidence": round(clf.confidence, 4) if clf else None,
                # Decode _source float: 0.0=regex, 1.0=mbert, 2.0=laya
                "source": (
                    {0.0: "regex", 1.0: "mbert", 2.0: "laya"}.get(
                        float(clf.signals.get("_source", 0.0)), "regex"
                    ) if clf else "regex"
                ),
                # Exclude private _-prefixed keys from the display chips
                "fired_signals": (
                    {k: v for k, v in clf.signals.items() if v > 0 and not k.startswith("_")} if clf else {}
                ),
                "all_signals": (
                    {k: v for k, v in clf.signals.items() if not k.startswith("_")} if clf else {}
                ),
            },
            "scorer": {
                "raw_score": round(sc.raw_score, 4) if sc else None,
                "scorer_tier": sc.tier.value if sc else None,
                "scorer_confidence": round(sc.confidence, 4) if sc else None,
                "dimension_scores": (
                    {k: round(v, 4) for k, v in sc.dimension_scores.items()} if sc else {}
                ),
            },
            "blend": {
                "scorer_tier": sc.tier.value if sc else None,
                "classifier_tier": clf.complexity_tier.value if clf else None,
                "blended_tier": decision.deployment.tier if decision.deployment else None,
            },
            "ml": {
                "enabled": settings.ml_routing_enabled,
                "predicted_tier": ctx.ml_recommended_deployment,
                "confidence": round(ctx.ml_recommendation_confidence or 0.0, 4),
                "active": ctx.routing_source == "ml_active",
                "threshold": settings.ml_confidence_threshold,
            },
            "policy_applied": {
                "task_override_used": bool(
                    settings.load_routing_config()
                    .get("task_deployment_overrides", {})
                    .get(clf.task_type.value if clf else "", {})
                    .get("preferred_deployments")
                ),
                "resolved_tier": decision.deployment.tier if decision.deployment else None,
            },
            "winner": {
                "model": decision.deployment.litellm_model if decision.deployment else None,
                "tier": decision.deployment.tier if decision.deployment else None,
                "stage": decision.source,
                "confidence": round(decision.confidence, 4),
                "reasoning": decision.reasoning,
            },
        },
        "candidates": [
            {
                "model": d.litellm_model,
                "provider": d.provider,
                "tier": d.tier,
                "input_per_mtok": d.input_per_mtok_usd,
                "capabilities": d.capabilities,
            }
            for d in (ctx.candidate_deployments or [])
        ],
        "cost": {
            "input_per_mtok_usd": decision.deployment.input_per_mtok_usd if decision.deployment else 0,
            "estimated_input_cost_usd": round(
                (f.total_tokens if f else 0) / 1_000_000 * (
                    decision.deployment.cost_per_1k_input * 1000 if decision.deployment else 0
                ),
                8,
            ),
        },
        "features": {
            "last_message_chars": len(
                body["messages"][-1].get("content", "") if body.get("messages") else ""
            ),
            "estimated_tokens": f.total_tokens if f else 0,
            "message_count": len(body.get("messages", [])),
            "has_tools": f.has_tools if f else False,
        },
        # Phase 7 — cache-aware, security, compression traces
        "security": security_trace,
        "cache_routing": cache_routing_trace,
        "compression": compression_trace,
    }


def _cache_routing_reason(ctx: Any, decision: Any) -> str:
    """Produce a human-readable string explaining why the session/cache state
    influenced (or didn't influence) the routing decision."""
    reuse = ctx.context_reuse_score
    session_id = ctx.session_id
    if not session_id:
        return "No session ID — stateless routing"
    if reuse >= 0.80:
        return f"Strong cache affinity (score {reuse:.2f}): session routed to warm model"
    if reuse >= 0.50:
        return f"Moderate cache affinity (score {reuse:.2f}): context-reuse bonus applied"
    if reuse > 0:
        return f"Weak cache signal (score {reuse:.2f}): base scoring dominated"
    return "Session present but no prefix cache match — cold start"


# ── Job store for retrain progress (Sub-Task 3.3) ────────────────────────────

class _JobStatus:
    """Tracks a single background retrain job."""
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.status: str = "queued"   # queued | running | done | failed
        self.progress: int = 0        # 0–100
        self.message: str = ""
        self.result: dict | None = None
        self.error: str | None = None
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def emit(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)

    def drain(self) -> list[dict]:
        with self._lock:
            out, self._events = self._events, []
            return out


_jobs: dict[str, _JobStatus] = {}
_jobs_lock = threading.Lock()


def _get_job(job_id: str) -> _JobStatus | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _register_job(job_id: str) -> _JobStatus:
    job = _JobStatus(job_id)
    with _jobs_lock:
        _jobs[job_id] = job
    return job


# ── Tenant and alert in-memory stores (simple; replaced by DB in production) ──

_tenants: dict[str, dict[str, Any]] = {}
_alerts: dict[str, dict[str, Any]] = {}


# ── ML Admin API endpoints (Sub-Task 3.3) ─────────────────────────────────────

def _load_tier_model_info(path: "pathlib.Path") -> dict[str, Any]:
    """Load metadata from a joblib tier-model artefact."""
    info: dict[str, Any] = {"model_loaded": False, "model_type": None, "class_names": [],
                             "feature_count": None, "accuracy": None, "feature_importances": {}}
    try:
        import joblib
        artefact = joblib.load(path)
        model = artefact.get("model") if isinstance(artefact, dict) else artefact
        if isinstance(artefact, dict):
            info["class_names"] = list(artefact.get("class_names", []))
        if model is not None:
            info["model_loaded"] = True
            info["model_type"] = type(model).__name__
            if hasattr(model, "n_features_in_"):
                info["feature_count"] = int(model.n_features_in_)
            if hasattr(model, "feature_importances_"):
                from model_plane.ml.trainer import FEATURE_COLUMNS
                imps = model.feature_importances_.tolist()
                info["feature_importances"] = {
                    FEATURE_COLUMNS[i] if i < len(FEATURE_COLUMNS) else f"dim_{i}": round(v, 5)
                    for i, v in enumerate(imps)
                }
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _load_task_model_info(path: "pathlib.Path") -> dict[str, Any]:
    """Load metadata from a task-classifier artefact (joblib or JSON)."""
    info: dict[str, Any] = {"model_loaded": False, "model_type": None,
                             "class_names": [], "feature_count": None, "accuracy": None,
                             "feature_importances": {}}
    suffix = path.suffix.lower()
    if suffix in (".joblib", ".pkl"):
        # Treat same as tier model — sklearn joblib
        info.update(_load_tier_model_info(path))
        return info
    # JSON logistic-regression backup
    try:
        with open(path) as fh:
            import json as _json
            m = _json.load(fh)
        info["model_loaded"] = True
        info["model_type"] = "logistic_regression (JSON)"
        info["class_names"] = m.get("classNames") or m.get("labels", [])
        info["accuracy"] = m.get("accuracy") or m.get("training_accuracy")
        info["training_rows"] = m.get("trainingSamples") or m.get("n_samples")
        info["last_trained"] = m.get("trainedOn")
        info["feature_count"] = m.get("n_features") or len(m.get("normMeans", m.get("mean", [])))
    except Exception as exc:
        info["error"] = str(exc)
    return info


@router.get("/ml/status")
async def ml_status(_auth: AdminAuth) -> dict[str, Any]:
    """Return metadata about both ML model artefacts (tier + task classifier)."""
    import pathlib, os, datetime, csv

    data_path = pathlib.Path("model_plane/data/training_data.csv")
    training_rows: int | None = None
    try:
        if data_path.exists():
            with open(data_path, newline="") as fh:
                training_rows = sum(1 for _ in csv.reader(fh)) - 1
    except Exception:
        pass

    def _mtime(p: pathlib.Path) -> str | None:
        try:
            return datetime.datetime.utcfromtimestamp(os.path.getmtime(p)).isoformat()
        except Exception:
            return None

    # ── tier model (joblib) ───────────────────────────────────────────────────
    tier_path = pathlib.Path(str(settings.ml_model_path))
    tier_info: dict[str, Any] = {
        "model_path": str(tier_path),
        "model_loaded": False,
        "last_modified_utc": _mtime(tier_path) if tier_path.exists() else None,
        "training_rows": training_rows,
    }
    if tier_path.exists():
        tier_info.update(_load_tier_model_info(tier_path))

    # ── task classifier (JSON) ────────────────────────────────────────────────
    task_path = pathlib.Path(str(settings.ml_task_classifier_path))
    task_info: dict[str, Any] = {
        "model_path": str(task_path),
        "model_loaded": False,
        "last_modified_utc": _mtime(task_path) if task_path.exists() else None,
        "training_rows": training_rows,
    }
    if task_path.exists():
        task_info.update(_load_task_model_info(task_path))

    # ── flat response (backward compat) + nested per-model detail ─────────────
    # The existing UI reads top-level keys (model_path, accuracy, etc.) —
    # those map to the tier model for backward compatibility.
    result: dict[str, Any] = {
        # backward-compat flat fields (tier model)
        "model_path": tier_info["model_path"],
        "model_loaded": tier_info["model_loaded"],
        "model_type": tier_info.get("model_type"),
        "class_names": tier_info.get("class_names", []),
        "training_rows": training_rows,
        "feature_count": tier_info.get("feature_count"),
        "accuracy": tier_info.get("accuracy"),
        "feature_importances": tier_info.get("feature_importances", {}),
        "last_modified_utc": tier_info.get("last_modified_utc"),
        "status": "ready" if tier_info["model_loaded"] else "not_loaded",
        "last_trained": tier_info.get("last_modified_utc"),
        # per-model details consumed by the updated UI
        "tier_model": tier_info,
        "task_model": task_info,
    }
    return result


# Separate data file for each model so uploads don't cross-contaminate.
# Both models read from the same shared CSV today, but uploads target the
# model-specific file so each can be curated independently.
_TRAINING_DATA_PATHS = {
    "tier": "model_plane/data/training_data.csv",       # tier + feature columns
    "task": "model_plane/data/training_data.csv",       # same file — task_type column
}


@router.post("/ml/retrain")
async def ml_retrain(
    mode: str = Query(default="tier", pattern="^(tier|tier-json|task|task-json|deployment)$"),
    model_type: str = Query(default="gradient_boost", pattern="^(random_forest|gradient_boost)$"),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Start a background retrain job.

    ``mode=tier``      — sklearn GBT/RF tier classifier (joblib)
    ``mode=tier-json`` — portable JSON logistic regression tier classifier
    ``mode=task``      — JSON logistic regression task-type classifier
    ``mode=deployment``— legacy deployment predictor (joblib)

    Returns a job_id to poll progress via SSE.
    """
    job_id = f"retrain-{uuid.uuid4().hex[:8]}"
    job = _register_job(job_id)
    job.status = "queued"
    job.emit({"event": "queued", "progress": 0, "message": "Job queued"})

    async def _run() -> None:
        job.status = "running"
        job.emit({"event": "start", "progress": 5, "message": "Loading training data…"})
        try:
            import pathlib as _pl

            data_path = _pl.Path(_TRAINING_DATA_PATHS.get(mode, "model_plane/data/training_data.csv"))

            if mode == "task":
                from model_plane.ml.trainer import train_task_model
                output_path = _pl.Path(str(settings.ml_task_classifier_path))
                job.emit({"event": "progress", "progress": 20,
                          "message": f"Training task classifier ({model_type})…"})
                report = await asyncio.to_thread(train_task_model, data_path, output_path, model_type)
            elif mode == "task-json":
                from model_plane.ml.trainer import train_task_json_model
                output_path = _pl.Path(str(settings.ml_task_classifier_json_path))
                job.emit({"event": "progress", "progress": 20,
                          "message": "Training task classifier JSON backup (logistic regression)…"})
                report = await asyncio.to_thread(train_task_json_model, data_path, output_path)
            elif mode == "tier-json":
                from model_plane.ml.trainer import train_tier_json_model
                output_path = _pl.Path("models/router_ml_tier_v1.json")
                job.emit({"event": "progress", "progress": 20,
                          "message": "Training tier classifier (JSON logistic regression)…"})
                report = await asyncio.to_thread(train_tier_json_model, data_path, output_path)
            elif mode == "deployment":
                from model_plane.ml.trainer import train_model
                output_path = _pl.Path("models/router_ml.joblib")
                job.emit({"event": "progress", "progress": 20,
                          "message": "Training deployment predictor (sklearn)…"})
                report = await asyncio.to_thread(train_model, data_path, output_path, model_type, "selected_deployment")
            else:
                # mode == "tier" — sklearn joblib
                from model_plane.ml.trainer import train_model
                output_path = _pl.Path(str(settings.ml_model_path))
                job.emit({"event": "progress", "progress": 20,
                          "message": f"Training tier classifier ({model_type})…"})
                report = await asyncio.to_thread(train_model, data_path, output_path, model_type, "complexity_tier")

            job.progress = 100
            job.status = "done"
            job.result = {
                "accuracy": report.get("accuracy"),
                "n_samples": report.get("n_samples"),
                "n_classes": report.get("n_classes"),
                "output_path": str(output_path),
                "mode": mode,
            }
            job.emit({"event": "done", "progress": 100, "message": "Training complete",
                      "result": job.result})
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.emit({"event": "error", "progress": job.progress, "message": str(exc)})
            log.warning("retrain_job_failed", job_id=job_id, error=str(exc))

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "queued"}


@router.get("/ml/retrain/{job_id}/stream")
async def ml_retrain_stream(
    job_id: str = Path(...),
    _auth: AdminAuth = None,
) -> StreamingResponse:
    """Stream SSE progress events for a retrain job.

    Events are text/event-stream items:  ``data: <json>\\n\\n``
    Streams until the job reaches status ``done`` or ``failed``.
    """
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def _generate():
        import json as _json
        sent_final = False
        while not sent_final:
            events = job.drain()
            for ev in events:
                yield f"data: {_json.dumps(ev)}\n\n"
                if ev.get("event") in ("done", "error"):
                    sent_final = True
                    break
            if not sent_final:
                if job.status in ("done", "failed"):
                    # Drain any remaining events then exit
                    for ev in job.drain():
                        yield f"data: {_json.dumps(ev)}\n\n"
                    break
                await asyncio.sleep(0.5)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Training data endpoints ───────────────────────────────────────────────────

# Sample CSV rows for the download template (one row per task_type / tier combo)
_SAMPLE_CSV = """timestamp,query,task_type,tier,reasoning_markers,code_presence,simple_indicators,multi_step_patterns,technical_terms,token_count_signal,creative_markers,question_complexity,constraint_count,imperative_verbs,output_format,domain_specificity,reference_complexity,negation_complexity,selected_deployment,latency_ms,cost_usd,validation_result,fallback_used,final_success,feedback_source,quality_score
2026-01-01T00:00:00Z,Write a Python function that sorts a list,code_generation,complex,0.0,0.0,0.0,0.0,0.0,0.001,0.0,0.0,0.0,0.3333,0.0,0.0,0.0,0.0,local-qwen-27b,2200.0,0.002,ok,False,1.0,manual,0.9
2026-01-01T00:01:00Z,Prove by induction that n³ minus n is divisible by 6,mathematical_reasoning,reasoning,0.2,0.0,0.0,0.0,0.0,0.001,0.0,0.0,0.0,0.3333,0.0,0.0,0.0,0.0,local-qwen-27b,1800.0,0.003,ok,False,1.0,manual,0.95
2026-01-01T00:02:00Z,What does HTTP stand for?,simple_qa,simple,0.0,0.0,0.3333,0.0,0.0,0.0006,0.0,0.3333,0.0,0.0,0.0,0.0,0.0,0.0,watsonx-granite-small,400.0,0.00005,ok,False,1.0,manual,0.9
2026-01-01T00:03:00Z,Summarize this document in five bullet points,summarization,medium,0.0,0.0,0.0,0.0,0.0,0.0011,0.0,0.0,0.0,0.3333,0.25,0.0,0.0,0.0,watsonx-granite-medium,1200.0,0.0004,ok,False,1.0,manual,0.85
2026-01-01T00:04:00Z,Create a migration plan from MySQL to PostgreSQL,planning,complex,0.0,0.0,0.0,0.0,0.0,0.001,0.0,0.0,0.0,0.3333,0.0,0.0,0.0,0.0,watsonx-mistral-large,3600.0,0.008,ok,False,1.0,manual,0.8
2026-01-01T00:05:00Z,Compare REST vs GraphQL for a mobile-first API,technical_reasoning,reasoning,0.0,0.0,0.0,0.0,0.0,0.0011,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,local-qwen-27b,3300.0,0.017,ok,False,1.0,manual,0.95
"""

_SAMPLE_JSONL = '\n'.join([
    '{"query":"Write a Python function that sorts a list","task_type":"code_generation","tier":"complex","reasoning_markers":0.0,"code_presence":0.0,"simple_indicators":0.0,"multi_step_patterns":0.0,"technical_terms":0.0,"token_count_signal":0.001,"creative_markers":0.0,"question_complexity":0.0,"constraint_count":0.0,"imperative_verbs":0.3333,"output_format":0.0,"domain_specificity":0.0,"reference_complexity":0.0,"negation_complexity":0.0,"final_success":1.0}',
    '{"query":"Prove by induction that n³ − n is divisible by 6","task_type":"mathematical_reasoning","tier":"reasoning","reasoning_markers":0.2,"code_presence":0.0,"simple_indicators":0.0,"multi_step_patterns":0.0,"technical_terms":0.0,"token_count_signal":0.001,"creative_markers":0.0,"question_complexity":0.0,"constraint_count":0.0,"imperative_verbs":0.3333,"output_format":0.0,"domain_specificity":0.0,"reference_complexity":0.0,"negation_complexity":0.0,"final_success":1.0}',
    '{"query":"What does HTTP stand for?","task_type":"simple_qa","tier":"simple","reasoning_markers":0.0,"code_presence":0.0,"simple_indicators":0.3333,"multi_step_patterns":0.0,"technical_terms":0.0,"token_count_signal":0.0006,"creative_markers":0.0,"question_complexity":0.3333,"constraint_count":0.0,"imperative_verbs":0.0,"output_format":0.0,"domain_specificity":0.0,"reference_complexity":0.0,"negation_complexity":0.0,"final_success":1.0}',
])


@router.get("/ml/training-data/sample")
async def ml_training_data_sample(
    fmt: str = Query(default="csv", pattern="^(csv|jsonl)$"),
    _auth: AdminAuth = None,
):
    """Download a sample training data file showing the expected format.

    ``fmt=csv``   — sample CSV with all columns including the ``query`` text field (default)
    ``fmt=jsonl`` — same structure as CSV but as newline-delimited JSON
    """
    from fastapi.responses import Response
    if fmt == "jsonl":
        return Response(
            content=_SAMPLE_JSONL,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=sample_training_data.jsonl"},
        )
    return Response(
        content=_SAMPLE_CSV,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_training_data.csv"},
    )


@router.get("/ml/training-data")
async def ml_training_data(
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Return a summary of the current training dataset plus a paginated row preview."""
    import pathlib, csv as _csv

    data_path = pathlib.Path("model_plane/data/training_data.csv")
    if not data_path.exists():
        return {
            "total_rows": 0, "columns": [], "rows": [],
            "task_type_counts": {}, "tier_counts": {},
            "file_path": str(data_path),
        }

    with open(data_path, newline="") as fh:
        reader = _csv.DictReader(fh)
        all_rows = list(reader)
        columns = list(reader.fieldnames or [])

    from collections import Counter
    task_counts = dict(Counter(r.get("task_type", "") for r in all_rows if r.get("task_type")).most_common())
    tier_counts  = dict(Counter(r.get("tier", "") for r in all_rows if r.get("tier")).most_common())

    return {
        "total_rows": len(all_rows),
        "columns": columns,
        "task_type_counts": task_counts,
        "tier_counts": tier_counts,
        "rows": all_rows[offset: offset + limit],
        "file_path": str(data_path),
    }


@router.post("/ml/training-data/upload")
async def ml_upload_training_data(
    file: UploadFile,
    merge: bool = Query(
        default=True,
        description="true = append new rows to existing CSV; false = replace the file entirely",
    ),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Upload a CSV or JSONL file to extend (or replace) the training dataset.

    Accepted formats:
    - **CSV** — must have a header row with at least ``task_type`` and ``tier`` columns.
      Extra columns are kept as-is. Include a ``query`` column with the prompt text.
    - **JSONL** — each line is a JSON object with the same column names as the CSV.

    ``merge=true`` (default): new rows are appended; the existing data is preserved.
    ``merge=false``: the file replaces ``training_data.csv`` entirely.

    Returns the new total row count and a breakdown per task_type.
    """
    import pathlib, csv as _csv, json as _json, io
    from collections import Counter

    data_path = pathlib.Path("model_plane/data/training_data.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    filename = (file.filename or "").lower()

    # ── parse uploaded file ───────────────────────────────────────────────────
    new_rows: list[dict] = []
    if filename.endswith(".jsonl") or filename.endswith(".ndjson"):
        for line in content.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    new_rows.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
    else:
        # Default: treat as CSV
        text = content.decode("utf-8", errors="replace")
        reader = _csv.DictReader(io.StringIO(text))
        new_rows = list(reader)

    if not new_rows:
        raise HTTPException(status_code=422, detail="Uploaded file contains no parseable rows.")

    required = {"task_type", "tier"}
    missing = required - set(new_rows[0].keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file is missing required columns: {sorted(missing)}. "
                   f"Found: {sorted(new_rows[0].keys())}",
        )

    # ── canonical CSV columns (superset of required; extras appended at end) ──
    _CANONICAL_COLS = [
        "timestamp", "query", "task_type", "tier",
        "reasoning_markers", "code_presence", "simple_indicators", "multi_step_patterns",
        "technical_terms", "token_count_signal", "creative_markers", "question_complexity",
        "constraint_count", "imperative_verbs", "output_format", "domain_specificity",
        "reference_complexity", "negation_complexity",
        "selected_deployment", "latency_ms", "cost_usd", "validation_result",
        "fallback_used", "final_success", "feedback_source", "quality_score",
    ]

    # Build unified column list: canonical first, then any extras from the upload
    upload_cols = list(new_rows[0].keys())
    extra_cols = [c for c in upload_cols if c not in _CANONICAL_COLS]
    all_cols = _CANONICAL_COLS + extra_cols

    if merge and data_path.exists():
        # Read existing data to preserve it and derive the full column union
        with open(data_path, newline="") as fh:
            existing_reader = _csv.DictReader(fh)
            existing_rows = list(existing_reader)
            existing_cols = list(existing_reader.fieldnames or [])
        # Union columns: keep existing order, append any new ones from upload
        extra_from_upload = [c for c in all_cols if c not in existing_cols]
        all_cols = existing_cols + extra_from_upload
        combined = existing_rows + new_rows
    else:
        combined = new_rows

    # ── write back ────────────────────────────────────────────────────────────
    with open(data_path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)

    task_counts = dict(Counter(r.get("task_type", "") for r in combined if r.get("task_type")).most_common())
    tier_counts  = dict(Counter(r.get("tier", "") for r in combined if r.get("tier")).most_common())

    log.info(
        "training_data_uploaded",
        filename=file.filename,
        new_rows=len(new_rows),
        total_rows=len(combined),
        merge=merge,
    )

    return {
        "accepted_rows": len(new_rows),
        "total_rows": len(combined),
        "merge": merge,
        "task_type_counts": task_counts,
        "tier_counts": tier_counts,
        "file_path": str(data_path),
    }


@router.post("/ml/activate")
async def ml_activate(
    model_path: str = Query(..., description="Path to the joblib model file to activate"),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Switch the active ML model path.

    Updates ``settings.ml_model_path`` in-memory. The change is active immediately
    for all new requests; existing recommender instances are dropped on next call
    to ``get_recommender()``.
    """
    import pathlib
    p = pathlib.Path(model_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    # Reset the cached recommender so it reloads with the new path
    try:
        import model_plane.ml.recommender as _rec_mod
        _rec_mod._recommender = None
    except Exception:
        pass

    # pydantic settings are immutable; update via object.__setattr__
    object.__setattr__(settings, "ml_model_path", p)
    log.info("ml_model_activated", path=model_path)
    return {"status": "ok", "active_model_path": str(p)}


# ── Tenant policy CRUD (Sub-Task 3.3) ────────────────────────────────────────

class TenantPolicy(BaseModel):
    tenant_id: str
    allowed_providers: list[str] = []
    denied_deployments: list[str] = []
    max_cost_per_request: float | None = None
    max_tokens_per_request: int | None = None
    regulated_tasks: list[str] = []
    regulated_deployment: str | None = None


@router.get("/tenants")
async def tenants_list(_auth: AdminAuth) -> dict[str, Any]:
    """Return all tenant policies (in-memory store + routing.yaml snapshot)."""
    # Merge in-memory overrides with routing.yaml tenant_policies
    cfg = settings.load_routing_config()
    yaml_tenants = {t["tenant_id"]: t for t in cfg.get("tenant_policies", [])}
    merged = {**yaml_tenants, **_tenants}
    return {"count": len(merged), "tenants": list(merged.values())}


@router.post("/tenants", status_code=201)
async def tenants_create(body: TenantPolicy, _auth: AdminAuth) -> dict[str, Any]:
    """Add or update a tenant policy (in-memory only; persisted on PUT routing/config)."""
    if body.tenant_id in _tenants:
        raise HTTPException(status_code=409, detail=f"Tenant '{body.tenant_id}' already exists")
    _tenants[body.tenant_id] = body.model_dump()
    return _tenants[body.tenant_id]


@router.put("/tenants/{tenant_id}")
async def tenants_update(
    tenant_id: str = Path(...),
    body: TenantPolicy = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    if tenant_id not in _tenants:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    _tenants[tenant_id] = body.model_dump()
    return _tenants[tenant_id]


@router.delete("/tenants/{tenant_id}", status_code=204)
async def tenants_delete(
    tenant_id: str = Path(...),
    _auth: AdminAuth = None,
) -> None:
    if tenant_id not in _tenants:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    del _tenants[tenant_id]


# ── Alert CRUD (Sub-Task 3.3) ─────────────────────────────────────────────────

class AlertConfig(BaseModel):
    alert_id: str
    name: str
    metric: str                       # e.g. "error_rate", "cost_per_hour", "latency_p99"
    threshold: float
    operator: str = "gt"              # gt | lt | gte | lte
    window_minutes: int = 60
    notification_channel: str = ""   # email | webhook | slack
    notification_target: str = ""    # email address, webhook URL, etc.
    enabled: bool = True


@router.get("/alerts")
async def alerts_list(_auth: AdminAuth) -> dict[str, Any]:
    return {"count": len(_alerts), "alerts": list(_alerts.values())}


@router.post("/alerts", status_code=201)
async def alerts_create(body: AlertConfig, _auth: AdminAuth) -> dict[str, Any]:
    if body.alert_id in _alerts:
        raise HTTPException(status_code=409, detail=f"Alert '{body.alert_id}' already exists")
    _alerts[body.alert_id] = body.model_dump()
    return _alerts[body.alert_id]


@router.put("/alerts/{alert_id}")
async def alerts_update(
    alert_id: str = Path(...),
    body: AlertConfig = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
    _alerts[alert_id] = body.model_dump()
    return _alerts[alert_id]


@router.delete("/alerts/{alert_id}", status_code=204)
async def alerts_delete(
    alert_id: str = Path(...),
    _auth: AdminAuth = None,
) -> None:
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
    del _alerts[alert_id]


# ── Sub-Task 3.4: Classifier status endpoints ────────────────────────────────

# In-memory classifier config overrides (persisted per process lifetime only).
# Backed by settings.classifier_model at startup.
_classifier_overrides: dict[str, Any] = {}

_CLASSIFIER_PATHS = ["regex", "mbert", "laya"]
_CLASSIFIER_META: dict[str, dict[str, Any]] = {
    "regex": {
        "name": "regex",
        "display_name": "Regex Classifier",
        "model_source": "built-in",
        "estimated_latency_ms": 1,
        "description": "Pure regex signal matching — zero dependencies, always available.",
        "install_hint": None,
    },
    "mbert": {
        "name": "mbert",
        "display_name": "BERT NLI Classifier",
        "model_source": "llm-semantic-router/mmbert32k-intent-classifier-merged",
        "estimated_latency_ms": 120,
        "description": "Semantic NLI re-ranking via HuggingFace cross-encoder.",
        "install_hint": "pip install 'model-plane[bert]'  (or run pnpm install)",
    },
    "laya": {
        "name": "laya",
        "display_name": "Laya Classifier",
        "model_source": "laya (local CPU inference)",
        "estimated_latency_ms": 250,
        "description": "IBM Laya local inference — answers task_type and complexity_tier in one call.",
        "install_hint": "pip install 'model-plane[laya]'  (or run pnpm install)",
    },
}

# Availability map: populated lazily on first /classifiers request so we
# don't delay startup when running tests.
_clf_availability: dict[str, bool] | None = None


def _get_clf_availability() -> dict[str, bool]:
    global _clf_availability
    if _clf_availability is None:
        from model_plane.classifier.factory import check_classifier_availability
        _clf_availability = check_classifier_availability()
    return _clf_availability


@router.get("/classifiers")
async def classifiers_list(request: Request, _auth: AdminAuth = None) -> dict[str, Any]:
    """Return all classifier paths with their enabled status and metadata.

    The ``active`` flag reflects the current effective classifier model
    (user/global SQLite override, or ``settings.classifier_model``).
    """
    global _classifier_config_loaded
    if not _classifier_config_loaded:
        _classifier_config_loaded = True
        try:
            _load_classifier_config_from_yaml()
        except Exception as exc:
            log.warning("classifier_config_load_failed", error=str(exc))

    import model_plane.db as _db
    owner_id = _get_routing_owner(request)
    # Check SQLite for user or global classifier overrides
    db_clf: dict[str, Any] = {}
    if owner_id:
        db_clf = _db.load_routing_override("classifier", owner_id)
    if not db_clf:
        db_clf = _db.load_routing_override("classifier", _GLOBAL_OWNER)

    avail = _get_clf_availability()
    active = db_clf.get("active") or _classifier_overrides.get("classifier_model", settings.classifier_model)
    result = []
    for name in _CLASSIFIER_PATHS:
        meta = dict(_CLASSIFIER_META[name])
        meta["active"] = name == active
        # Check db_clf first, then in-memory _classifier_overrides, then default
        enabled_val = db_clf.get(f"{name}_enabled")
        if enabled_val is None:
            enabled_val = _classifier_overrides.get(f"{name}_enabled", name == "regex")
        meta["enabled"] = bool(enabled_val)
        # available=True for regex always; for others check if the package is installed
        meta["available"] = avail.get(name, True) if name != "regex" else True
        result.append(meta)
    return {"classifiers": result, "active_classifier": active}


def _persist_classifier_config(owner_id: str = "") -> None:
    """Write the current classifier overrides into SQLite routing_overrides.

    Persists to SQLite routing_overrides table under (namespace='classifier', owner_id).
    Does NOT touch routing.yaml.
    """
    import model_plane.db as _db

    active = _classifier_overrides.get("classifier_model", settings.classifier_model)
    clf_cfg: dict[str, Any] = {"active": active}
    for n in _CLASSIFIER_PATHS:
        if n == "regex":
            continue
        key = f"{n}_enabled"
        if key in _classifier_overrides:
            clf_cfg[f"{n}_enabled"] = _classifier_overrides[key]

    # Save to SQLite for the active user (or global if unauthenticated)
    target_owner = owner_id or _GLOBAL_OWNER
    _db.save_routing_override("classifier", clf_cfg, target_owner)
    log.info("classifier_config_persisted", active=active, owner=target_owner)


def _load_classifier_config_from_yaml() -> None:
    """Restore classifier overrides from SQLite / routing.yaml on startup."""
    import model_plane.db as _db

    # Check SQLite first
    global_clf = _db.load_routing_override("classifier", _GLOBAL_OWNER)
    if global_clf:
        active = global_clf.get("active")
        if active and active in _CLASSIFIER_PATHS:
            _classifier_overrides["classifier_model"] = active
            try:
                object.__setattr__(settings, "classifier_model", active)
            except Exception:
                pass
        for n in _CLASSIFIER_PATHS:
            if n == "regex":
                continue
            key = f"{n}_enabled"
            if key in global_clf:
                _classifier_overrides[key] = bool(global_clf[key])
        return

    import pathlib

    path = pathlib.Path(settings.routing_config_path)
    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return

    clf_cfg: dict[str, Any] = raw.get("classifier", {})
    if not clf_cfg:
        return

    active = clf_cfg.get("active")
    if active and active in _CLASSIFIER_PATHS:
        _classifier_overrides["classifier_model"] = active
        try:
            object.__setattr__(settings, "classifier_model", active)
        except Exception:
            pass

    for n in _CLASSIFIER_PATHS:
        if n == "regex":
            continue
        key = f"{n}_enabled"
        if key in clf_cfg:
            _classifier_overrides[key] = bool(clf_cfg[key])


_classifier_config_loaded: bool = False


@router.put("/classifiers/{name}")
async def classifiers_update(
    request: Request,
    name: str = Path(..., description="Classifier name: regex | mbert | laya"),
    body: dict[str, Any] = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Enable or disable a classifier path, or set it as the active one.

    Body fields (all optional):
    - ``enabled``: bool — flip enabled flag.
    - ``set_active``: bool — make this the active classifier.

    Changes are applied to SQLite in the routing_overrides table under the user's
    owner_id and __global__, as well as in-memory settings.
    """
    if name not in _CLASSIFIER_META:
        raise HTTPException(status_code=404, detail=f"Classifier '{name}' not found")

    if "enabled" in body:
        _classifier_overrides[f"{name}_enabled"] = bool(body["enabled"])

    if body.get("set_active"):
        _classifier_overrides["classifier_model"] = name
        # Also propagate to settings object for the running process
        try:
            object.__setattr__(settings, "classifier_model", name)
        except Exception:
            pass  # best-effort; settings may be frozen

    owner_id = _get_routing_owner(request)
    # Persist every change to SQLite and routing.yaml
    try:
        _persist_classifier_config(owner_id=owner_id)
    except Exception as exc:
        log.warning("classifier_config_persist_failed", error=str(exc))

    active = _classifier_overrides.get("classifier_model", settings.classifier_model)
    meta = dict(_CLASSIFIER_META[name])
    meta["active"] = name == active
    meta["enabled"] = _classifier_overrides.get(f"{name}_enabled", name == "regex")
    return {"updated": meta, "active_classifier": active}


# ── Security & Governance API (cache-aware routing PR) ────────────────────────

_GLOBAL_OWNER = "__global__"


@router.get("/security/summary")
async def security_summary(request: Request, _auth: AdminAuth = None) -> dict[str, Any]:
    """Return security configuration and a live snapshot of recent security events.

    Used by the Security & Governance page in the admin UI.
    """
    from model_plane.routing.security import (
        DataSensitivity,
        ProviderTrust,
        _DEFAULT_PROVIDER_TRUST,
        _POLICY_MATRIX,
        get_security_guard,
    )

    # ── Policy matrix ──────────────────────────────────────────────────────────
    matrix_rows = []
    for sensitivity in DataSensitivity:
        row: dict[str, Any] = {"sensitivity": sensitivity.value}
        for trust in ProviderTrust:
            row[trust.value] = _POLICY_MATRIX.get((sensitivity, trust), False)
        matrix_rows.append(row)

    # ── Provider trust map (current + any routing.yaml overrides) ─────────────
    routing_cfg = settings.load_routing_config()
    trust_overrides = routing_cfg.get("provider_trust_overrides", {})
    import model_plane.db as _db
    owner_id = _get_routing_owner(request)
    # Check DB routing rules overrides
    if owner_id:
        db_rules = _db.load_routing_override("routing_rules", owner_id)
        if "provider_trust_overrides" in db_rules:
            trust_overrides = {**trust_overrides, **db_rules["provider_trust_overrides"]}
    guard = get_security_guard(trust_overrides or None, owner_id=owner_id or _GLOBAL_OWNER)

    provider_trust_map = {}
    catalog = get_catalog()
    seen_providers: set[str] = set()
    for dep in catalog.deployments.values():
        seen_providers.add(dep.provider)
    # Include all known providers even if not in catalog
    for prov in list(_DEFAULT_PROVIDER_TRUST.keys()) + list(seen_providers):
        provider_trust_map[prov] = guard.provider_trust(prov).value

    # ── Recent security events from request buffer ─────────────────────────────
    buf = get_request_buffer()
    all_recs = buf.snapshot(limit=500)
    events: list[dict] = []
    sensitivity_counts: dict[str, int] = {}
    blocked_counts: dict[str, int] = {}
    for rec in all_recs:
        sens = rec.get("security_sensitivity")
        if not sens or sens == "disabled":
            continue
        sensitivity_counts[sens] = sensitivity_counts.get(sens, 0) + 1
        blocked = rec.get("security_blocked") or []
        for dep_name in blocked:
            blocked_counts[dep_name] = blocked_counts.get(dep_name, 0) + 1
        if sens in ("internal", "confidential", "restricted") or blocked:
            pii_list = rec.get("pii_types") or []
            sec_list = rec.get("secret_types") or []
            signals_list = []
            if pii_list:
                signals_list.append(f"PII: {', '.join(pii_list)}")
            if sec_list:
                signals_list.append(f"Secrets: {', '.join(sec_list)}")
            reason = " | ".join(signals_list) if signals_list else (
                f"Elevated sensitivity ({sens.upper()})" if sens != "public" else "Policy restricted"
            )

            events.append({
                "timestamp": rec.get("timestamp") or rec.get("ts"),
                "request_id": rec.get("id") or rec.get("request_id"),
                "sensitivity": sens,
                "blocked": blocked,
                "reason": reason,
                "pii_types": pii_list,
                "secret_types": sec_list,
                "provider": rec.get("provider"),
                "deployment": rec.get("deployment"),
                "context_reuse_score": rec.get("context_reuse_score"),
            })

    return {
        "enabled": settings.security_routing_enabled,
        "policy_matrix": matrix_rows,
        "provider_trust_map": provider_trust_map,
        "trust_overrides": trust_overrides,
        "sensitivity_counts": sensitivity_counts,
        "blocked_counts": blocked_counts,
        "recent_events": events[:100],
        "total_requests_sampled": len(all_recs),
    }


@router.get("/security/flags")
async def security_flags(request: Request, _auth: AdminAuth) -> dict[str, Any]:
    """Return smart-routing feature flag values merged with the user's DB overrides."""
    import model_plane.db as _db
    base = {
        "security_routing_enabled": settings.security_routing_enabled,
        "context_reuse_enabled": settings.context_reuse_enabled,
        "context_reuse_weight": settings.context_reuse_weight,
        "context_reuse_token_threshold": settings.context_reuse_token_threshold,
        "cache_mode": settings.cache_mode,
        "compression_enabled": settings.compression_enabled,
        "compression_token_threshold": settings.compression_token_threshold,
        # Cache-aware routing level flags
        "cache_session_affinity_enabled": settings.cache_session_affinity_enabled,
        "cache_session_affinity_min_tokens": settings.cache_session_affinity_min_tokens,
        "cache_content_hash_routing_enabled": settings.cache_content_hash_routing_enabled,
        "cache_content_hash_max_sticky": settings.cache_content_hash_max_sticky,
    }
    owner_id = _get_routing_owner(request)
    # Check global first, then user-specific override
    global_overrides = _db.load_routing_override("smart_routing", _PII_RULES_GLOBAL_OWNER)
    base = {**base, **global_overrides}
    if owner_id:
        overrides = _db.load_routing_override("smart_routing", owner_id)
        base = {**base, **overrides}
    return base


@router.put("/security/flags")
async def security_flags_put(
    request: Request,
    body: dict[str, Any] = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Save smart-routing feature flag overrides to SQLite for the calling user.

    Accepts any subset of the flags returned by GET /security/flags.
    Changes are user-scoped, take effect immediately for this user's requests,
    and survive restart.  routing.yaml is not touched.
    """
    import model_plane.db as _db
    _ALLOWED_KEYS = {
        "security_routing_enabled", "context_reuse_enabled",
        "context_reuse_weight", "context_reuse_token_threshold",
        "cache_mode",
        # Cache-aware routing level flags
        "cache_session_affinity_enabled", "cache_session_affinity_min_tokens",
        "cache_content_hash_routing_enabled", "cache_content_hash_max_sticky",
    }
    clean = {k: v for k, v in body.items() if k in _ALLOWED_KEYS}
    owner_id = _get_routing_owner(request)
    if clean:
        target_owner = owner_id if owner_id else _PII_RULES_GLOBAL_OWNER
        _db.save_routing_override("smart_routing", clean, target_owner)
        # Also update in-process settings so requests in this server process
        # use the new values immediately (best-effort; settings may be frozen)
        for k, v in clean.items():
            try:
                object.__setattr__(settings, k, v)
            except Exception:
                pass
        log.info("smart_routing_flags_updated", keys=list(clean.keys()), owner=target_owner)
    return {**clean, "status": "ok"}


# ── PII Rules API ─────────────────────────────────────────────────────────────

_PII_RULES_GLOBAL_OWNER = "__global__"


@router.get("/security/pii-rules")
async def pii_rules_get(request: Request, _auth: AdminAuth) -> dict[str, Any]:
    """Return current disabled PII rule IDs — global config merged with per-user overrides."""
    import model_plane.db as _db
    # Merge global + per-user (per-user wins on conflict)
    disabled: set[str] = set()
    global_cfg = _db.load_routing_override("pii_rules", _PII_RULES_GLOBAL_OWNER)
    disabled.update(global_cfg.get("disabled", []))
    owner_id = _get_routing_owner(request)
    if owner_id:
        user_cfg = _db.load_routing_override("pii_rules", owner_id)
        disabled.update(user_cfg.get("disabled", []))
    return {"disabled": sorted(disabled)}


@router.put("/security/pii-rules")
async def pii_rules_put(
    request: Request,
    body: dict[str, Any] = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Persist disabled PII rule IDs.

    Body: ``{"disabled": ["IP_ADDRESS", "PHONE"]}``
    Always saves under the global key so rules apply to ALL requests regardless
    of tenant.  Also saves under the per-user key when a JWT owner is present.
    Changes take effect immediately — no restart needed.
    """
    import model_plane.db as _db
    disabled = body.get("disabled", [])
    if not isinstance(disabled, list):
        raise HTTPException(status_code=422, detail="'disabled' must be a list of rule IDs")
    owner_id = _get_routing_owner(request)
    target_owner = owner_id if owner_id else _PII_RULES_GLOBAL_OWNER
    _db.save_routing_override("pii_rules", {"disabled": disabled}, target_owner)
    log.info("pii_rules_updated", disabled=disabled, owner=target_owner)
    return {"disabled": disabled, "status": "ok"}


# ── Compression Config & Metrics API ─────────────────────────────────────────

from model_plane.runtime_overrides import compression_overrides as _compression_overrides


@router.get("/compression/config")
async def compression_config_get(request: Request, _auth: AdminAuth) -> dict[str, Any]:
    """Return current compression configuration, merged with per-user DB overrides."""
    import model_plane.db as _db
    base = {
        "enabled": _compression_overrides.get("enabled", settings.compression_enabled),
        "profile": _compression_overrides.get("profile", "auto"),
        "token_threshold": _compression_overrides.get(
            "token_threshold", settings.compression_token_threshold
        ),
        "cache_aware_gate_enabled": _compression_overrides.get(
            "cache_aware_gate_enabled", settings.context_reuse_enabled
        ),
        "cache_aware_threshold": _compression_overrides.get("cache_aware_threshold", 0.60),
        "available_profiles": [
            "auto",
            "passthrough",
            "tool_output_compaction",
            "conversation_summary",
            "code_context_reduction",
        ],
    }
    owner_id = _get_routing_owner(request)
    if owner_id:
        user_overrides = _db.load_routing_override("compression", owner_id)
        base = {**base, **{k: v for k, v in user_overrides.items() if k != "available_profiles"}}
    return base


@router.put("/compression/config")
async def compression_config_put(
    request: Request,
    body: dict[str, Any] = Body(...),
    _auth: AdminAuth = None,
) -> dict[str, Any]:
    """Update compression settings live — no restart required.

    Changes are persisted per-user to SQLite (routing_overrides namespace
    'compression') and also applied to the in-process shared dict so they
    take effect immediately for this server process.

    Accepted fields (all optional):
    - ``enabled`` (bool) — master on/off switch
    - ``profile`` (str) — one of: auto | passthrough | tool_output_compaction |
      conversation_summary | code_context_reduction
    - ``token_threshold`` (int) — token count above which auto-profile kicks in
    - ``cache_aware_gate_enabled`` (bool) — skip compression when session is warm
    - ``cache_aware_threshold`` (float 0–1) — context_reuse_score above which to skip
    """
    import model_plane.db as _db
    from model_plane.compression.processor import PROFILES

    valid_profiles = list(PROFILES.keys()) + ["auto"]
    patch: dict[str, Any] = {}
    if "enabled" in body:
        patch["enabled"] = _compression_overrides["enabled"] = bool(body["enabled"])
    if "profile" in body:
        p = str(body["profile"])
        if p not in valid_profiles:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown profile '{p}'. Valid: {valid_profiles}",
            )
        patch["profile"] = _compression_overrides["profile"] = p
    if "token_threshold" in body:
        patch["token_threshold"] = _compression_overrides["token_threshold"] = int(body["token_threshold"])
    if "cache_aware_gate_enabled" in body:
        patch["cache_aware_gate_enabled"] = _compression_overrides["cache_aware_gate_enabled"] = bool(body["cache_aware_gate_enabled"])
    if "cache_aware_threshold" in body:
        patch["cache_aware_threshold"] = _compression_overrides["cache_aware_threshold"] = float(body["cache_aware_threshold"])

    # Persist to DB per user so changes survive restart
    owner_id = _get_routing_owner(request)
    if owner_id and patch:
        _db.save_routing_override("compression", patch, owner_id)

    log.info("compression_config_updated", overrides=_compression_overrides, owner=owner_id)
    return {
        "status": "ok",
        "config": {
            "enabled": _compression_overrides.get("enabled", settings.compression_enabled),
            "profile": _compression_overrides.get("profile", "auto"),
            "token_threshold": _compression_overrides.get(
                "token_threshold", settings.compression_token_threshold
            ),
            "cache_aware_gate_enabled": _compression_overrides.get(
                "cache_aware_gate_enabled", settings.context_reuse_enabled
            ),
            "cache_aware_threshold": _compression_overrides.get("cache_aware_threshold", 0.60),
        },
    }


@router.get("/compression/metrics")
async def compression_metrics(_auth: AdminAuth) -> dict[str, Any]:
    """Return compression savings metrics from the last 500 buffered requests."""
    buf = get_request_buffer()
    all_recs = buf.snapshot(limit=500)

    total = len(all_recs)
    compressed_count = 0
    cache_skip_count = 0
    total_tokens_before = 0
    total_tokens_after = 0
    total_savings = 0
    profile_counts: dict[str, int] = {}

    for rec in all_recs:
        profile = rec.get("compression_profile") or "passthrough"
        tb = rec.get("tokens_before_compression") or 0
        ta = rec.get("tokens_after_compression") or tb
        savings = max(0, tb - ta)

        profile_counts[profile] = profile_counts.get(profile, 0) + 1

        if tb > 0:
            total_tokens_before += tb
            total_tokens_after += ta
            total_savings += savings
            if profile != "passthrough":
                compressed_count += 1
            # A cache-aware skip: passthrough but tokens are unchanged and
            # context_reuse_score was high enough
            if profile == "passthrough" and (rec.get("context_reuse_score") or 0) >= 0.60:
                cache_skip_count += 1

    overall_savings_pct = (
        round((1 - total_tokens_after / total_tokens_before) * 100, 1)
        if total_tokens_before > 0 else 0.0
    )

    return {
        "total_requests_sampled": total,
        "compressed_count": compressed_count,
        "cache_skip_count": cache_skip_count,
        "total_tokens_before": total_tokens_before,
        "total_tokens_after": total_tokens_after,
        "total_savings_tokens": total_savings,
        "overall_savings_pct": overall_savings_pct,
        "profile_distribution": profile_counts,
    }
