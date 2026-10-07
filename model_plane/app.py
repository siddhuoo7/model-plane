"""Model Plane FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_plane.adapters.admin_api_router import router as admin_api_router
from model_plane.adapters.admin_router import router as admin_router
from model_plane.adapters.auth import _load_key_hashes
from model_plane.adapters.chat_router import router as chat_router
from model_plane.adapters.messages_router import router as messages_router
from model_plane.adapters.settings_router import bootstrap_credentials_from_db, router as settings_router
from model_plane.config import settings
from model_plane.hooks.litellm_hooks import register_hooks
from model_plane.logging_setup import configure_logging, get_logger
from model_plane.registry.catalog import get_catalog

log = get_logger(__name__)


def _sync_provider_env() -> None:
    """Copy provider credentials from pydantic settings into os.environ.

    LiteLLM's get_secret_str() reads os.environ directly. Pydantic-settings
    loads .env into the Settings object but does NOT write values back into
    os.environ, so we do it explicitly here at startup.
    """
    mapping = {
        "WATSONX_APIKEY": settings.watsonx_api_key,     # canonical LiteLLM name
        "WATSONX_PROJECT_ID": settings.watsonx_project_id,
        "WATSONX_URL": settings.watsonx_url,
        "OPENAI_API_KEY": settings.openai_api_key,
        "ANTHROPIC_API_KEY": settings.anthropic_api_key,
    }
    synced = []
    for key, value in mapping.items():
        if value:
            os.environ[key] = value   # overwrite — always wins over shell env
            synced.append(key)
    log.info("provider_env_synced", keys=synced,
             watsonx_project_id_set=bool(settings.watsonx_project_id))


def _probe_local_deployments() -> None:
    """Synchronously probe every local/vLLM deployment at startup.

    Marks unreachable local deployments as unhealthy so the routing pipeline
    skips them immediately rather than discovering the failure mid-request.
    Only runs for provider == 'local'; remote providers are not probed at
    startup to avoid blocking the server boot.
    """
    from model_plane.adapters.providers import get_adapter
    catalog = get_catalog()
    for dep in catalog.deployments.values():
        if dep.provider != "local":
            continue
        adapter = get_adapter(dep.provider)
        try:
            ok = adapter.health_probe(dep)
            if not ok:
                catalog.mark_unhealthy(dep.name)
                log.warning("local_deployment_unhealthy_at_startup", deployment=dep.name)
        except Exception as exc:
            catalog.mark_unhealthy(dep.name)
            log.warning("local_deployment_probe_failed", deployment=dep.name, error=str(exc))


def _validate_catalog_against_providers() -> None:
    """For each provider that has credentials, call list_models() and mark
    deployments whose model ID does not appear in the live list as unhealthy.

    Runs in a background thread at startup so it does not block server boot.
    Only providers with credentials configured are probed.
    Remote calls that fail (network error, bad key) are silently ignored —
    the existing healthy flag is kept so routing continues to work.
    """
    from model_plane.adapters.providers import _REGISTRY
    from model_plane.provider_creds import provider_has_creds

    catalog = get_catalog()

    for provider_name, adapter in _REGISTRY.items():
        if not provider_has_creds(provider_name):
            continue
        if not hasattr(adapter, "list_models"):
            continue
        try:
            live_models = adapter.list_models()
        except Exception as exc:
            log.debug("startup_list_models_failed", provider=provider_name, error=str(exc))
            continue

        if not live_models:
            log.debug("startup_list_models_empty", provider=provider_name)
            continue

        live_set = {m.lower().strip() for m in live_models}

        def _in_live(litellm_model: str) -> bool:
            bare = litellm_model.lower().strip()
            if bare in live_set:
                return True
            if "/" in bare:
                stripped = bare.split("/", 1)[-1]
                if stripped in live_set:
                    return True
                if any(stripped in m or m in stripped for m in live_set):
                    return True
            if any(bare in m or m in bare for m in live_set):
                return True
            return False

        for dep in catalog.deployments.values():
            if dep.provider != provider_name:
                continue
            if _in_live(dep.litellm_model):
                dep.healthy = True
                log.debug("startup_model_available", deployment=dep.name, provider=provider_name)
            else:
                dep.healthy = False
                log.warning(
                    "startup_model_not_in_provider_list",
                    deployment=dep.name,
                    provider=provider_name,
                    litellm_model=dep.litellm_model,
                )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── startup ──────────────────────────────────────────────────────────────
    configure_logging()
    log.info("model_plane_starting", version="0.1.0", env=settings.env)

    # Bootstrap SQLite-persisted credentials into os.environ/settings first,
    # then fill any remaining gaps from the pydantic settings object (.env fallback).
    bootstrap_credentials_from_db()
    _sync_provider_env()

    _load_key_hashes()
    get_catalog()            # pre-load catalog
    _probe_local_deployments()  # mark unreachable local vLLM as unhealthy immediately
    register_hooks()         # register LiteLLM hooks

    # Validate catalog against live provider model lists in a background thread.
    # This marks deployments whose model ID is not returned by the provider's
    # /models API as unhealthy so the UI shows them as unavailable immediately.
    import threading as _threading
    _threading.Thread(
        target=_validate_catalog_against_providers,
        name="catalog-validation",
        daemon=True,
    ).start()

    # Check optional classifier backends; logs warnings for any missing deps
    from model_plane.classifier.factory import check_classifier_availability
    clf_avail = check_classifier_availability()
    log.info("classifier_availability", **clf_avail)

    # Init session cache (will connect to Redis if configured)
    from model_plane.cache.session import get_session_cache
    get_session_cache()

    if settings.ml_routing_enabled:
        from model_plane.ml.recommender import get_recommender
        get_recommender()

    log.info("model_plane_ready", host=settings.host, port=settings.port)
    yield

    # ── shutdown ─────────────────────────────────────────────────────────────
    log.info("model_plane_shutting_down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Model Plane",
        description="Production-grade model control plane and smart router",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "production" else None,
        redoc_url="/redoc" if settings.env != "production" else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(messages_router)
    app.include_router(admin_router)
    app.include_router(admin_api_router)
    app.include_router(settings_router)

    # Prometheus metrics endpoint
    if settings.metrics_enabled:
        try:
            from prometheus_client import make_asgi_app

            metrics_app = make_asgi_app()
            app.mount(settings.metrics_path, metrics_app)
        except ImportError:
            log.warning("prometheus_client_not_installed_metrics_disabled")

    # Admin UI static files (Sub-Task 3.1 placeholder; replaced by React build in 3.4)
    ui_dist = __import__("pathlib").Path("model_plane/ui/dist")
    if ui_dist.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/admin", StaticFiles(directory=str(ui_dist), html=True), name="admin-ui")

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
