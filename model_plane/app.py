"""Model Plane FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_plane.adapters.admin_router import router as admin_router
from model_plane.adapters.auth import _load_key_hashes
from model_plane.adapters.chat_router import router as chat_router
from model_plane.adapters.messages_router import router as messages_router
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ── startup ──────────────────────────────────────────────────────────────
    configure_logging()
    log.info("model_plane_starting", version="0.1.0", env=settings.env)

    # Backfill provider credentials into os.environ so LiteLLM's internal
    # get_secret_str() calls find them (it reads os.environ, not pydantic settings).
    _sync_provider_env()

    _load_key_hashes()
    get_catalog()            # pre-load catalog
    register_hooks()         # register LiteLLM hooks

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

    # Prometheus metrics endpoint
    if settings.metrics_enabled:
        try:
            from prometheus_client import make_asgi_app

            metrics_app = make_asgi_app()
            app.mount(settings.metrics_path, metrics_app)
        except ImportError:
            log.warning("prometheus_client_not_installed_metrics_disabled")

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
