"""Centralised application settings loaded from environment variables and config files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODEL_PLANE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── server ──────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8081
    workers: int = 1
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    env: Literal["development", "staging", "production"] = "development"

    # ── auth ─────────────────────────────────────────────────────────────────
    # Declared as plain str to prevent pydantic-settings v2 from trying to
    # JSON-decode the raw env value (it does so for list/set fields).
    # _split_api_keys() below converts it to list[str] at runtime.
    api_keys: str = Field(default="")
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"

    # ── config files ─────────────────────────────────────────────────────────
    model_catalog_path: Path = Path("config/models.yaml")
    routing_config_path: Path = Path("config/routing.yaml")
    ml_model_path: Path = Path("models/router_ml.joblib")

    # ── redis (optional) ─────────────────────────────────────────────────────
    redis_url: str | None = None
    cache_ttl_seconds: int = 3600

    # ── routing behaviour ────────────────────────────────────────────────────
    default_model: str = "watsonx-granite-small"
    routing_mode: Literal["fixed", "auto_router", "custom_plugin", "hybrid"] = "custom_plugin"
    ml_routing_enabled: bool = False
    ml_confidence_threshold: float = 0.70
    shadow_traffic_fraction: float = 0.05

    # ── similarity routing ────────────────────────────────────────────────────
    similarity_routing_enabled: bool = False
    similarity_confidence_threshold: float = 0.55
    similarity_min_hits: int = 3
    similarity_top_k: int = 5

    # ── compression ───────────────────────────────────────────────────────────
    compression_enabled: bool = True
    compression_token_threshold: int = 6000  # tokens before compression kicks in

    # ── observability ────────────────────────────────────────────────────────
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    trace_enabled: bool = False

    # ── providers ────────────────────────────────────────────────────────────
    # AliasChoices lets these be read without the MODEL_PLANE_ prefix,
    # matching the standard env var names expected by LiteLLM.
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "MODEL_PLANE_OPENAI_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "MODEL_PLANE_ANTHROPIC_API_KEY"),
    )
    # LiteLLM docs: the env var is WATSONX_APIKEY (no underscore before KEY)
    # We accept both spellings from .env for user convenience.
    watsonx_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WATSONX_APIKEY",           # LiteLLM canonical name
            "WATSONX_API_KEY",          # common alternative
            "MODEL_PLANE_WATSONX_API_KEY",
        ),
    )
    watsonx_project_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("WATSONX_PROJECT_ID", "MODEL_PLANE_WATSONX_PROJECT_ID"),
    )
    watsonx_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("WATSONX_URL", "MODEL_PLANE_WATSONX_URL"),
    )

    @property
    def api_key_list(self) -> list[str]:
        """Return api_keys split on commas, filtering blanks."""
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    def load_model_catalog(self) -> dict:
        path = self.model_catalog_path
        if not path.exists():
            return {}
        with open(path) as fh:
            return yaml.safe_load(fh) or {}

    def load_routing_config(self) -> dict:
        path = self.routing_config_path
        if not path.exists():
            return {}
        with open(path) as fh:
            return yaml.safe_load(fh) or {}


# Module-level singleton — imported by all other modules.
settings = Settings()
