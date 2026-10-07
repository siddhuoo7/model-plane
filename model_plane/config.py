"""Centralised application settings loaded from environment variables and config files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, Field, model_validator
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

    @model_validator(mode="after")
    def validate_production_jwt_secret(self) -> "Settings":
        if self.env == "production" and self.jwt_secret == "change-me-in-production":
            raise ValueError(
                "MODEL_PLANE_JWT_SECRET must be set to a secure secret in production environments."
            )
        return self

    # ── config files ─────────────────────────────────────────────────────────
    model_catalog_path: Path = Path("config/models.yaml")
    routing_config_path: Path = Path("config/routing.yaml")
    ml_model_path: Path = Path("models/router_ml_tier_v1.joblib")
    # Set to "" to disable price enrichment from prices.yaml
    prices_yaml_path: str = "config/prices.yaml"

    # ── admin API ─────────────────────────────────────────────────────────────
    # Bearer token that gates all /admin/api/* endpoints.
    # Leave unset (or empty) to disable auth during local development.
    admin_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ADMIN_API_KEY", "MODEL_PLANE_ADMIN_API_KEY"),
    )
    admin_buffer_size: int = 10_000  # max routing records kept in the ring buffer

    # ── redis (optional) ─────────────────────────────────────────────────────
    redis_url: str | None = None
    cache_ttl_seconds: int = 3600

    # ── routing behaviour ────────────────────────────────────────────────────
    default_model: str = "watsonx-granite-small"
    routing_mode: Literal["fixed", "auto_router", "custom_plugin", "hybrid"] = "custom_plugin"
    ml_routing_enabled: bool = False
    ml_confidence_threshold: float = 0.70
    shadow_traffic_fraction: float = 0.05
    # "tier"       → model predicts ComplexityTier (default, new artefact)
    # "deployment" → model predicts deployment name (legacy rollback)
    ml_output_mode: str = "tier"

    # ── cache economics ───────────────────────────────────────────────────────
    # "soft_preference" → existing heuristic (stay if context_tokens > 2000)
    # "switch_cost"     → Sub-Task 4.1 USD switch-cost model
    cache_mode: Literal["soft_preference", "switch_cost"] = "soft_preference"

    # ── security routing (Stage 0.5) ──────────────────────────────────────────
    # Master switch for the SecurityGuard hard-eligibility filter.
    # When false the pipeline behaves exactly as before this feature was added.
    security_routing_enabled: bool = True

    # ── context reuse scoring (Stage 6e) ──────────────────────────────────────
    # Master switch for the ContextReuseScore soft re-rank.
    context_reuse_enabled: bool = True
    # Maximum score adjustment a candidate can receive from context-reuse bonus.
    # 0.20 means the bonus can shift a candidate's rank by at most 0.20 points.
    context_reuse_weight: float = 0.20
    # Accumulated-token threshold above which CacheValueWeight saturates at 1.0.
    context_reuse_token_threshold: int = 2000

    # ── cache-aware routing levels ────────────────────────────────────────────
    # Level 1 — Session affinity: route same session_id to same deployment.
    # Highest ROI; implemented via session-state lookup before tier scoring.
    cache_session_affinity_enabled: bool = True
    # Minimum accumulated-token count before session affinity is enforced.
    # Requests below this threshold proceed through normal tier scoring.
    cache_session_affinity_min_tokens: int = 500

    # Level 3 — Content-hash routing: route requests that share the same
    # retrieved-document prefix hash to the same deployment.
    # High ROI for RAG workloads; requires prefix_hash to be set on the context.
    cache_content_hash_routing_enabled: bool = True

    # Maximum active sticky-routing sessions per content-hash before we stop
    # enforcing affinity (prevents one hot document starving other candidates).
    cache_content_hash_max_sticky: int = 8

    # ── feature flags ────────────────────────────────────────────────────────
    features_15_dim_enabled: bool = False  # append task_type_tier_signal as dim[14]

    # ── classifier backend selector ───────────────────────────────────────────
    # CLASSIFIER_MODEL=regex   → pure regex (default, zero deps)
    # CLASSIFIER_MODEL=mbert   → regex + BERT NLI (overrides task_type, re-derives tier)
    # CLASSIFIER_MODEL=laya    → regex + Laya (overrides task_type AND complexity_tier)
    classifier_model: str = "regex"
    classifier_min_confidence: float = 0.60  # below this, regex result is kept
    # Task classifier — primary joblib (sklearn GBT/RF), backup JSON (logistic regression)
    ml_task_classifier_path: str = "models/classifier_task.joblib"
    ml_task_classifier_json_path: str = "models/classifier_ml.json"

    # ── bert classifier (deprecated — use CLASSIFIER_MODEL=mbert) ─────────────
    # bert_classifier_enabled=true is treated as CLASSIFIER_MODEL=mbert when
    # classifier_model is still "regex" (backward compatibility alias).
    bert_classifier_enabled: bool = False
    bert_classifier_model: str = "cross-encoder/nli-MiniLM2-L6-H768"
    bert_classifier_min_confidence: float = 0.60  # kept for backward compat; superseded by classifier_min_confidence

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
    requests_ca_bundle: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REQUESTS_CA_BUNDLE", "MODEL_PLANE_REQUESTS_CA_BUNDLE"),
    )
    ssl_cert_file: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SSL_CERT_FILE", "MODEL_PLANE_SSL_CERT_FILE"),
    )
    local_vllm_insecure_skip_verify: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LOCAL_VLLM_INSECURE_SKIP_VERIFY",
            "MODEL_PLANE_LOCAL_VLLM_INSECURE_SKIP_VERIFY",
        ),
    )
    local_vllm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LOCAL_VLLM_API_KEY",
            "MODEL_PLANE_LOCAL_VLLM_API_KEY",
        ),
    )

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    aws_access_key_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_ACCESS_KEY_ID", "MODEL_PLANE_AWS_ACCESS_KEY_ID"),
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_SECRET_ACCESS_KEY", "MODEL_PLANE_AWS_SECRET_ACCESS_KEY"),
    )
    aws_region_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_REGION_NAME", "AWS_DEFAULT_REGION", "MODEL_PLANE_AWS_REGION_NAME"),
    )
    # API-key-only auth for Bedrock (LiteLLM reads AWS_BEARER_TOKEN_BEDROCK).
    # Set this instead of AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY when you
    # only have a Bedrock bearer token (e.g. from AWS Console → "API keys").
    aws_bearer_token_bedrock: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AWS_BEARER_TOKEN_BEDROCK",
            "MODEL_PLANE_AWS_BEARER_TOKEN_BEDROCK",
        ),
    )

    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "MODEL_PLANE_AZURE_API_KEY"),
    )
    azure_api_base: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT", "MODEL_PLANE_AZURE_API_BASE"),
    )
    azure_api_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AZURE_API_VERSION", "MODEL_PLANE_AZURE_API_VERSION"),
    )

    # ── Google Vertex AI ──────────────────────────────────────────────────────
    vertex_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VERTEXAI_PROJECT", "VERTEX_PROJECT", "MODEL_PLANE_VERTEX_PROJECT"),
    )
    vertex_location: str | None = Field(
        default=None,
        validation_alias=AliasChoices("VERTEXAI_LOCATION", "VERTEX_LOCATION", "MODEL_PLANE_VERTEX_LOCATION"),
    )

    # ── Cohere ────────────────────────────────────────────────────────────────
    cohere_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COHERE_API_KEY", "MODEL_PLANE_COHERE_API_KEY"),
    )

    # ── Mistral ───────────────────────────────────────────────────────────────
    mistral_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MISTRAL_API_KEY", "MODEL_PLANE_MISTRAL_API_KEY"),
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

    def load_prices(self) -> dict:
        """Load prices.yaml; returns {} if path is unset or file is missing."""
        if not self.prices_yaml_path:
            return {}
        path = Path(self.prices_yaml_path)
        if not path.exists():
            return {}
        with open(path) as fh:
            return yaml.safe_load(fh) or {}


# Module-level singleton — imported by all other modules.
settings = Settings()
