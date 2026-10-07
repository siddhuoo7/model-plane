"""Shared provider credential checker.

Used by both the model catalog (routing-time filtering) and the admin API
(health-probe / UI display).  Lives at the top of the model_plane package so
it can be imported by model_plane.registry.catalog without causing a circular
import through model_plane.routing.

Resolution order at routing time
---------------------------------
Credentials are checked against os.environ / settings.  When a user saves
credentials via the UI, they are written into os.environ immediately (see
settings_router._apply_creds_to_process), so the routing layer sees them
through the environment — no per-user lookup is needed here.

The UI credential status (GET /settings/providers) is always user-scoped via
the session token and reads directly from SQLite in settings_router.py.
"""

from __future__ import annotations

import os

from model_plane.config import settings


def provider_has_creds(provider: str) -> bool:
    """Return True when the required credentials for *provider* are available
    in the current process environment.

    Checks os.environ / settings only — never makes a live network call.
    ``local`` and ``vllm`` providers are always considered credentialed because
    they typically run without an API key.
    """
    if provider in ("local", "vllm"):
        # Treat local as credentialed only when the base URL is actually set.
        # This respects the user clearing Local/vLLM credentials from the UI —
        # provider_creds_clear pops LOCAL_VLLM_API_BASE from os.environ, so this
        # returns False immediately after a clear and False on restart (tombstone
        # prevents re-import from .env in bootstrap_credentials_from_db).
        return bool(os.environ.get("LOCAL_VLLM_API_BASE"))

    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY") or settings.openai_api_key)

    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key)

    if provider == "watsonx":
        return bool(os.environ.get("WATSONX_APIKEY") or settings.watsonx_api_key)

    if provider == "bedrock":
        return bool(
            os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or getattr(settings, "aws_bearer_token_bedrock", None)
            or (
                getattr(settings, "aws_access_key_id", None)
                and getattr(settings, "aws_secret_access_key", None)
            )
        )

    if provider == "azure":
        return bool(os.environ.get("AZURE_API_KEY") or getattr(settings, "azure_api_key", None))

    if provider == "vertex_ai":
        return bool(os.environ.get("VERTEXAI_PROJECT") or getattr(settings, "vertex_project", None))

    if provider == "cohere":
        return bool(os.environ.get("COHERE_API_KEY") or getattr(settings, "cohere_api_key", None))

    if provider == "mistral":
        return bool(os.environ.get("MISTRAL_API_KEY") or getattr(settings, "mistral_api_key", None))

    # Unknown provider — assume uncredentialed to avoid routing to dead backends.
    return False
