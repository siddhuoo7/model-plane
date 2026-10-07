"""Mistral provider adapter.

LiteLLM handles Mistral natively via the MISTRAL_API_KEY env var.
This adapter also calls GET /v1/models to validate credentials.
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig


class MistralAdapter:
    provider_name = "mistral"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:  # noqa: ARG002
        """LiteLLM reads MISTRAL_API_KEY from env automatically."""
        return {}

    def list_models(self) -> list[str]:
        """GET https://api.mistral.ai/v1/models to list available model IDs."""
        import httpx

        api_key = os.environ.get("MISTRAL_API_KEY") or settings.mistral_api_key
        if not api_key:
            return []

        try:
            resp = httpx.get(
                "https://api.mistral.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except Exception:
            return []

    def health_probe(self, dep: DeploymentConfig) -> bool:  # noqa: ARG002
        return len(self.list_models()) > 0
