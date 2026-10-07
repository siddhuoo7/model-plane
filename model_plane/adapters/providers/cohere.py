"""Cohere provider adapter.

LiteLLM docs: https://docs.litellm.ai/docs/providers/cohere
Required env vars: COHERE_API_KEY
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig


class CohereAdapter:
    provider_name = "cohere"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:
        kwargs: dict = {}

        api_key = (
            settings.cohere_api_key
            or dep.extra.get("api_key")
            or os.environ.get("COHERE_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key

        return kwargs

    def list_models(self) -> list[str]:
        """GET https://api.cohere.com/v2/models to list available model IDs."""
        import httpx

        api_key = settings.cohere_api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            return []

        try:
            resp = httpx.get(
                "https://api.cohere.com/v2/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def health_probe(self, dep: DeploymentConfig) -> bool:  # noqa: ARG002
        return len(self.list_models()) > 0
