"""Azure OpenAI provider adapter.

LiteLLM docs: https://docs.litellm.ai/docs/providers/azure
Required env vars: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig


class AzureOpenAIAdapter:
    provider_name = "azure"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:
        kwargs: dict = {}

        api_key = (
            settings.azure_api_key
            or dep.extra.get("api_key")
            or os.environ.get("AZURE_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        if api_key:
            kwargs["api_key"] = api_key

        api_base = (
            settings.azure_api_base
            or dep.extra.get("api_base")
            or os.environ.get("AZURE_API_BASE")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if api_base:
            kwargs["api_base"] = api_base

        api_version = (
            settings.azure_api_version
            or dep.extra.get("api_version")
            or os.environ.get("AZURE_API_VERSION")
            or "2024-02-01"
        )
        kwargs["api_version"] = api_version

        return kwargs

    def list_models(self) -> list[str]:
        """Call Azure OpenAI /openai/deployments to list deployed model IDs.

        Azure does not expose a simple /v1/models endpoint — it exposes
        /openai/deployments which lists the *deployment* names (not model IDs).
        We return those deployment names so the catalog cross-reference works.
        """
        import httpx

        api_key = settings.azure_api_key or os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        api_base = (
            settings.azure_api_base
            or os.environ.get("AZURE_API_BASE")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        api_version = settings.azure_api_version or os.environ.get("AZURE_API_VERSION") or "2024-02-01"

        if not api_key or not api_base:
            return []

        base = api_base.rstrip("/")
        url = f"{base}/openai/deployments?api-version={api_version}"
        try:
            resp = httpx.get(
                url,
                headers={"api-key": api_key},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return [d.get("id", "") for d in data.get("data", []) if d.get("id")]
        except Exception:
            return []

    def health_probe(self, dep: DeploymentConfig) -> bool:  # noqa: ARG002
        return len(self.list_models()) > 0
