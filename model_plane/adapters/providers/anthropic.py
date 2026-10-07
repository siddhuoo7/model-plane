"""Anthropic provider adapter.

Anthropic exposes GET https://api.anthropic.com/v1/models to list
available model IDs. Requires anthropic-version header.
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig
from model_plane.adapters.providers.default import DefaultAdapter


class AnthropicAdapter(DefaultAdapter):
    provider_name = "anthropic"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:  # noqa: ARG002
        return {}

    def list_models(self) -> list[str]:
        """GET https://api.anthropic.com/v1/models and return model IDs.

        Raises RuntimeError on any failure so the test endpoint can surface
        the real error message instead of silently returning an empty list.
        """
        import httpx

        api_key = os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        try:
            resp = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Anthropic API returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Anthropic connection failed: {exc}") from exc

    def health_probe(self, dep: DeploymentConfig) -> bool:  # noqa: ARG002
        try:
            return len(self.list_models()) > 0
        except Exception:
            return False
