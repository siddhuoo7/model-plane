"""Default (no-op) provider adapter.

Used for openai and anthropic providers where LiteLLM handles all auth
natively. ``build_kwargs`` intentionally returns an empty dict.
"""

from __future__ import annotations

import os

from model_plane.registry.catalog import DeploymentConfig


class DefaultAdapter:
    provider_name = "default"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:  # noqa: ARG002
        """LiteLLM handles these providers natively — no extra kwargs needed."""
        return {}

    def list_models(self) -> list[str]:  # noqa: ARG002
        """Subclasses override. Default returns empty list."""
        return []

    def health_probe(self, dep: DeploymentConfig) -> bool:
        """Probe by attempting to list models; fall back to optimistic True."""
        try:
            models = self.list_models()
            return len(models) > 0
        except Exception:
            return True
