"""Google Vertex AI provider adapter.

LiteLLM docs: https://docs.litellm.ai/docs/providers/vertex
Required env vars: VERTEXAI_PROJECT, VERTEXAI_LOCATION
Optional: GOOGLE_APPLICATION_CREDENTIALS (path to service-account JSON)
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig


class VertexAIAdapter:
    provider_name = "vertex_ai"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:
        kwargs: dict = {}

        project = (
            settings.vertex_project
            or dep.extra.get("vertex_project")
            or os.environ.get("VERTEXAI_PROJECT")
            or os.environ.get("VERTEX_PROJECT")
        )
        if project:
            kwargs["vertex_project"] = project

        location = (
            settings.vertex_location
            or dep.extra.get("vertex_location")
            or os.environ.get("VERTEXAI_LOCATION")
            or os.environ.get("VERTEX_LOCATION")
            or "us-central1"
        )
        kwargs["vertex_location"] = location

        return kwargs

    def list_models(self) -> list[str]:
        """Use google-cloud-aiplatform to list available Vertex models.

        Falls back to a curated static list when the SDK is unavailable or
        credentials are missing — Vertex IAM auth is complex and not always
        testable outside GCP.
        """
        project = (
            settings.vertex_project
            or os.environ.get("VERTEXAI_PROJECT")
            or os.environ.get("VERTEX_PROJECT")
        )
        if not project:
            return []

        location = (
            settings.vertex_location
            or os.environ.get("VERTEXAI_LOCATION")
            or os.environ.get("VERTEX_LOCATION")
            or "us-central1"
        )

        try:
            from google.cloud import aiplatform  # type: ignore[import]

            aiplatform.init(project=project, location=location)
            models = aiplatform.Model.list()
            return [m.resource_name.split("/")[-1] for m in models]
        except Exception:
            return []

    def health_probe(self, dep: DeploymentConfig) -> bool:  # noqa: ARG002
        # Vertex auth is complex; treat cred presence as "healthy"
        project = settings.vertex_project or os.environ.get("VERTEXAI_PROJECT") or os.environ.get("VERTEX_PROJECT")
        return bool(project)
