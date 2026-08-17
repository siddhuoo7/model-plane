"""Model registry — loads and exposes the model catalog YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from model_plane.config import settings
from model_plane.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class DeploymentConfig:
    """A single LiteLLM-resolvable deployment."""

    name: str  # logical alias, e.g. "local-small"
    litellm_model: str  # e.g. "openai/gpt-3.5-turbo"
    provider: str  # openai | anthropic | watsonx | local
    api_base: str | None = None
    api_key_env: str | None = None
    context_limit: int = 8192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    max_parallel_requests: int = 50
    rpm: int | None = None
    tpm: int | None = None
    fallback_to: list[str] = field(default_factory=list)
    healthy: bool = True

    # derived from catalog
    tier: str = "medium"  # small | medium | large | reasoning

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass
class ModelCatalog:
    """In-memory model registry loaded from YAML."""

    deployments: dict[str, DeploymentConfig] = field(default_factory=dict)
    tier_map: dict[str, list[str]] = field(default_factory=dict)  # tier -> [deployment names]

    @classmethod
    def from_yaml(cls, path_or_dict: Any = None) -> "ModelCatalog":
        if path_or_dict is None:
            raw = settings.load_model_catalog()
        elif isinstance(path_or_dict, dict):
            raw = path_or_dict
        else:
            with open(path_or_dict) as fh:
                raw = yaml.safe_load(fh) or {}

        catalog = cls()
        for entry in raw.get("deployments", []):
            dep = DeploymentConfig(
                name=entry["name"],
                litellm_model=entry["litellm_model"],
                provider=entry.get("provider", "openai"),
                api_base=entry.get("api_base"),
                api_key_env=entry.get("api_key_env"),
                context_limit=entry.get("context_limit", 8192),
                cost_per_1k_input=entry.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=entry.get("cost_per_1k_output", 0.0),
                capabilities=entry.get("capabilities", []),
                tags=entry.get("tags", []),
                max_parallel_requests=entry.get("max_parallel_requests", 50),
                rpm=entry.get("rpm"),
                tpm=entry.get("tpm"),
                fallback_to=entry.get("fallback_to", []),
                tier=entry.get("tier", "medium"),
            )
            catalog.deployments[dep.name] = dep

        # build tier_map
        for name, dep in catalog.deployments.items():
            catalog.tier_map.setdefault(dep.tier, []).append(name)

        log.info("model_catalog_loaded", count=len(catalog.deployments))
        return catalog

    def get(self, name: str) -> DeploymentConfig | None:
        return self.deployments.get(name)

    def all_healthy(self) -> list[DeploymentConfig]:
        return [d for d in self.deployments.values() if d.healthy]

    def by_tier(self, tier: str) -> list[DeploymentConfig]:
        names = self.tier_map.get(tier, [])
        return [self.deployments[n] for n in names if n in self.deployments and self.deployments[n].healthy]

    def mark_unhealthy(self, name: str) -> None:
        if dep := self.deployments.get(name):
            dep.healthy = False
            log.warning("deployment_marked_unhealthy", deployment=name)

    def mark_healthy(self, name: str) -> None:
        if dep := self.deployments.get(name):
            dep.healthy = True


# Module-level singleton
_catalog: ModelCatalog | None = None


def get_catalog() -> ModelCatalog:
    global _catalog
    if _catalog is None:
        _catalog = ModelCatalog.from_yaml()
    return _catalog


def reload_catalog() -> ModelCatalog:
    global _catalog
    _catalog = ModelCatalog.from_yaml()
    return _catalog
