"""Model registry — loads and exposes the model catalog YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from model_plane.config import settings
from model_plane.logging_setup import get_logger
from model_plane.provider_creds import provider_has_creds

log = get_logger(__name__)


@dataclass
class DeploymentConfig:
    """A single LiteLLM-resolvable deployment."""

    name: str  # logical alias, e.g. "local-small"
    litellm_model: str  # e.g. "openai/gpt-3.5-turbo"
    provider: str  # openai | anthropic | watsonx | local
    api_base: str | None = None
    api_base_env: str | None = None
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

    # provider-specific extra params (e.g. aws_region_name, vertex_project, api_version)
    extra: dict = field(default_factory=dict)

    # cache economics — used by Phase 4 cost optimiser
    cache_read_per_1k: float = 0.0
    cache_write_per_1k: float = 0.0

    # security routing — provider trust override (empty = use default map)
    # Valid values: external_public | approved_external | private_cloud | on_prem
    provider_trust: str = ""

    # cache-capability abstraction — none | response | prefix | kv_events
    cache_capability: str = "none"

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    @property
    def input_per_mtok_usd(self) -> float:
        """Input cost in USD per 1M tokens (= cost_per_1k_input * 1000)."""
        return round(self.cost_per_1k_input * 1000, 6)

    @property
    def output_per_mtok_usd(self) -> float:
        """Output cost in USD per 1M tokens (= cost_per_1k_output * 1000)."""
        return round(self.cost_per_1k_output * 1000, 6)


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
            api_base = entry.get("api_base")
            api_base_env = entry.get("api_base_env")
            if api_base_env:
                api_base = settings.__dict__.get(api_base_env.lower()) or settings.model_config.get("env_file") and None
                api_base = api_base or __import__("os").environ.get(api_base_env) or entry.get("api_base")

            dep = DeploymentConfig(
                name=entry["name"],
                litellm_model=entry["litellm_model"],
                provider=entry.get("provider", "openai"),
                api_base=api_base,
                api_base_env=api_base_env,
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
                extra=entry.get("extra", {}),
                cache_read_per_1k=entry.get("cache_read_per_1k", 0.0),
                cache_write_per_1k=entry.get("cache_write_per_1k", 0.0),
                provider_trust=entry.get("provider_trust", ""),
                cache_capability=entry.get("cache_capability", "none"),
            )
            catalog.deployments[dep.name] = dep

        # build tier_map
        for name, dep in catalog.deployments.items():
            catalog.tier_map.setdefault(dep.tier, []).append(name)

        # prices.yaml enrichment is disabled — models.yaml is now the single
        # source of truth and already contains all pricing, capabilities, and
        # context limits inline on each deployment entry.

        log.info("model_catalog_loaded", count=len(catalog.deployments))
        return catalog

    def get(self, name: str) -> DeploymentConfig | None:
        return self.deployments.get(name)

    def all_healthy(self) -> list[DeploymentConfig]:
        """Return deployments that are both flagged healthy AND have provider credentials."""
        return [
            d for d in self.deployments.values()
            if d.healthy and provider_has_creds(d.provider)
        ]

    def by_tier(self, tier: str) -> list[DeploymentConfig]:
        names = self.tier_map.get(tier, [])
        return [
            self.deployments[n] for n in names
            if n in self.deployments
            and self.deployments[n].healthy
            and provider_has_creds(self.deployments[n].provider)
        ]

    def mark_unhealthy(self, name: str) -> None:
        if dep := self.deployments.get(name):
            dep.healthy = False
            log.warning("deployment_marked_unhealthy", deployment=name)

    def mark_healthy(self, name: str) -> None:
        if dep := self.deployments.get(name):
            dep.healthy = True

    def apply_user_overrides(self, overrides: dict[str, dict]) -> "ModelCatalog":
        """Return a new ModelCatalog with user overrides applied on top of self.

        self (the global YAML-loaded catalog) is never mutated.

        Override patch keys:
          deleted: true           — exclude this deployment from the returned catalog
          disabled: true          — set dep.healthy = False in the returned catalog
          fields: {k: v, …}      — overwrite any DeploymentConfig fields
          added: {<full dep>}     — add a new deployment that doesn't exist in YAML
        """
        import copy
        result = ModelCatalog()

        # Start from YAML base (deep-copy so mutations don't affect the singleton)
        for name, dep in self.deployments.items():
            patch = overrides.get(name, {})
            if patch.get("deleted"):
                continue  # user deleted this deployment
            new_dep = copy.copy(dep)
            if patch.get("disabled"):
                new_dep.healthy = False
            # Apply field-level overrides
            for field_name, val in patch.get("fields", {}).items():
                if hasattr(new_dep, field_name):
                    setattr(new_dep, field_name, val)
            result.deployments[name] = new_dep

        # Add user-created deployments (not in YAML)
        for name, patch in overrides.items():
            if "added" in patch and name not in result.deployments:
                added = patch["added"]
                try:
                    dep = DeploymentConfig(
                        name=name,
                        litellm_model=added.get("litellm_model", ""),
                        provider=added.get("provider", "openai"),
                        tier=added.get("tier", "medium"),
                        context_limit=added.get("context_limit", 8192),
                        cost_per_1k_input=added.get("cost_per_1k_input", 0.0),
                        cost_per_1k_output=added.get("cost_per_1k_output", 0.0),
                        capabilities=list(added.get("capabilities", [])),
                        tags=list(added.get("tags", [])),
                        fallback_to=list(added.get("fallback_to", [])),
                        max_parallel_requests=added.get("max_parallel_requests", 50),
                        rpm=added.get("rpm"),
                        api_base=added.get("api_base"),
                        api_key_env=added.get("api_key_env"),
                        healthy=added.get("healthy", True),
                        provider_trust=added.get("provider_trust", ""),
                        cache_capability=added.get("cache_capability", "none"),
                    )
                    result.deployments[name] = dep
                except Exception:
                    pass  # malformed added entry — skip silently

        # Rebuild tier_map
        for n, d in result.deployments.items():
            result.tier_map.setdefault(d.tier, []).append(n)

        return result

    def to_yaml(self, path: str | Path) -> None:
        """Serialise in-memory catalog back to YAML at *path*.

        Only fields that models.yaml supports are written; computed properties
        (input_per_mtok_usd etc.) are excluded. The file is written atomically
        via a temp-rename so a crash mid-write never corrupts the config.
        """
        rows = []
        for dep in self.deployments.values():
            row: dict[str, Any] = {
                "name": dep.name,
                "litellm_model": dep.litellm_model,
                "provider": dep.provider,
                "tier": dep.tier,
                "context_limit": dep.context_limit,
                "cost_per_1k_input": dep.cost_per_1k_input,
                "cost_per_1k_output": dep.cost_per_1k_output,
                "capabilities": list(dep.capabilities),
                "tags": list(dep.tags),
                "max_parallel_requests": dep.max_parallel_requests,
                "fallback_to": list(dep.fallback_to),
                "healthy": dep.healthy,
            }
            if dep.api_base:
                row["api_base"] = dep.api_base
            if dep.api_base_env:
                row["api_base_env"] = dep.api_base_env
            if dep.api_key_env:
                row["api_key_env"] = dep.api_key_env
            if dep.rpm is not None:
                row["rpm"] = dep.rpm
            if dep.tpm is not None:
                row["tpm"] = dep.tpm
            if dep.extra:
                row["extra"] = dep.extra
            if dep.cache_read_per_1k:
                row["cache_read_per_1k"] = dep.cache_read_per_1k
            if dep.cache_write_per_1k:
                row["cache_write_per_1k"] = dep.cache_write_per_1k
            if dep.provider_trust:
                row["provider_trust"] = dep.provider_trust
            if dep.cache_capability and dep.cache_capability != "none":
                row["cache_capability"] = dep.cache_capability
            rows.append(row)

        dest = Path(path)
        tmp = dest.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.dump({"deployments": rows}, default_flow_style=False, sort_keys=False))
        tmp.replace(dest)
        log.info("catalog_saved", path=str(dest), count=len(rows))


# ── Price registry helpers (Sub-Task 3.0) ────────────────────────────────────

# Maps model_plane provider names to the keys used in prices.yaml
_PROVIDER_PRICE_KEY: dict[str, str] = {
    "openai":     "openai",
    "anthropic":  "anthropic",
    "watsonx":    "watsonx",
    "bedrock":    "bedrock",      # dedicated bedrock section in prices.yaml
    "azure":      "openai",       # azure models share openai ids after prefix strip
    "litellm":    "litellm",
    "vllm":       "vllm",
    "local":      "vllm",         # local vLLM hosts use the vllm section
    "vertex_ai":  "openai",       # vertex mirrors openai ids in some cases
    "cohere":     "litellm",
    "mistral":    "litellm",
}


def _bare_model_id(litellm_model: str) -> str:
    """Strip the provider prefix from a litellm model string.

    Examples:
        "watsonx/ibm/granite-4-h-small"  → "ibm/granite-4-h-small"
        "anthropic/claude-3-haiku-20240307" → "claude-3-haiku-20240307"
        "hosted_vllm/qwen3.8-27b"        → "qwen3.8-27b"
        "gpt-4o"                          → "gpt-4o"  (no prefix)
    """
    # Remove known litellm transport prefixes (hosted_vllm, bedrock, azure, vertex_ai…)
    for prefix in ("hosted_vllm/", "bedrock/", "azure/", "vertex_ai/",
                   "cohere/", "mistral/", "anthropic/", "openai/"):
        if litellm_model.startswith(prefix):
            return litellm_model[len(prefix):]
    # For watsonx/<org/model> remove only the first "watsonx/" segment
    if litellm_model.startswith("watsonx/"):
        return litellm_model[len("watsonx/"):]
    return litellm_model


def _find_price_entry(bare_id: str, entries: list[dict]) -> dict | None:
    """Find the best (longest-id-prefix) matching price entry for *bare_id*.

    Strata's convention: 'id' is a prefix — the entry whose id is the longest
    prefix of (or equal to) bare_id wins.
    """
    best: dict | None = None
    best_len = -1
    for entry in entries:
        entry_id: str = entry.get("id", "")
        if bare_id == entry_id or bare_id.startswith(entry_id):
            if len(entry_id) > best_len:
                best = entry
                best_len = len(entry_id)
    return best


def _enrich_from_prices(dep: DeploymentConfig, prices: dict) -> bool:
    """Overwrite cost fields on *dep* from *prices* if a match is found.

    Returns True when enrichment occurred, False otherwise.
    Prices.yaml stores USD/MTok; DeploymentConfig uses USD/1k — divide by 1000.
    """
    section_key = _PROVIDER_PRICE_KEY.get(dep.provider)
    if not section_key:
        return False

    entries: list[dict] = prices.get(section_key, [])
    if not entries:
        return False

    bare_id = _bare_model_id(dep.litellm_model)
    entry = _find_price_entry(bare_id, entries)
    if not entry:
        return False

    # Overwrite cost fields (prices.yaml is USD/MTok; catalog uses USD/1k)
    dep.cost_per_1k_input = entry["input"] / 1000.0
    dep.cost_per_1k_output = entry["output"] / 1000.0

    # Enrich capabilities and context_limit only if not explicitly set in models.yaml
    # (non-default values in models.yaml take precedence)
    if not dep.capabilities and entry.get("capabilities"):
        dep.capabilities = list(entry["capabilities"])
    if dep.context_limit == 8192 and entry.get("contextLimit"):
        dep.context_limit = entry["contextLimit"]

    # Enrich cache costs if present in prices.yaml
    if "cacheRead" in entry:
        dep.cache_read_per_1k = entry["cacheRead"] / 1000.0
    if "cacheWrite" in entry:
        dep.cache_write_per_1k = entry["cacheWrite"] / 1000.0

    return True


def load_prices_for_provider(provider: str) -> list[dict]:
    """Return all price entries for *provider* from prices.yaml.

    Used by GET /admin/api/providers/{provider}/available (Sub-Task 3.2).
    Returns entries with their original USD/MTok values (not divided by 1000).
    """
    prices = settings.load_prices()
    section_key = _PROVIDER_PRICE_KEY.get(provider, provider)
    return prices.get(section_key, [])


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
