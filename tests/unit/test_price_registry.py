"""Unit tests for Sub-Task 3.0 — Price Registry Integration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from model_plane.registry.catalog import (
    DeploymentConfig,
    ModelCatalog,
    _bare_model_id,
    _enrich_from_prices,
    _find_price_entry,
    load_prices_for_provider,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _dep(
    name: str = "test-dep",
    litellm_model: str = "gpt-4o",
    provider: str = "openai",
    cost_per_1k_input: float = 0.001,
    cost_per_1k_output: float = 0.003,
    capabilities: list | None = None,
    context_limit: int = 8192,
) -> DeploymentConfig:
    return DeploymentConfig(
        name=name,
        litellm_model=litellm_model,
        provider=provider,
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_output,
        capabilities=capabilities or [],
        context_limit=context_limit,
    )


_SAMPLE_PRICES = {
    "openai": [
        {
            "id": "gpt-4o",
            "input": 2.5,
            "output": 10.0,
            "cacheRead": 1.25,
            "tier": "complex",
            "capabilities": ["chat", "function-calling", "json-mode", "reasoning"],
            "contextLimit": 128000,
        },
        {
            "id": "gpt-4o-mini",
            "input": 0.15,
            "output": 0.6,
            "tier": "simple",
            "capabilities": ["chat", "function-calling"],
            "contextLimit": 128000,
        },
    ],
    "anthropic": [
        {
            "id": "claude-3-haiku",
            "input": 0.8,
            "output": 4.0,
            "cacheRead": 0.08,
            "cacheWrite": 1.0,
            "tier": "simple",
            "capabilities": ["chat", "function-calling"],
            "contextLimit": 200000,
        },
    ],
    "watsonx": [
        {
            "id": "ibm/granite-4-h-small",
            "input": 0.0636,
            "output": 0.265,
            "tier": "simple",
            "capabilities": ["chat", "function-calling", "json-mode"],
            "contextLimit": 8192,
        },
    ],
    "vllm": [
        {
            "id": "vllm/",
            "input": 0,
            "output": 0,
        },
        {
            "id": "qwen3.8-27b",
            "input": 0,
            "output": 0,
            "tier": ["simple", "medium", "complex"],
            "capabilities": ["chat", "function-calling", "json-mode", "reasoning", "code"],
            "contextLimit": 32768,
        },
    ],
    "litellm": [],
}


# ── _bare_model_id ─────────────────────────────────────────────────────────────

class TestBareModelId:
    def test_strips_watsonx_prefix(self):
        assert _bare_model_id("watsonx/ibm/granite-4-h-small") == "ibm/granite-4-h-small"

    def test_strips_hosted_vllm_prefix(self):
        assert _bare_model_id("hosted_vllm/qwen3.8-27b") == "qwen3.8-27b"

    def test_strips_anthropic_prefix(self):
        assert _bare_model_id("anthropic/claude-3-haiku-20240307") == "claude-3-haiku-20240307"

    def test_no_prefix(self):
        assert _bare_model_id("gpt-4o") == "gpt-4o"

    def test_strips_bedrock_prefix(self):
        assert _bare_model_id("bedrock/us.anthropic.claude-3-5-haiku") == "us.anthropic.claude-3-5-haiku"

    def test_strips_openai_prefix(self):
        assert _bare_model_id("openai/gpt-4o") == "gpt-4o"


# ── _find_price_entry ──────────────────────────────────────────────────────────

class TestFindPriceEntry:
    def test_exact_match(self):
        entries = [{"id": "gpt-4o", "input": 2.5, "output": 10.0}]
        result = _find_price_entry("gpt-4o", entries)
        assert result is not None
        assert result["input"] == 2.5

    def test_prefix_match(self):
        # "gpt-4o" is a prefix of "gpt-4o-mini"
        entries = [{"id": "gpt-4o", "input": 2.5, "output": 10.0}]
        result = _find_price_entry("gpt-4o-2024-11-20", entries)
        assert result is not None

    def test_longest_prefix_wins(self):
        entries = [
            {"id": "gpt-4o", "input": 1.0, "output": 1.0},
            {"id": "gpt-4o-mini", "input": 0.15, "output": 0.6},
        ]
        result = _find_price_entry("gpt-4o-mini", entries)
        assert result["input"] == 0.15  # longer match wins

    def test_no_match(self):
        entries = [{"id": "gpt-4o", "input": 2.5, "output": 10.0}]
        assert _find_price_entry("claude-3-haiku", entries) is None


# ── _enrich_from_prices ────────────────────────────────────────────────────────

class TestEnrichFromPrices:
    def test_match_overwrites_costs(self):
        """When a price entry is found, cost_per_1k fields are overwritten."""
        dep = _dep(litellm_model="gpt-4o", provider="openai",
                   cost_per_1k_input=0.001, cost_per_1k_output=0.003)
        result = _enrich_from_prices(dep, _SAMPLE_PRICES)

        assert result is True
        # prices.yaml: input=2.5 USD/MTok → 0.0025 USD/1k
        assert dep.cost_per_1k_input == pytest.approx(0.0025)
        assert dep.cost_per_1k_output == pytest.approx(0.01)

    def test_no_match_keeps_original_costs(self):
        """When no price entry matches, the original costs are preserved."""
        dep = _dep(litellm_model="my-private-model", provider="openai",
                   cost_per_1k_input=0.001, cost_per_1k_output=0.003)
        result = _enrich_from_prices(dep, _SAMPLE_PRICES)

        assert result is False
        assert dep.cost_per_1k_input == pytest.approx(0.001)
        assert dep.cost_per_1k_output == pytest.approx(0.003)

    def test_watsonx_model_matched(self):
        dep = _dep(
            litellm_model="watsonx/ibm/granite-4-h-small",
            provider="watsonx",
            cost_per_1k_input=0.0,
        )
        result = _enrich_from_prices(dep, _SAMPLE_PRICES)
        assert result is True
        assert dep.cost_per_1k_input == pytest.approx(0.0636 / 1000)

    def test_capabilities_enriched_when_empty(self):
        dep = _dep(litellm_model="gpt-4o", provider="openai", capabilities=[])
        _enrich_from_prices(dep, _SAMPLE_PRICES)
        assert "chat" in dep.capabilities
        assert "reasoning" in dep.capabilities

    def test_capabilities_not_overwritten_when_set(self):
        dep = _dep(litellm_model="gpt-4o", provider="openai",
                   capabilities=["text", "vision"])
        _enrich_from_prices(dep, _SAMPLE_PRICES)
        # original capabilities preserved — prices.yaml doesn't override
        assert dep.capabilities == ["text", "vision"]

    def test_context_limit_enriched_when_default(self):
        dep = _dep(litellm_model="gpt-4o", provider="openai", context_limit=8192)
        _enrich_from_prices(dep, _SAMPLE_PRICES)
        assert dep.context_limit == 128000

    def test_context_limit_not_overwritten_when_custom(self):
        dep = _dep(litellm_model="gpt-4o", provider="openai", context_limit=200000)
        _enrich_from_prices(dep, _SAMPLE_PRICES)
        assert dep.context_limit == 200000

    def test_cache_costs_enriched(self):
        dep = _dep(litellm_model="gpt-4o", provider="openai")
        _enrich_from_prices(dep, _SAMPLE_PRICES)
        assert dep.cache_read_per_1k == pytest.approx(1.25 / 1000)

    def test_unknown_provider_returns_false(self):
        dep = _dep(provider="unknown_provider")
        result = _enrich_from_prices(dep, _SAMPLE_PRICES)
        assert result is False

    def test_empty_prices_returns_false(self):
        dep = _dep(litellm_model="gpt-4o", provider="openai")
        result = _enrich_from_prices(dep, {})
        assert result is False

    def test_vllm_local_provider_matched(self):
        dep = _dep(
            litellm_model="hosted_vllm/qwen3.8-27b",
            provider="local",
            cost_per_1k_input=0.0,
        )
        result = _enrich_from_prices(dep, _SAMPLE_PRICES)
        assert result is True
        assert dep.cost_per_1k_input == 0.0


# ── computed properties ────────────────────────────────────────────────────────

class TestComputedPriceProperties:
    def test_input_per_mtok_usd(self):
        dep = _dep(cost_per_1k_input=0.0025)
        assert dep.input_per_mtok_usd == pytest.approx(2.5)

    def test_output_per_mtok_usd(self):
        dep = _dep(cost_per_1k_output=0.01)
        assert dep.output_per_mtok_usd == pytest.approx(10.0)

    def test_zero_cost(self):
        dep = _dep(cost_per_1k_input=0.0, cost_per_1k_output=0.0)
        assert dep.input_per_mtok_usd == 0.0
        assert dep.output_per_mtok_usd == 0.0


# ── missing prices.yaml ────────────────────────────────────────────────────────

class TestMissingPricesFile:
    def test_catalog_loads_without_prices_yaml(self):
        """If prices.yaml is absent, catalog loads normally using models.yaml values."""
        raw = {
            "deployments": [
                {
                    "name": "test-dep",
                    "litellm_model": "gpt-4o",
                    "provider": "openai",
                    "cost_per_1k_input": 0.005,
                    "cost_per_1k_output": 0.015,
                    "tier": "complex",
                }
            ]
        }
        with patch("model_plane.registry.catalog.settings") as mock_s:
            mock_s.load_model_catalog.return_value = raw
            mock_s.load_prices.return_value = {}  # no prices.yaml
            mock_s.prices_yaml_path = ""          # disabled
            catalog = ModelCatalog.from_yaml(raw)

        dep = catalog.get("test-dep")
        assert dep is not None
        # original costs from models.yaml kept intact
        assert dep.cost_per_1k_input == pytest.approx(0.005)
        assert dep.cost_per_1k_output == pytest.approx(0.015)


# ── load_prices_for_provider ───────────────────────────────────────────────────

class TestLoadPricesForProvider:
    def test_returns_entries_for_known_provider(self):
        with patch("model_plane.registry.catalog.settings") as mock_s:
            mock_s.load_prices.return_value = _SAMPLE_PRICES
            mock_s.prices_yaml_path = "config/prices.yaml"
            entries = load_prices_for_provider("openai")

        assert len(entries) == 2
        assert entries[0]["id"] == "gpt-4o"

    def test_returns_empty_for_unknown_provider(self):
        with patch("model_plane.registry.catalog.settings") as mock_s:
            mock_s.load_prices.return_value = _SAMPLE_PRICES
            mock_s.prices_yaml_path = "config/prices.yaml"
            entries = load_prices_for_provider("unknown_xyz")

        assert entries == []
