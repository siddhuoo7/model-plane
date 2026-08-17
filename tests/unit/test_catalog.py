"""Unit tests for model catalog."""
from model_plane.registry.catalog import ModelCatalog

SAMPLE_CATALOG = {
    "deployments": [
        {
            "name": "local-small",
            "litellm_model": "openai/gpt-3.5-turbo",
            "provider": "local",
            "context_limit": 8192,
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
            "tier": "small",
            "capabilities": ["text"],
            "fallback_to": ["local-medium"],
        },
        {
            "name": "local-medium",
            "litellm_model": "openai/mistral-7b",
            "provider": "local",
            "context_limit": 16384,
            "cost_per_1k_input": 0.0,
            "cost_per_1k_output": 0.0,
            "tier": "medium",
            "capabilities": ["text", "code"],
            "fallback_to": [],
        },
    ]
}


def test_catalog_loads_deployments():
    cat = ModelCatalog.from_yaml(SAMPLE_CATALOG)
    assert len(cat.deployments) == 2
    assert "local-small" in cat.deployments


def test_catalog_tier_map():
    cat = ModelCatalog.from_yaml(SAMPLE_CATALOG)
    small_deps = cat.by_tier("small")
    assert len(small_deps) == 1
    assert small_deps[0].name == "local-small"


def test_catalog_mark_unhealthy():
    cat = ModelCatalog.from_yaml(SAMPLE_CATALOG)
    cat.mark_unhealthy("local-small")
    healthy = cat.all_healthy()
    assert all(d.name != "local-small" for d in healthy)


def test_catalog_supports_capability():
    cat = ModelCatalog.from_yaml(SAMPLE_CATALOG)
    dep = cat.get("local-medium")
    assert dep.supports("code")
    assert not dep.supports("vision")


def test_empty_catalog():
    cat = ModelCatalog.from_yaml({"deployments": []})
    assert cat.all_healthy() == []
