"""Unit tests for Sub-Task 3.2 — Catalog and Routing Config APIs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_plane.adapters.admin_api_router import router
from model_plane.registry.catalog import DeploymentConfig, ModelCatalog


# ── shared test app fixture ───────────────────────────────────────────────────

@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _make_catalog(*deps: DeploymentConfig) -> ModelCatalog:
    cat = ModelCatalog()
    for d in deps:
        cat.deployments[d.name] = d
        cat.tier_map.setdefault(d.tier, []).append(d.name)
    return cat


def _dep(name="dep-a", litellm_model="gpt-4o", provider="openai", tier="complex") -> DeploymentConfig:
    return DeploymentConfig(
        name=name, litellm_model=litellm_model, provider=provider, tier=tier,
        cost_per_1k_input=0.005, cost_per_1k_output=0.015, capabilities=["chat"],
    )


# ── ModelCatalog.to_yaml ──────────────────────────────────────────────────────

class TestCatalogToYaml:
    def test_round_trip(self, tmp_path):
        """to_yaml → from_yaml produces the same deployments."""
        cat = _make_catalog(
            _dep("dep-1", "gpt-4o", "openai", "complex"),
            _dep("dep-2", "gpt-4o-mini", "openai", "medium"),
        )
        path = tmp_path / "catalog.yaml"
        # Patch settings so prices enrichment is skipped on reload
        with patch("model_plane.registry.catalog.settings") as ms:
            ms.load_prices.return_value = {}
            ms.prices_yaml_path = ""
            cat.to_yaml(path)
            reloaded = ModelCatalog.from_yaml(path)

        assert set(reloaded.deployments.keys()) == {"dep-1", "dep-2"}
        assert reloaded.deployments["dep-1"].tier == "complex"
        assert reloaded.deployments["dep-2"].litellm_model == "gpt-4o-mini"

    def test_atomic_write(self, tmp_path):
        """No .tmp file remains after a successful write."""
        cat = _make_catalog(_dep())
        path = tmp_path / "catalog.yaml"
        with patch("model_plane.registry.catalog.settings") as ms:
            ms.load_prices.return_value = {}
            ms.prices_yaml_path = ""
            cat.to_yaml(path)
        assert path.exists()
        assert not (tmp_path / "catalog.yaml.tmp").exists()

    def test_optional_fields_included_when_set(self, tmp_path):
        dep = _dep()
        dep.api_base = "https://my.server/v1"
        dep.rpm = 200
        dep.extra = {"region": "us-east-1"}
        cat = _make_catalog(dep)
        path = tmp_path / "catalog.yaml"
        with patch("model_plane.registry.catalog.settings") as ms:
            ms.load_prices.return_value = {}
            ms.prices_yaml_path = ""
            cat.to_yaml(path)
        raw = yaml.safe_load(path.read_text())
        row = raw["deployments"][0]
        assert row["api_base"] == "https://my.server/v1"
        assert row["rpm"] == 200
        assert row["extra"] == {"region": "us-east-1"}


# ── GET /admin/api/catalog ────────────────────────────────────────────────────

class TestCatalogList:
    def test_returns_all_deployments(self, client):
        cat = _make_catalog(_dep("dep-1"), _dep("dep-2", tier="simple"))
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat), \
             patch("model_plane.adapters.admin_api_router._provider_has_creds", return_value=True):
            resp = client.get("/admin/api/catalog")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        names = {d["name"] for d in body["deployments"]}
        assert names == {"dep-1", "dep-2"}

    def test_includes_mtok_pricing(self, client):
        dep = _dep()
        dep.cost_per_1k_input = 0.005   # = $5/MTok
        cat = _make_catalog(dep)
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat), \
             patch("model_plane.adapters.admin_api_router._provider_has_creds", return_value=True):
            resp = client.get("/admin/api/catalog")
        body = resp.json()
        assert body["deployments"][0]["input_per_mtok_usd"] == pytest.approx(5.0)


# ── POST /admin/api/catalog ───────────────────────────────────────────────────

class TestCatalogCreate:
    def test_creates_deployment(self, client):
        cat = _make_catalog()  # empty catalog
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat), \
             patch.object(cat, "to_yaml"):
            with patch("model_plane.adapters.admin_api_router.settings") as ms:
                ms.admin_api_key = None
                ms.model_catalog_path = "config/models.yaml"
                resp = client.post("/admin/api/catalog", json={
                    "name": "new-dep",
                    "litellm_model": "gpt-4o",
                    "provider": "openai",
                    "tier": "complex",
                })
        assert resp.status_code == 201
        assert resp.json()["name"] == "new-dep"
        assert "new-dep" in cat.deployments

    def test_conflict_on_duplicate(self, client):
        cat = _make_catalog(_dep("existing-dep"))
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat):
            resp = client.post("/admin/api/catalog", json={
                "name": "existing-dep",
                "litellm_model": "gpt-4o",
                "provider": "openai",
            })
        assert resp.status_code == 409


# ── PUT /admin/api/catalog/{name} ─────────────────────────────────────────────

class TestCatalogUpdate:
    def test_updates_existing(self, client):
        cat = _make_catalog(_dep("dep-a", tier="simple"))
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat), \
             patch.object(cat, "to_yaml"):
            with patch("model_plane.adapters.admin_api_router.settings") as ms:
                ms.admin_api_key = None
                ms.model_catalog_path = "config/models.yaml"
                resp = client.put("/admin/api/catalog/dep-a", json={
                    "name": "dep-a",
                    "litellm_model": "gpt-4o-mini",
                    "provider": "openai",
                    "tier": "medium",
                })
        assert resp.status_code == 200
        assert resp.json()["tier"] == "medium"
        assert cat.deployments["dep-a"].tier == "medium"

    def test_404_for_missing(self, client):
        cat = _make_catalog()
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat):
            resp = client.put("/admin/api/catalog/ghost", json={
                "name": "ghost", "litellm_model": "x", "provider": "openai",
            })
        assert resp.status_code == 404


# ── DELETE /admin/api/catalog/{name} ─────────────────────────────────────────

class TestCatalogDelete:
    def test_removes_deployment(self, client):
        cat = _make_catalog(_dep("dep-a"))
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat), \
             patch.object(cat, "to_yaml"):
            with patch("model_plane.adapters.admin_api_router.settings") as ms:
                ms.admin_api_key = None
                ms.model_catalog_path = "config/models.yaml"
                resp = client.delete("/admin/api/catalog/dep-a")
        assert resp.status_code == 204
        assert "dep-a" not in cat.deployments

    def test_404_for_missing(self, client):
        cat = _make_catalog()
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat):
            resp = client.delete("/admin/api/catalog/ghost")
        assert resp.status_code == 404


# ── POST /admin/api/catalog/reload ───────────────────────────────────────────

class TestCatalogReload:
    def test_returns_deployment_count(self, client):
        cat = _make_catalog(_dep("dep-a"), _dep("dep-b"))
        with patch("model_plane.adapters.admin_api_router.reload_catalog", return_value=cat):
            resp = client.post("/admin/api/catalog/reload")
        assert resp.status_code == 200
        assert resp.json()["deployments"] == 2


# ── GET /admin/api/routing/config ─────────────────────────────────────────────

class TestRoutingConfigGet:
    def test_returns_config(self, client):
        cfg = {"scorer_weights": {"reasoning_markers": 0.18}, "task_overrides": {}}
        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.load_routing_config.return_value = cfg
            resp = client.get("/admin/api/routing/config")
        assert resp.status_code == 200
        assert resp.json()["config"]["scorer_weights"]["reasoning_markers"] == pytest.approx(0.18)


# ── PUT /admin/api/routing/config ─────────────────────────────────────────────

class TestRoutingConfigPut:
    def test_persists_config(self, client, tmp_path):
        cfg_path = tmp_path / "routing.yaml"
        cfg_path.write_text("scorer_weights: {}")
        new_cfg = {"scorer_weights": {"reasoning_markers": 0.25}, "task_overrides": {}}
        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.routing_config_path = cfg_path
            resp = client.put("/admin/api/routing/config", json=new_cfg)
        assert resp.status_code == 200
        written = yaml.safe_load(cfg_path.read_text())
        assert written["scorer_weights"]["reasoning_markers"] == pytest.approx(0.25)


# ── GET /admin/api/providers ─────────────────────────────────────────────────

class TestProvidersListEndpoint:
    def test_lists_providers(self, client):
        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.openai_api_key = "sk-test"
            ms.anthropic_api_key = None
            ms.watsonx_api_key = None
            ms.watsonx_url = None
            ms.aws_access_key_id = None
            ms.aws_region_name = None
            ms.azure_api_key = None
            ms.azure_api_base = None
            ms.cohere_api_key = None
            ms.mistral_api_key = None
            resp = client.get("/admin/api/providers")
        assert resp.status_code == 200
        providers = {p["provider"]: p for p in resp.json()["providers"]}
        assert providers["openai"]["configured"] is True
        assert providers["anthropic"]["configured"] is False


# ── POST /admin/api/providers/{name}/test ────────────────────────────────────

class TestProviderTest:
    def test_calls_health_probe(self, client):
        cat = _make_catalog(_dep("dep-a", provider="openai"))
        mock_adapter = MagicMock()
        # Endpoint calls list_models() (not health_probe); return a non-empty list
        mock_adapter.list_models.return_value = ["openai/gpt-4o", "openai/gpt-4o-mini"]
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat), \
             patch("model_plane.adapters.providers.get_adapter", return_value=mock_adapter), \
             patch("model_plane.adapters.admin_api_router._provider_has_creds", return_value=True):
            resp = client.post("/admin/api/providers/openai/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["healthy"] is True
        assert body["provider"] == "openai"

    def test_no_deployments_for_provider_returns_structured_error(self, client):
        """When no deployments exist for a provider, the endpoint returns 200 with
        healthy=False and an error message (not 404) so the UI gets a readable message."""
        cat = _make_catalog()
        mock_adapter = MagicMock()
        mock_adapter.list_models.return_value = []
        with patch("model_plane.adapters.admin_api_router.get_catalog", return_value=cat), \
             patch("model_plane.adapters.providers.get_adapter", return_value=mock_adapter), \
             patch("model_plane.adapters.admin_api_router._provider_has_creds", return_value=True):
            resp = client.post("/admin/api/providers/nonexistent/test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["healthy"] is False


# ── POST /admin/api/simulate ─────────────────────────────────────────────────

class TestSimulateEndpoint:
    def test_returns_pipeline_trace(self, client):
        """Let the real pipeline run (fast, no LLM call); check response shape."""
        with patch("model_plane.routing.pipeline.settings") as ms:
            ms.ml_routing_enabled = False
            ms.similarity_routing_enabled = False
            ms.load_routing_config.return_value = {}
            resp = client.post("/admin/api/simulate", json={
                "messages": [{"role": "user", "content": "write a python sort algorithm"}]
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["dry_run"] is True
        assert "pipeline_steps" in body
        assert "features" in body["pipeline_steps"]
        assert "classifier" in body["pipeline_steps"]
        assert "scorer" in body["pipeline_steps"]
        assert "winner" in body["pipeline_steps"]
        assert body["pipeline_steps"]["winner"]["stage"] is not None

    def test_rejects_missing_messages(self, client):
        resp = client.post("/admin/api/simulate", json={"model": "gpt-4o"})
        assert resp.status_code == 422
