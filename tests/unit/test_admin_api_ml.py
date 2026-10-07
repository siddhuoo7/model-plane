"""Unit tests for Sub-Task 3.3 — ML Admin API, Tenant CRUD, Alert CRUD."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_plane.adapters.admin_api_router import (
    _JobStatus,
    _register_job,
    _alerts,
    _tenants,
    router,
)


@pytest.fixture(autouse=True)
def _clear_stores():
    """Reset in-memory stores between tests."""
    _tenants.clear()
    _alerts.clear()
    yield
    _tenants.clear()
    _alerts.clear()


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── _JobStatus ────────────────────────────────────────────────────────────────

class TestJobStatus:
    def test_emit_and_drain(self):
        job = _JobStatus("test-job")
        job.emit({"event": "start", "progress": 5})
        job.emit({"event": "progress", "progress": 50})
        events = job.drain()
        assert len(events) == 2
        assert events[0]["event"] == "start"
        # Drain clears the queue
        assert job.drain() == []

    def test_status_default_queued(self):
        job = _JobStatus("j1")
        assert job.status == "queued"
        assert job.progress == 0


# ── GET /admin/api/ml/status ──────────────────────────────────────────────────

class TestMlStatus:
    def test_returns_not_loaded_when_no_model_file(self, client):
        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.ml_model_path = "/nonexistent/path/model.joblib"
            resp = client.get("/admin/api/ml/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is False
        assert body["model_type"] is None

    def test_returns_metadata_when_model_exists(self, client, tmp_path):
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        y = np.array([0])
        clf.fit(X, y)

        model_file = tmp_path / "model.joblib"
        joblib.dump({"model": clf, "class_names": ["simple", "medium"]}, model_file)

        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.ml_model_path = str(model_file)
            resp = client.get("/admin/api/ml/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["model_type"] == "RandomForestClassifier"
        assert "simple" in body["class_names"]
        assert body["last_modified_utc"] is not None
        assert body["feature_count"] == 14


# ── POST /admin/api/ml/retrain ────────────────────────────────────────────────

class TestMlRetrain:
    def test_returns_job_id(self, client):
        resp = client.post("/admin/api/ml/retrain")
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["job_id"].startswith("retrain-")
        assert body["status"] == "queued"

    def test_job_registered_in_store(self, client):
        from model_plane.adapters.admin_api_router import _jobs
        resp = client.post("/admin/api/ml/retrain")
        job_id = resp.json()["job_id"]
        assert job_id in _jobs

    def test_invalid_model_type_rejected(self, client):
        resp = client.post("/admin/api/ml/retrain?model_type=invalid_type")
        assert resp.status_code == 422


# ── GET /admin/api/ml/retrain/{job_id}/stream ────────────────────────────────

class TestMlRetrainStream:
    def test_404_for_unknown_job(self, client):
        resp = client.get("/admin/api/ml/retrain/nonexistent-job/stream")
        assert resp.status_code == 404

    def test_streams_events_until_done(self, client):
        """A pre-loaded done job streams its events then closes."""
        import json as _json
        job = _register_job("test-done-job")
        job.emit({"event": "start", "progress": 5})
        job.emit({"event": "done", "progress": 100})
        job.status = "done"

        resp = client.get("/admin/api/ml/retrain/test-done-job/stream")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        parsed = [_json.loads(l[len("data: "):]) for l in data_lines]
        events = [e["event"] for e in parsed]
        assert "start" in events
        assert "done" in events


# ── POST /admin/api/ml/activate ───────────────────────────────────────────────

class TestMlActivate:
    def test_activates_existing_model(self, client, tmp_path):
        model_file = tmp_path / "new_model.joblib"
        model_file.write_bytes(b"dummy")  # just needs to exist

        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            # Patch object.__setattr__ side-effect indirectly by checking return value
            resp = client.post(f"/admin/api/ml/activate?model_path={model_file}")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert str(model_file) in resp.json()["active_model_path"]

    def test_404_for_missing_file(self, client):
        resp = client.post("/admin/api/ml/activate?model_path=/nonexistent/model.joblib")
        assert resp.status_code == 404


# ── Tenant policy CRUD ────────────────────────────────────────────────────────

class TestTenantCrud:
    def test_list_empty(self, client):
        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.load_routing_config.return_value = {}
            resp = client.get("/admin/api/tenants")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_create_and_list(self, client):
        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.load_routing_config.return_value = {}
            resp = client.post("/admin/api/tenants", json={
                "tenant_id": "tenant-1",
                "allowed_providers": ["watsonx"],
            })
            assert resp.status_code == 201
            assert resp.json()["tenant_id"] == "tenant-1"

            resp2 = client.get("/admin/api/tenants")
        assert resp2.json()["count"] == 1

    def test_create_conflict(self, client):
        _tenants["tenant-1"] = {"tenant_id": "tenant-1"}
        resp = client.post("/admin/api/tenants", json={"tenant_id": "tenant-1"})
        assert resp.status_code == 409

    def test_update(self, client):
        _tenants["tenant-1"] = {"tenant_id": "tenant-1", "allowed_providers": []}
        resp = client.put("/admin/api/tenants/tenant-1", json={
            "tenant_id": "tenant-1",
            "allowed_providers": ["openai"],
        })
        assert resp.status_code == 200
        assert "openai" in resp.json()["allowed_providers"]

    def test_update_404(self, client):
        resp = client.put("/admin/api/tenants/ghost", json={"tenant_id": "ghost"})
        assert resp.status_code == 404

    def test_delete(self, client):
        _tenants["tenant-x"] = {"tenant_id": "tenant-x"}
        resp = client.delete("/admin/api/tenants/tenant-x")
        assert resp.status_code == 204
        assert "tenant-x" not in _tenants

    def test_delete_404(self, client):
        resp = client.delete("/admin/api/tenants/ghost")
        assert resp.status_code == 404

    def test_list_merges_yaml_tenants(self, client):
        """Tenants from routing.yaml and in-memory store are merged."""
        _tenants["mem-tenant"] = {"tenant_id": "mem-tenant"}
        with patch("model_plane.adapters.admin_api_router.settings") as ms:
            ms.admin_api_key = None
            ms.load_routing_config.return_value = {
                "tenant_policies": [{"tenant_id": "yaml-tenant"}]
            }
            resp = client.get("/admin/api/tenants")
        ids = {t["tenant_id"] for t in resp.json()["tenants"]}
        assert "mem-tenant" in ids
        assert "yaml-tenant" in ids


# ── Alert CRUD ────────────────────────────────────────────────────────────────

class TestAlertCrud:
    def _alert_payload(self, alert_id: str = "alert-1") -> dict:
        return {
            "alert_id": alert_id,
            "name": "High error rate",
            "metric": "error_rate",
            "threshold": 0.05,
            "operator": "gt",
        }

    def test_list_empty(self, client):
        resp = client.get("/admin/api/alerts")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_create_and_list(self, client):
        resp = client.post("/admin/api/alerts", json=self._alert_payload())
        assert resp.status_code == 201
        assert resp.json()["alert_id"] == "alert-1"

        resp2 = client.get("/admin/api/alerts")
        assert resp2.json()["count"] == 1

    def test_create_conflict(self, client):
        _alerts["alert-1"] = self._alert_payload()
        resp = client.post("/admin/api/alerts", json=self._alert_payload())
        assert resp.status_code == 409

    def test_update(self, client):
        _alerts["alert-1"] = self._alert_payload()
        updated = {**self._alert_payload(), "threshold": 0.10}
        resp = client.put("/admin/api/alerts/alert-1", json=updated)
        assert resp.status_code == 200
        assert resp.json()["threshold"] == pytest.approx(0.10)

    def test_update_404(self, client):
        resp = client.put("/admin/api/alerts/ghost", json=self._alert_payload("ghost"))
        assert resp.status_code == 404

    def test_delete(self, client):
        _alerts["alert-1"] = self._alert_payload()
        resp = client.delete("/admin/api/alerts/alert-1")
        assert resp.status_code == 204
        assert "alert-1" not in _alerts

    def test_delete_404(self, client):
        resp = client.delete("/admin/api/alerts/ghost")
        assert resp.status_code == 404
