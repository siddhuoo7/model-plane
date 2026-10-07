"""Unit tests for Sub-Task 3.1 — Admin API Foundation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from model_plane.observability.cost_accumulator import CostAccumulator, CostSeries
from model_plane.observability.request_buffer import RequestRingBuffer


# ── RequestRingBuffer ──────────────────────────────────────────────────────────

class TestRequestRingBuffer:
    def test_push_and_len(self):
        buf = RequestRingBuffer(maxlen=10)
        buf.push({"id": "1"})
        buf.push({"id": "2"})
        assert len(buf) == 2

    def test_oldest_dropped_when_full(self):
        buf = RequestRingBuffer(maxlen=3)
        for i in range(5):
            buf.push({"id": str(i)})
        assert len(buf) == 3
        # Most recent 3 should be kept
        snap = buf.snapshot(limit=10)
        ids = [r["id"] for r in snap]
        assert "4" in ids
        assert "0" not in ids

    def test_snapshot_newest_first(self):
        buf = RequestRingBuffer(maxlen=10)
        buf.push({"id": "a", "ts": 1})
        buf.push({"id": "b", "ts": 2})
        buf.push({"id": "c", "ts": 3})
        snap = buf.snapshot(limit=10)
        assert snap[0]["id"] == "c"
        assert snap[-1]["id"] == "a"

    def test_snapshot_limit(self):
        buf = RequestRingBuffer(maxlen=100)
        for i in range(20):
            buf.push({"id": str(i)})
        snap = buf.snapshot(limit=5)
        assert len(snap) == 5

    def test_snapshot_filter_by_provider(self):
        buf = RequestRingBuffer(maxlen=100)
        buf.push({"id": "1", "provider": "openai", "tier": "complex"})
        buf.push({"id": "2", "provider": "watsonx", "tier": "medium"})
        buf.push({"id": "3", "provider": "openai", "tier": "simple"})
        snap = buf.snapshot(limit=100, provider="openai")
        assert all(r["provider"] == "openai" for r in snap)
        assert len(snap) == 2

    def test_snapshot_filter_by_tier(self):
        buf = RequestRingBuffer(maxlen=100)
        buf.push({"id": "1", "tier": "complex"})
        buf.push({"id": "2", "tier": "simple"})
        snap = buf.snapshot(limit=100, tier="complex")
        assert len(snap) == 1
        assert snap[0]["id"] == "1"

    def test_clear(self):
        buf = RequestRingBuffer(maxlen=10)
        buf.push({"id": "x"})
        buf.clear()
        assert len(buf) == 0


# ── CostSeries / CostAccumulator ───────────────────────────────────────────────

class TestCostSeries:
    def test_push_accumulates(self):
        s = CostSeries()
        s.push(0.001, 100, 50)
        s.push(0.002, 200, 100)
        w = s.window_sum(hours=24)
        assert w["cost_usd"] == pytest.approx(0.003)
        assert w["requests"] == 2
        assert w["input_tokens"] == 300
        assert w["output_tokens"] == 150

    def test_window_sum_zero_when_empty(self):
        s = CostSeries()
        w = s.window_sum(hours=24)
        assert w["cost_usd"] == 0.0
        assert w["requests"] == 0


class TestCostAccumulator:
    def test_record_and_summary(self):
        acc = CostAccumulator()
        acc.record(
            provider="openai", tier="complex", task_type="code_generation",
            tenant_id="", cost_usd=0.005, input_tokens=500, output_tokens=200,
        )
        acc.record(
            provider="watsonx", tier="simple", task_type="simple_qa",
            tenant_id="", cost_usd=0.001, input_tokens=100, output_tokens=50,
        )
        rows = acc.summary(hours=24)
        assert len(rows) == 2
        total = acc.total_cost(hours=24)
        assert total == pytest.approx(0.006)

    def test_summary_excludes_zero_request_keys(self):
        acc = CostAccumulator()
        acc.record(
            provider="openai", tier="complex", task_type="code_generation",
            tenant_id="", cost_usd=0.0, input_tokens=0, output_tokens=0,
        )
        # A zero-cost record still has 1 request — it should appear
        rows = acc.summary(hours=24)
        assert any(r["requests"] == 1 for r in rows)

    def test_summary_sorted_by_cost_descending(self):
        acc = CostAccumulator()
        acc.record(
            provider="a", tier="simple", task_type="x",
            tenant_id="", cost_usd=0.001, input_tokens=0, output_tokens=0,
        )
        acc.record(
            provider="b", tier="complex", task_type="y",
            tenant_id="", cost_usd=0.010, input_tokens=0, output_tokens=0,
        )
        rows = acc.summary(hours=24)
        assert rows[0]["cost_usd"] >= rows[-1]["cost_usd"]


# ── Admin API router HTTP tests ───────────────────────────────────────────────

@pytest.fixture()
def admin_client():
    """TestClient for admin_api_router with auth disabled."""
    from fastapi import FastAPI
    from model_plane.adapters.admin_api_router import router

    app = FastAPI()
    app.include_router(router)
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


class TestAdminHealthEndpoint:
    def test_returns_200(self, admin_client):
        resp = admin_client.get("/admin/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "healthy_deployments" in body
        assert "buffer_size" in body

    def test_requires_auth_when_key_set(self):
        from fastapi import FastAPI
        from model_plane.adapters.admin_api_router import router

        app = FastAPI()
        app.include_router(router)
        with patch("model_plane.adapters.admin_api_router.settings") as mock_s:
            mock_s.admin_api_key = "secret-key"
            mock_s.routing_mode = "custom_plugin"
            mock_s.ml_routing_enabled = False
            mock_s.admin_buffer_size = 10000
            with TestClient(app) as client:
                # No token
                resp = client.get("/admin/api/health")
                assert resp.status_code == 401
                # Wrong token
                resp = client.get(
                    "/admin/api/health",
                    headers={"Authorization": "Bearer wrong-key"},
                )
                assert resp.status_code == 401
                # Correct token
                resp = client.get(
                    "/admin/api/health",
                    headers={"Authorization": "Bearer secret-key"},
                )
                assert resp.status_code == 200


class TestAdminTrafficEndpoint:
    def test_returns_200_with_empty_data(self, admin_client):
        from model_plane.observability.cost_accumulator import _acc_lock
        import model_plane.observability.cost_accumulator as acc_mod
        # reset accumulator to empty state for isolation
        original = acc_mod._accumulator
        acc_mod._accumulator = CostAccumulator()
        try:
            resp = admin_client.get("/admin/api/traffic")
            assert resp.status_code == 200
            body = resp.json()
            assert "total_requests" in body
            assert "by_provider" in body
            assert "by_tier" in body
        finally:
            acc_mod._accumulator = original

    def test_window_hours_param(self, admin_client):
        resp = admin_client.get("/admin/api/traffic?window_hours=48")
        assert resp.status_code == 200
        assert resp.json()["window_hours"] == 48

    def test_invalid_window_hours_rejected(self, admin_client):
        resp = admin_client.get("/admin/api/traffic?window_hours=0")
        assert resp.status_code == 422


class TestAdminRequestsEndpoint:
    def test_returns_records(self, admin_client):
        import model_plane.observability.request_buffer as buf_mod
        original = buf_mod._buffer
        buf = RequestRingBuffer(maxlen=100)
        buf.push({"request_id": "r1", "provider": "openai", "tier": "complex",
                  "task_type": "code_generation", "tenant_id": ""})
        buf_mod._buffer = buf
        try:
            resp = admin_client.get("/admin/api/requests")
            assert resp.status_code == 200
            body = resp.json()
            assert body["count"] == 1
            assert body["records"][0]["request_id"] == "r1"
        finally:
            buf_mod._buffer = original

    def test_filter_by_provider(self, admin_client):
        import model_plane.observability.request_buffer as buf_mod
        original = buf_mod._buffer
        buf = RequestRingBuffer(maxlen=100)
        buf.push({"request_id": "r1", "provider": "openai", "tier": "complex",
                  "task_type": "x", "tenant_id": ""})
        buf.push({"request_id": "r2", "provider": "watsonx", "tier": "simple",
                  "task_type": "y", "tenant_id": ""})
        buf_mod._buffer = buf
        try:
            resp = admin_client.get("/admin/api/requests?provider=openai")
            assert resp.status_code == 200
            body = resp.json()
            assert body["count"] == 1
            assert body["records"][0]["provider"] == "openai"
        finally:
            buf_mod._buffer = original


class TestAdminCostEndpoint:
    def test_returns_200(self, admin_client):
        resp = admin_client.get("/admin/api/cost")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_cost_usd" in body
        assert "rows" in body
        assert "by_task_type" in body


class TestProviderAvailableEndpoint:
    def test_returns_models_for_openai(self, admin_client):
        with patch(
            "model_plane.adapters.admin_api_router.load_prices_for_provider",
            return_value=[{"id": "gpt-4o", "input": 2.5, "output": 10.0}],
        ):
            resp = admin_client.get("/admin/api/providers/openai/available")
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "openai"
        assert body["count"] == 1
        assert body["models"][0]["id"] == "gpt-4o"

    def test_returns_empty_for_unknown_provider(self, admin_client):
        with patch(
            "model_plane.adapters.admin_api_router.load_prices_for_provider",
            return_value=[],
        ):
            resp = admin_client.get("/admin/api/providers/unknown_xyz/available")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0
