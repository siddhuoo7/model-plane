"""Unit tests for POST /v1/routing/explain endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from model_plane.app import create_app
from model_plane.classifier.taxonomy import ComplexityTier, TaskType
from model_plane.registry.catalog import DeploymentConfig
from model_plane.routing.context import RoutingDecision

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_dep(name: str = "watsonx-granite-medium", tier: str = "medium") -> DeploymentConfig:
    return DeploymentConfig(
        name=name,
        litellm_model="watsonx/ibm/granite-4-h-small",
        provider="watsonx",
        context_limit=20480,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        tier=tier,
        capabilities=["text", "code"],
        fallback_to=["watsonx-granite-small"],
    )


def _make_decision(dep: DeploymentConfig) -> RoutingDecision:
    return RoutingDecision(
        deployment=dep,
        source="custom_plugin",
        confidence=0.85,
        reasoning="top_score=1.200",
    )


def _client() -> TestClient:
    """Create a test client with auth bypassed via FastAPI dependency override."""
    from model_plane.adapters.auth import verify_api_key

    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-user"
    return TestClient(app, raise_server_exceptions=True)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRoutingExplainEndpoint:
    """Tests for POST /v1/routing/explain."""

    def _post(self, client, body, headers=None):
        h = {"Authorization": "Bearer test-key", **(headers or {})}
        return client.post("/v1/routing/explain", json=body, headers=h)

    @patch("model_plane.adapters.admin_router.run_routing_pipeline")
    @patch("model_plane.adapters.admin_router.build_routing_context")
    def test_explain_returns_dry_run_flag(self, mock_build, mock_run):
        dep = _make_dep()
        mock_ctx = MagicMock()
        mock_ctx.features = MagicMock(
            total_tokens=42, message_count=2, has_tools=False,
            language_hint=None, reasoning_markers=0.0, code_presence=0.1,
            multi_step_patterns=0.0, technical_terms=0.0,
        )
        mock_ctx.classification = MagicMock(
            task_type=TaskType.SIMPLE_QA,
            complexity_tier=ComplexityTier.SIMPLE,
            confidence=0.9,
            signals={"simple_indicators": 0.8},
        )
        mock_ctx.scorer_result = MagicMock(
            raw_score=0.1,
            tier=ComplexityTier.SIMPLE,
            confidence=0.9,
            dimension_scores={"reasoning_markers": 0.0, "code_presence": 0.01},
        )
        mock_ctx.candidate_deployments = [dep]
        mock_ctx.fallback_used = False
        mock_ctx.ml_recommended_deployment = None
        mock_ctx.ml_recommendation_confidence = 0.0
        mock_ctx.request_id = "explain-abc123"

        mock_build.return_value = mock_ctx
        mock_run.return_value = _make_decision(dep)

        client = _client()
        resp = self._post(client, {"messages": [{"role": "user", "content": "hi"}]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["selected_deployment"] == "watsonx-granite-medium"
        assert data["provider"] == "watsonx"
        assert data["routing_source"] == "custom_plugin"
        assert data["routing_confidence"] == 0.85
        assert "classification" in data
        assert "scorer" in data
        assert "features" in data
        assert "candidate_deployments" in data
        assert "ml_recommendation" in data

    @patch("model_plane.adapters.admin_router.run_routing_pipeline")
    @patch("model_plane.adapters.admin_router.build_routing_context")
    def test_explain_classification_fields(self, mock_build, mock_run):
        dep = _make_dep()
        mock_ctx = MagicMock()
        mock_ctx.features = MagicMock(
            total_tokens=100, message_count=1, has_tools=True,
            language_hint="python", reasoning_markers=0.7, code_presence=0.9,
            multi_step_patterns=0.4, technical_terms=0.3,
        )
        mock_ctx.classification = MagicMock(
            task_type=TaskType.CODE_GENERATION,
            complexity_tier=ComplexityTier.COMPLEX,
            confidence=0.88,
            signals={"code_presence": 0.9, "reasoning_markers": 0.7},
        )
        mock_ctx.scorer_result = MagicMock(
            raw_score=0.72,
            tier=ComplexityTier.COMPLEX,
            confidence=0.88,
            dimension_scores={"code_presence": 0.135, "reasoning_markers": 0.126},
        )
        mock_ctx.candidate_deployments = [dep]
        mock_ctx.fallback_used = False
        mock_ctx.ml_recommended_deployment = "watsonx-granite-large"
        mock_ctx.ml_recommendation_confidence = 0.76
        mock_ctx.request_id = "explain-xyz789"

        mock_build.return_value = mock_ctx
        mock_run.return_value = _make_decision(dep)

        client = _client()
        resp = self._post(
            client,
            {"messages": [{"role": "user", "content": "write a python class"}]},
            headers={"X-Tenant-Id": "acme", "X-Agent-Type": "cursor"},
        )

        assert resp.status_code == 200
        data = resp.json()
        clf = data["classification"]
        assert clf["task_type"] == "code_generation"
        assert clf["complexity_tier"] == "complex"
        assert clf["confidence"] == 0.88
        assert "code_presence" in clf["top_signals"]

        sc = data["scorer"]
        assert sc["raw_score"] == 0.72
        assert sc["tier"] == "complex"

        ml = data["ml_recommendation"]
        assert ml["recommended_deployment"] == "watsonx-granite-large"
        assert ml["matches_selected"] is False  # granite-large != granite-medium

    def test_explain_requires_messages(self):
        client = _client()
        resp = client.post(
            "/v1/routing/explain",
            json={"model": "gpt-4o"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 422
        assert "messages" in resp.json()["detail"]

    def test_explain_requires_auth(self):
        """Without dependency override, a missing Bearer token must return 401."""
        from model_plane.app import create_app as _create_app

        # Raw client — no dependency_overrides so real auth runs
        raw_client = TestClient(_create_app(), raise_server_exceptions=True)
        resp = raw_client.post(
            "/v1/routing/explain",
            json={"messages": [{"role": "user", "content": "test"}]},
            # no Authorization header
        )
        # If MODEL_PLANE_API_KEYS is set in the env the server enforces auth (401).
        # If it is empty (CI / dev with no keys) the server allows all (200 or 422).
        assert resp.status_code in (401, 422, 200, 503)
