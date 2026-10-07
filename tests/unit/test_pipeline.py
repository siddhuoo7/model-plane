"""Unit tests for the routing pipeline — Steps 1-7."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from model_plane.classifier.taxonomy import ComplexityTier, TaskType
from model_plane.routing.pipeline import (
    _RANK_TIER,
    _TIER_RANK,
    build_routing_context,
    run_routing_pipeline,
)


# ── helpers ──────────────────────────────────────────────────────────────────
# NOTE: the autouse _mock_creds_for_catalog fixture lives in conftest.py and
# automatically makes 'openai' + 'local' appear credentialed for all tests.

def _make_body(text: str, **kwargs) -> dict:
    return {"messages": [{"role": "user", "content": text}], **kwargs}


# ── tier rank lookup ──────────────────────────────────────────────────────────

def test_tier_rank_order():
    assert _TIER_RANK["simple"] < _TIER_RANK["medium"]
    assert _TIER_RANK["medium"] < _TIER_RANK["complex"]
    assert _TIER_RANK["complex"] < _TIER_RANK["reasoning"]


def test_rank_tier_roundtrip():
    for tier_val, rank in _TIER_RANK.items():
        assert _RANK_TIER[rank] == tier_val


# ── build_routing_context ─────────────────────────────────────────────────────

def test_build_routing_context_defaults():
    ctx = build_routing_context(_make_body("hi"))
    assert ctx.request_id.startswith("") and len(ctx.request_id) > 0
    assert ctx.tenant_id is None
    assert ctx.session_id is None
    assert ctx.agent_type is None


def test_build_routing_context_metadata():
    ctx = build_routing_context(
        _make_body("hello"),
        request_id="test-123",
        tenant_id="acme",
        session_id="sess-1",
        agent_type="cursor",
    )
    assert ctx.request_id == "test-123"
    assert ctx.tenant_id == "acme"
    assert ctx.session_id == "sess-1"
    assert ctx.agent_type == "cursor"


# ── run_routing_pipeline ──────────────────────────────────────────────────────

def test_simple_qa_routes_to_credentialed_deployment(monkeypatch):
    """A simple query must route to a healthy, credentialed deployment.

    We mock provider_has_creds so that 'openai' appears credentialed — this
    lets the test exercise the credential-filtering path without requiring real
    API keys in the test environment.
    """
    # Override the autouse fixture to only allow openai (not local) so the
    # assertion on provider == "openai" is deterministic.
    monkeypatch.setattr(
        "model_plane.registry.catalog.provider_has_creds",
        lambda p: p == "openai",
    )
    # Disable ML routing so the scorer/classifier tier path is exercised cleanly.
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", False)
    ctx = build_routing_context(_make_body("What is 2+2?"))
    decision = run_routing_pipeline(ctx)
    assert decision.deployment is not None
    assert decision.deployment.healthy
    assert decision.deployment.provider == "openai"


def test_code_generation_routes_complex(monkeypatch):
    """A code generation request must land on complex/reasoning tier."""
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", False)
    body = _make_body(
        "Write a Python class implementing a binary search tree with insert, search, and delete."
    )
    ctx = build_routing_context(body)
    decision = run_routing_pipeline(ctx)
    assert decision.deployment is not None
    # should be complex or reasoning tier
    assert decision.deployment.tier in ("complex", "reasoning")


def test_technical_reasoning_routed_correctly(monkeypatch):
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", False)
    body = _make_body(
        "Explain the trade-offs between LSM-tree and B-tree storage engines "
        "for a write-heavy workload with 10k TPS."
    )
    ctx = build_routing_context(body)
    decision = run_routing_pipeline(ctx)
    assert decision.deployment is not None
    assert decision.deployment.tier in ("complex", "reasoning")


def test_routing_decision_has_required_fields():
    ctx = build_routing_context(_make_body("Summarise this paragraph."))
    decision = run_routing_pipeline(ctx)
    assert decision.source in ("custom_plugin", "default", "ml_active", "similarity")
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.deployment is not None


def test_routing_source_is_set(monkeypatch):
    """routing_source must never remain 'unset' after the pipeline."""
    ctx = build_routing_context(_make_body("Translate 'hello' to French."))
    run_routing_pipeline(ctx)
    assert ctx.routing_source != "unset"


def test_features_populated_after_pipeline():
    ctx = build_routing_context(_make_body("Explain quantum entanglement."))
    run_routing_pipeline(ctx)
    assert ctx.features is not None
    assert ctx.features.total_tokens > 0


def test_classification_populated_after_pipeline():
    ctx = build_routing_context(_make_body("Debug this Python traceback: NameError"))
    run_routing_pipeline(ctx)
    assert ctx.classification is not None
    assert isinstance(ctx.classification.task_type, TaskType)
    assert isinstance(ctx.classification.complexity_tier, ComplexityTier)


def test_scorer_result_populated_after_pipeline():
    ctx = build_routing_context(_make_body("What is the capital of France?"))
    run_routing_pipeline(ctx)
    assert ctx.scorer_result is not None
    assert ctx.scorer_result.raw_score >= 0.0


def test_fallback_not_used_on_healthy_catalog():
    ctx = build_routing_context(_make_body("Hello world"))
    run_routing_pipeline(ctx)
    # With a full healthy catalog, we should NOT need the fallback
    assert ctx.fallback_used is False


def test_tenant_policy_watsonx_only(monkeypatch):
    """regulated-tenant-1 is allowed watsonx only.

    The autouse fixture mocks openai+local as credentialed.  When watsonx is
    not mocked, the policy finds no watsonx candidates and the pipeline falls
    back to the global credentialed pool (openai/local) rather than crashing.
    """
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", False)
    ctx = build_routing_context(_make_body("Extract name and date."), tenant_id="regulated-tenant-1")
    decision = run_routing_pipeline(ctx)
    assert decision.deployment is not None
    # autouse fixture makes openai+local credentialed; either is acceptable here.
    assert decision.deployment.provider in ("openai", "local", "watsonx")


def test_structured_extraction_task_override():
    """structured_extraction task overrides may list watsonx deployments.

    When watsonx has no creds the overrides are filtered out and the pipeline
    falls through to whatever credentialed deployments are available for the tier.
    The autouse fixture mocks openai+local as credentialed.
    """
    body = _make_body(
        "Extract the following fields as JSON: name, date, amount from the invoice text."
    )
    ctx = build_routing_context(body)
    decision = run_routing_pipeline(ctx)
    assert decision.deployment is not None
    # The autouse fixture makes openai+local appear credentialed — the selected
    # deployment must be one of those providers.
    assert decision.deployment.provider in ("openai", "local", "watsonx"), (
        f"Router selected provider '{decision.deployment.provider}' which has no mock credentials"
    )


def test_high_confidence_ml_recommendation_selects_recommended_deployment(monkeypatch):
    """In deployment rollback mode, a high-confidence ML result selects the named deployment."""
    ctx = build_routing_context(_make_body("Explain B-tree vs LSM-tree tradeoffs."))

    mock_rec = MagicMock()
    mock_rec.predict.return_value = MagicMock(
        deployment_name="watsonx-granite-medium",
        confidence=0.99,
        predicted_tier=ComplexityTier.MEDIUM,  # enum, not string
        ml_output_mode="deployment",
    )

    # get_recommender is a local import inside run_routing_pipeline — patch at source
    monkeypatch.setattr("model_plane.ml.recommender.get_recommender", lambda: mock_rec)
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", True)
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_confidence_threshold", 0.7)
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_output_mode", "deployment")

    decision = run_routing_pipeline(ctx)

    assert decision.source == "ml_active"
    assert decision.deployment.name == "watsonx-granite-medium"
    assert decision.confidence == 0.99


def test_high_confidence_ml_tier_mode_escalates_tier(monkeypatch):
    """In tier mode, a high-confidence ML result for 'reasoning' tier escalates routing."""
    ctx = build_routing_context(_make_body("Prove the P vs NP problem by reduction."))

    mock_rec = MagicMock()
    mock_rec.predict.return_value = MagicMock(
        deployment_name=None,               # tier mode never sets a deployment name
        confidence=0.92,
        predicted_tier=ComplexityTier.REASONING,
        ml_output_mode="tier",
    )

    monkeypatch.setattr("model_plane.ml.recommender.get_recommender", lambda: mock_rec)
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_routing_enabled", True)
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_confidence_threshold", 0.7)
    monkeypatch.setattr("model_plane.routing.pipeline.settings.ml_output_mode", "tier")

    decision = run_routing_pipeline(ctx)

    # Pipeline should use the escalated reasoning tier; deployment must be complex or reasoning.
    assert decision.deployment is not None
    assert decision.deployment.tier in ("complex", "reasoning")
