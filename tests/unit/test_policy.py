"""Unit tests for policy engine."""
from model_plane.classifier.taxonomy import TaskType
from model_plane.policy.engine import PolicyEngine, TenantPolicy
from model_plane.registry.catalog import DeploymentConfig


def _dep(name: str, provider: str = "openai", tier: str = "medium") -> DeploymentConfig:
    return DeploymentConfig(name=name, litellm_model=f"{provider}/{name}", provider=provider, tier=tier)


def test_no_policy_allows_all():
    engine = PolicyEngine()
    candidates = [_dep("gpt4o", "openai"), _dep("claude", "anthropic")]
    decision = engine.enforce("tenant-x", TaskType.SIMPLE_QA, candidates)
    assert decision.allowed
    assert len(decision.filtered_deployments) == 2


def test_allowed_providers_filter():
    engine = PolicyEngine({"t1": TenantPolicy(
        tenant_id="t1", allowed_providers=["anthropic"]
    )})
    candidates = [_dep("gpt4o", "openai"), _dep("claude", "anthropic")]
    decision = engine.enforce("t1", TaskType.SIMPLE_QA, candidates)
    assert decision.allowed
    assert len(decision.filtered_deployments) == 1
    assert decision.filtered_deployments[0].name == "claude"


def test_denied_providers_filter():
    engine = PolicyEngine({"t1": TenantPolicy(
        tenant_id="t1", denied_providers=["openai"]
    )})
    candidates = [_dep("gpt4o", "openai"), _dep("claude", "anthropic")]
    decision = engine.enforce("t1", TaskType.SIMPLE_QA, candidates)
    assert decision.allowed
    assert all(d.provider != "openai" for d in decision.filtered_deployments)


def test_require_local_filters_cloud():
    engine = PolicyEngine({"t1": TenantPolicy(
        tenant_id="t1", require_local=True
    )})
    candidates = [_dep("gpt4o", "openai"), _dep("local-small", "local")]
    decision = engine.enforce("t1", TaskType.SIMPLE_QA, candidates)
    assert len(decision.filtered_deployments) == 1
    assert decision.filtered_deployments[0].provider == "local"


def test_token_budget_exceeded_blocks():
    engine = PolicyEngine({"t1": TenantPolicy(
        tenant_id="t1", max_tokens_per_request=100
    )})
    candidates = [_dep("gpt4o", "openai")]
    decision = engine.enforce("t1", TaskType.SIMPLE_QA, candidates, estimated_tokens=50000)
    assert not decision.allowed
    assert decision.reason == "token_budget_exceeded"


def test_regulated_task_override():
    engine = PolicyEngine({"t1": TenantPolicy(
        tenant_id="t1",
        regulated_tasks=["structured_extraction"],
        regulated_deployment="watsonx-regulated",
    )})
    candidates = [_dep("watsonx-regulated", "watsonx"), _dep("gpt4o", "openai")]
    decision = engine.enforce("t1", TaskType.STRUCTURED_EXTRACTION, candidates)
    assert decision.allowed
    assert len(decision.filtered_deployments) == 1
    assert decision.filtered_deployments[0].name == "watsonx-regulated"
