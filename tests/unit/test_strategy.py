"""Unit tests for the routing strategy (Step 7)."""

from __future__ import annotations

from model_plane.classifier.taxonomy import ComplexityTier, TaskType
from model_plane.registry.catalog import DeploymentConfig
from model_plane.routing.context import RoutingContext
from model_plane.routing.strategy import _score_candidate, select_deployment

# ── fixture helpers ───────────────────────────────────────────────────────────

def _dep(name: str, tier: str = "medium", provider: str = "watsonx",
         cost: float = 0.001, context_limit: int = 8192,
         capabilities: list[str] | None = None) -> DeploymentConfig:
    return DeploymentConfig(
        name=name,
        litellm_model=f"fake/{name}",
        provider=provider,
        tier=tier,
        cost_per_1k_input=cost,
        cost_per_1k_output=cost * 3,
        context_limit=context_limit,
        capabilities=capabilities or ["text"],
    )


def _ctx(task: TaskType = TaskType.SIMPLE_QA, tier: ComplexityTier = ComplexityTier.MEDIUM,
         tokens: int = 100) -> RoutingContext:
    from model_plane.classifier.features import RequestFeatures

    ctx = RoutingContext(request_id="test", raw_request={})
    ctx.classification = type("C", (), {
        "task_type": task,
        "complexity_tier": tier,
    })()
    from model_plane.scorer.scorer import ScorerResult
    ctx.scorer_result = ScorerResult(
        tier=tier, raw_score=0.3, confidence=0.8, dimension_scores={}
    )
    ctx.features = RequestFeatures(
        total_tokens=tokens,
        user_message_count=1,
        reasoning_markers=0.0,
        code_presence=0.0,
        simple_indicators=0.5,
        multi_step_patterns=0.0,
        technical_terms=0.0,
        token_count_signal=0.1,
        creative_markers=0.0,
        question_complexity=0.1,
        constraint_count=0.0,
        imperative_verbs=0.0,
        output_format=0.0,
        domain_specificity=0.0,
        reference_complexity=0.0,
        negation_complexity=0.0,
        has_tools=False,
        language_hint=None,
        last_user_text="hello",
    )
    return ctx


# ── _score_candidate ──────────────────────────────────────────────────────────

def test_tier_match_scores_higher_than_mismatch():
    dep_match = _dep("a", tier="medium")
    dep_mismatch = _dep("b", tier="simple")
    ctx = _ctx(tier=ComplexityTier.MEDIUM)
    s_match = _score_candidate(dep_match, TaskType.SIMPLE_QA, ComplexityTier.MEDIUM, ctx)
    s_mismatch = _score_candidate(dep_mismatch, TaskType.SIMPLE_QA, ComplexityTier.MEDIUM, ctx)
    assert s_match > s_mismatch


def test_code_capability_affinity_bonus():
    dep_with_code = _dep("coder", capabilities=["text", "code", "function-calling"])
    dep_no_code = _dep("plain", capabilities=["text"])
    ctx = _ctx(task=TaskType.CODE_GENERATION, tier=ComplexityTier.COMPLEX)
    s_code = _score_candidate(dep_with_code, TaskType.CODE_GENERATION, ComplexityTier.COMPLEX, ctx)
    s_plain = _score_candidate(dep_no_code, TaskType.CODE_GENERATION, ComplexityTier.COMPLEX, ctx)
    assert s_code > s_plain


def test_context_overflow_penalised():
    dep = _dep("small-ctx", context_limit=100)  # tiny context
    ctx = _ctx(tokens=200)                       # request exceeds it
    s = _score_candidate(dep, TaskType.SIMPLE_QA, ComplexityTier.SIMPLE, ctx)
    dep_large = _dep("large-ctx", context_limit=10000)
    s_large = _score_candidate(dep_large, TaskType.SIMPLE_QA, ComplexityTier.SIMPLE, ctx)
    assert s < s_large


# ── select_deployment ─────────────────────────────────────────────────────────

def test_select_from_single_candidate():
    dep = _dep("only-one", tier="medium")
    ctx = _ctx()
    decision = select_deployment(ctx, [dep])
    assert decision is not None
    assert decision.deployment.name == "only-one"


def test_select_from_empty_returns_none():
    ctx = _ctx()
    assert select_deployment(ctx, []) is None


def test_select_prefers_watsonx_over_openai_on_tie():
    """When scores are close, preferred_providers order breaks ties (watsonx first)."""
    dep_watsonx = _dep("wx", tier="medium", provider="watsonx")
    dep_openai = _dep("oai", tier="medium", provider="openai")
    ctx = _ctx()
    decision = select_deployment(ctx, [dep_watsonx, dep_openai])
    assert decision is not None
    assert decision.deployment.name == "wx"


def test_confidence_is_bounded():
    deps = [_dep(f"d{i}", tier="medium", provider="watsonx") for i in range(5)]
    ctx = _ctx()
    decision = select_deployment(ctx, deps)
    assert decision is not None
    assert 0.0 <= decision.confidence <= 1.0
