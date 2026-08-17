"""Unit tests for the LocalMLRecommender — especially _tier_to_default_deployment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from model_plane.classifier.taxonomy import ComplexityTier
from model_plane.ml.recommender import LocalMLRecommender

# ── _tier_to_default_deployment ───────────────────────────────────────────────

def test_prefers_watsonx_over_openai_same_tier():
    """When both watsonx and openai share a tier, watsonx should be picked first.

    get_catalog is imported *locally* inside _tier_to_default_deployment, so
    we patch it at the registry level and also swap the module-level singleton.
    """
    import model_plane.registry.catalog as cat_mod
    from model_plane.registry.catalog import DeploymentConfig, ModelCatalog

    wx_dep = DeploymentConfig(
        name="watsonx-medium",
        litellm_model="watsonx/ibm/granite-4-h-small",
        provider="watsonx",
        tier="medium",
    )
    oai_dep = DeploymentConfig(
        name="openai-gpt4o-mini",
        litellm_model="gpt-4o-mini",
        provider="openai",
        tier="medium",
    )
    catalog = ModelCatalog()
    catalog.deployments = {"openai-gpt4o-mini": oai_dep, "watsonx-medium": wx_dep}
    catalog.tier_map = {"medium": ["openai-gpt4o-mini", "watsonx-medium"]}

    routing_cfg = {"preferred_providers": ["watsonx", "local", "anthropic", "openai"]}

    rec = LocalMLRecommender.__new__(LocalMLRecommender)
    rec._model = None
    rec._loaded = False

    original_catalog = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.config.settings") as mock_settings:
            mock_settings.load_routing_config.return_value = routing_cfg
            mock_settings.default_model = "watsonx-medium"
            result = rec._tier_to_default_deployment("medium")
    finally:
        cat_mod._catalog = original_catalog

    assert result == "watsonx-medium"


def test_fallback_when_no_tier_match():
    """When no deployment matches the tier, return first healthy deployment."""
    import model_plane.registry.catalog as cat_mod
    from model_plane.registry.catalog import DeploymentConfig, ModelCatalog

    dep = DeploymentConfig(
        name="fallback-dep",
        litellm_model="watsonx/ibm/granite-4-h-small",
        provider="watsonx",
        tier="medium",
    )
    catalog = ModelCatalog()
    catalog.deployments = {"fallback-dep": dep}
    catalog.tier_map = {}  # no "reasoning" tier

    rec = LocalMLRecommender.__new__(LocalMLRecommender)
    rec._model = None
    rec._loaded = False

    original_catalog = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.config.settings") as mock_settings:
            mock_settings.load_routing_config.return_value = {}
            mock_settings.default_model = "fallback-dep"
            result = rec._tier_to_default_deployment("reasoning")
    finally:
        cat_mod._catalog = original_catalog

    assert result == "fallback-dep"


def test_predict_without_model_returns_scorer_fallback():
    """With no model loaded, predict() must not raise and return scorer_fallback."""
    import model_plane.registry.catalog as cat_mod
    from model_plane.classifier.features import RequestFeatures
    from model_plane.registry.catalog import DeploymentConfig, ModelCatalog
    from model_plane.scorer.scorer import ScorerResult

    dep = DeploymentConfig(name="wx-small", litellm_model="watsonx/x", provider="watsonx", tier="simple")
    catalog = ModelCatalog()
    catalog.deployments = {"wx-small": dep}
    catalog.tier_map = {"simple": ["wx-small"]}

    rec = LocalMLRecommender.__new__(LocalMLRecommender)
    rec._model = None
    rec._loaded = False
    rec._model_path = Path("nonexistent.joblib")

    feats = RequestFeatures(
        total_tokens=50, user_message_count=1,
        reasoning_markers=0.0, code_presence=0.0, simple_indicators=0.9,
        multi_step_patterns=0.0, technical_terms=0.0, token_count_signal=0.05,
        creative_markers=0.0, question_complexity=0.0, constraint_count=0.0,
        imperative_verbs=0.0, output_format=0.0, domain_specificity=0.0,
        reference_complexity=0.0, negation_complexity=0.0,
        has_tools=False, language_hint=None, last_user_text="hi",
    )
    scorer_res = ScorerResult(
        tier=ComplexityTier.SIMPLE, raw_score=0.1, confidence=0.9, dimension_scores={}
    )

    original_catalog = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.config.settings") as mock_settings:
            mock_settings.load_routing_config.return_value = {}
            mock_settings.default_model = "wx-small"
            result = rec.predict(feats, scorer_res)
    finally:
        cat_mod._catalog = original_catalog

    assert result.source == "scorer_fallback"
    assert result.confidence == 0.0
