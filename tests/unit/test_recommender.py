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
    Credentials are stubbed to True for both providers so the test is isolated
    from whatever API keys happen to be in the test environment.
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
        with patch("model_plane.ml.recommender.settings") as mock_settings, \
             patch("model_plane.provider_creds.provider_has_creds", return_value=True), \
             patch("model_plane.registry.catalog.provider_has_creds", return_value=True):
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
        with patch("model_plane.ml.recommender.settings") as mock_settings:
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
        with patch("model_plane.ml.recommender.settings") as mock_settings:
            mock_settings.ml_output_mode = "tier"
            mock_settings.load_routing_config.return_value = {}
            mock_settings.default_model = "wx-small"
            result = rec.predict(feats, scorer_res)
    finally:
        cat_mod._catalog = original_catalog

    assert result.source == "scorer_fallback"
    assert result.confidence == 0.0


# ── Sub-Task 2.2: tier output mode ────────────────────────────────────────────

def _make_tier_recommender() -> "LocalMLRecommender":
    """Instantiate a LocalMLRecommender backed by the retrained tier model."""
    from model_plane.ml.recommender import LocalMLRecommender
    return LocalMLRecommender(Path("models/router_ml_tier_v1.joblib"))


def _make_features(
    reasoning: float = 0.0,
    code: float = 0.0,
    simple: float = 0.0,
    token: float = 0.05,
) -> "RequestFeatures":
    from model_plane.classifier.features import RequestFeatures
    return RequestFeatures(
        total_tokens=int(token * 8000),
        user_message_count=1,
        reasoning_markers=reasoning,
        code_presence=code,
        simple_indicators=simple,
        multi_step_patterns=0.0,
        technical_terms=0.0,
        token_count_signal=token,
        creative_markers=0.0,
        question_complexity=0.0,
        constraint_count=0.0,
        imperative_verbs=0.0,
        output_format=0.0,
        domain_specificity=0.0,
        reference_complexity=0.0,
        negation_complexity=0.0,
        has_tools=False,
        language_hint=None,
        last_user_text="test",
    )


def test_tier_recommender_output_is_enum():
    """predict() in tier mode must return a ComplexityTier enum, not a string."""
    import model_plane.registry.catalog as cat_mod
    from model_plane.classifier.taxonomy import ComplexityTier
    from model_plane.ml.recommender import MLRecommendation
    from model_plane.registry.catalog import DeploymentConfig, ModelCatalog
    from model_plane.scorer.scorer import ScorerResult

    dep = DeploymentConfig(name="wx-small", litellm_model="watsonx/x", provider="watsonx", tier="simple")
    catalog = ModelCatalog()
    catalog.deployments = {"wx-small": dep}
    catalog.tier_map = {"simple": ["wx-small"]}

    rec = _make_tier_recommender()
    feats = _make_features(simple=0.9, token=0.05)
    scorer_res = ScorerResult(
        tier=ComplexityTier.SIMPLE, raw_score=0.1, confidence=0.9, dimension_scores={}
    )

    original_catalog = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.ml.recommender.settings") as s:
            s.ml_output_mode = "tier"
            result = rec.predict(feats, scorer_res)
    finally:
        cat_mod._catalog = original_catalog

    assert isinstance(result, MLRecommendation)
    assert isinstance(result.predicted_tier, ComplexityTier), (
        f"Expected ComplexityTier, got {type(result.predicted_tier)}: {result.predicted_tier}"
    )
    assert result.ml_output_mode == "tier"
    assert result.deployment_name is None  # tier mode never returns a deployment name


def test_tier_recommender_fallback_returns_enum():
    """scorer_fallback path also returns a ComplexityTier enum."""
    from model_plane.classifier.taxonomy import ComplexityTier
    from model_plane.ml.recommender import LocalMLRecommender
    from model_plane.scorer.scorer import ScorerResult

    rec = LocalMLRecommender.__new__(LocalMLRecommender)
    rec._model = None
    rec._loaded = False
    rec._model_path = Path("nonexistent.joblib")

    feats = _make_features()
    scorer_res = ScorerResult(
        tier=ComplexityTier.MEDIUM, raw_score=0.3, confidence=0.7, dimension_scores={}
    )

    with patch("model_plane.ml.recommender.settings") as s:
        s.ml_output_mode = "tier"
        result = rec.predict(feats, scorer_res)

    assert result.source == "scorer_fallback"
    assert isinstance(result.predicted_tier, ComplexityTier)
    assert result.predicted_tier == ComplexityTier.MEDIUM


def test_tier_recommender_deployment_mode_returns_string():
    """In deployment rollback mode, predicted_tier is still returned but deployment_name is set."""
    import model_plane.registry.catalog as cat_mod
    from model_plane.classifier.taxonomy import ComplexityTier
    from model_plane.ml.recommender import LocalMLRecommender
    from model_plane.registry.catalog import DeploymentConfig, ModelCatalog
    from model_plane.scorer.scorer import ScorerResult

    # Build legacy deployment-name model inline using the old label
    import io
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import tempfile

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    # Two trivial classes
    clf.fit([[0.0] * 14, [1.0] * 14], [0, 1])
    dep_names = ["watsonx-small", "watsonx-large"]

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
        joblib.dump({"model": clf, "class_names": dep_names, "label_column": "selected_deployment"}, tf.name)
        model_path = Path(tf.name)

    dep_small = DeploymentConfig(name="watsonx-small", litellm_model="watsonx/x", provider="watsonx", tier="simple")
    dep_large = DeploymentConfig(name="watsonx-large", litellm_model="watsonx/y", provider="watsonx", tier="reasoning")
    catalog = ModelCatalog()
    catalog.deployments = {"watsonx-small": dep_small, "watsonx-large": dep_large}
    catalog.tier_map = {"simple": ["watsonx-small"], "reasoning": ["watsonx-large"]}

    rec = LocalMLRecommender(model_path)
    feats = _make_features()
    scorer_res = ScorerResult(
        tier=ComplexityTier.SIMPLE, raw_score=0.1, confidence=0.9, dimension_scores={}
    )

    original_catalog = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.ml.recommender.settings") as s:
            s.ml_output_mode = "deployment"
            s.load_routing_config.return_value = {}
            s.default_model = "watsonx-small"
            result = rec.predict(feats, scorer_res)
    finally:
        cat_mod._catalog = original_catalog
        model_path.unlink(missing_ok=True)

    assert result.ml_output_mode == "deployment"
    assert result.deployment_name is not None
    assert isinstance(result.deployment_name, str)


def test_trainer_load_dataset_tier_column():
    """load_dataset with label_column='complexity_tier' reads the 'tier' CSV column."""
    from model_plane.ml.trainer import load_dataset

    X, y, class_names = load_dataset(
        Path("model_plane/data/training_data.csv"),
        label_column="complexity_tier",
    )
    assert len(X) > 0
    assert set(class_names) == {"simple", "medium", "complex", "reasoning"}
    # All returned class indices are valid
    assert all(0 <= idx < len(class_names) for idx in y)


def test_trainer_tier_class_order_is_stable():
    """Tier labels are always in canonical order: simple=0, medium=1, complex=2, reasoning=3."""
    from model_plane.ml.trainer import TIER_LABELS, load_dataset

    _, _, class_names = load_dataset(
        Path("model_plane/data/training_data.csv"),
        label_column="complexity_tier",
    )
    assert class_names == TIER_LABELS


# ── Sub-Task 2.4: ML Guard Rails ──────────────────────────────────────────────

def _make_rec_no_model() -> "LocalMLRecommender":
    """Unloaded recommender — useful for guard method unit tests."""
    from model_plane.ml.recommender import LocalMLRecommender
    rec = LocalMLRecommender.__new__(LocalMLRecommender)
    rec._model = None
    rec._loaded = False
    rec._model_path = Path("nonexistent.joblib")
    return rec


def _scorer(tier: "ComplexityTier", conf: float = 0.6) -> "ScorerResult":
    from model_plane.scorer.scorer import ScorerResult
    return ScorerResult(tier=tier, raw_score=0.3, confidence=conf, dimension_scores={})


# ── ComplexityTier ordering ────────────────────────────────────────────────────

class TestComplexityTierOrdering:
    def test_simple_lt_medium(self):
        assert ComplexityTier.SIMPLE < ComplexityTier.MEDIUM

    def test_medium_lt_complex(self):
        assert ComplexityTier.MEDIUM < ComplexityTier.COMPLEX

    def test_complex_lt_reasoning(self):
        assert ComplexityTier.COMPLEX < ComplexityTier.REASONING

    def test_reasoning_not_lt_complex(self):
        assert not (ComplexityTier.REASONING < ComplexityTier.COMPLEX)

    def test_equal_tiers(self):
        assert ComplexityTier.MEDIUM <= ComplexityTier.MEDIUM
        assert ComplexityTier.MEDIUM >= ComplexityTier.MEDIUM

    def test_gt_and_ge(self):
        assert ComplexityTier.REASONING > ComplexityTier.SIMPLE
        assert ComplexityTier.COMPLEX >= ComplexityTier.COMPLEX


# ── skipMl guard ──────────────────────────────────────────────────────────────

class TestSkipMlGuard:
    def _rec(self):
        return _make_rec_no_model()

    def test_fires_when_both_conditions_met(self):
        """classifier_conf >= 0.90 AND scorer_conf < 0.45 → skip."""
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {
                "ml_guards": {"skip_ml_classifier_conf": 0.90, "skip_ml_scorer_conf": 0.45}
            }
            assert rec._should_skip_ml(classifier_conf=0.92, scorer_conf=0.30) is True

    def test_does_not_fire_when_scorer_conf_high(self):
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {
                "ml_guards": {"skip_ml_classifier_conf": 0.90, "skip_ml_scorer_conf": 0.45}
            }
            assert rec._should_skip_ml(classifier_conf=0.92, scorer_conf=0.60) is False

    def test_does_not_fire_when_classifier_conf_low(self):
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {
                "ml_guards": {"skip_ml_classifier_conf": 0.90, "skip_ml_scorer_conf": 0.45}
            }
            assert rec._should_skip_ml(classifier_conf=0.80, scorer_conf=0.30) is False

    def test_predict_returns_skip_ml_guard_source(self):
        """When skipMl fires, predict() returns source='skip_ml_guard'."""
        rec = _make_tier_recommender()
        feats = _make_features()
        scorer_res = _scorer(ComplexityTier.COMPLEX, conf=0.30)  # low scorer conf

        with patch("model_plane.ml.recommender.settings") as s:
            s.ml_output_mode = "tier"
            s.load_routing_config.return_value = {
                "ml_guards": {"skip_ml_classifier_conf": 0.90, "skip_ml_scorer_conf": 0.45}
            }
            result = rec.predict(
                feats, scorer_res,
                blended_tier=ComplexityTier.COMPLEX,
                classifier_conf=0.95,  # high → triggers skipMl
            )

        assert result.source == "skip_ml_guard"
        assert result.predicted_tier == ComplexityTier.COMPLEX  # blended tier returned
        assert result.confidence == 0.0


# ── upgrade_cap guard ─────────────────────────────────────────────────────────

class TestUpgradeCapGuard:
    def _rec(self):
        return _make_rec_no_model()

    def test_caps_reasoning_to_medium_when_scorer_simple_and_confident(self):
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"upgrade_cap_conf": 0.80}}
            result = rec._apply_upgrade_cap(
                ml_tier=ComplexityTier.REASONING,
                scorer_tier=ComplexityTier.SIMPLE,
                classifier_conf=0.85,
            )
        assert result == ComplexityTier.MEDIUM

    def test_caps_complex_to_medium(self):
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"upgrade_cap_conf": 0.80}}
            result = rec._apply_upgrade_cap(
                ml_tier=ComplexityTier.COMPLEX,
                scorer_tier=ComplexityTier.SIMPLE,
                classifier_conf=0.82,
            )
        assert result == ComplexityTier.MEDIUM

    def test_does_not_cap_when_scorer_not_simple(self):
        """Guard only fires when scorer says SIMPLE."""
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"upgrade_cap_conf": 0.80}}
            result = rec._apply_upgrade_cap(
                ml_tier=ComplexityTier.REASONING,
                scorer_tier=ComplexityTier.MEDIUM,
                classifier_conf=0.90,
            )
        assert result == ComplexityTier.REASONING  # not capped

    def test_does_not_cap_below_conf_threshold(self):
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"upgrade_cap_conf": 0.80}}
            result = rec._apply_upgrade_cap(
                ml_tier=ComplexityTier.REASONING,
                scorer_tier=ComplexityTier.SIMPLE,
                classifier_conf=0.75,  # below 0.80
            )
        assert result == ComplexityTier.REASONING  # not capped

    def test_does_not_cap_when_ml_tier_is_medium(self):
        """Guard only caps tiers ABOVE MEDIUM."""
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"upgrade_cap_conf": 0.80}}
            result = rec._apply_upgrade_cap(
                ml_tier=ComplexityTier.MEDIUM,
                scorer_tier=ComplexityTier.SIMPLE,
                classifier_conf=0.90,
            )
        assert result == ComplexityTier.MEDIUM  # already at cap, no change


# ── downgrade_veto guard ──────────────────────────────────────────────────────

class TestDowngradeVetoGuard:
    def _rec(self):
        return _make_rec_no_model()

    def test_vetoes_downgrade_when_confident(self):
        """ML predicts SIMPLE but blended is COMPLEX → veto to COMPLEX."""
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"downgrade_veto_conf": 0.75}}
            result = rec._apply_downgrade_veto(
                ml_tier=ComplexityTier.SIMPLE,
                blended_tier=ComplexityTier.COMPLEX,
                classifier_conf=0.80,
            )
        assert result == ComplexityTier.COMPLEX

    def test_does_not_veto_when_ml_equals_blended(self):
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"downgrade_veto_conf": 0.75}}
            result = rec._apply_downgrade_veto(
                ml_tier=ComplexityTier.COMPLEX,
                blended_tier=ComplexityTier.COMPLEX,
                classifier_conf=0.90,
            )
        assert result == ComplexityTier.COMPLEX  # no change (not a downgrade)

    def test_does_not_veto_when_blended_is_simple(self):
        """Guard does not fire when blended_tier == SIMPLE."""
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"downgrade_veto_conf": 0.75}}
            result = rec._apply_downgrade_veto(
                ml_tier=ComplexityTier.SIMPLE,
                blended_tier=ComplexityTier.SIMPLE,
                classifier_conf=0.95,
            )
        assert result == ComplexityTier.SIMPLE  # blended is SIMPLE → no veto

    def test_does_not_veto_below_conf_threshold(self):
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"downgrade_veto_conf": 0.75}}
            result = rec._apply_downgrade_veto(
                ml_tier=ComplexityTier.SIMPLE,
                blended_tier=ComplexityTier.COMPLEX,
                classifier_conf=0.70,  # below 0.75
            )
        assert result == ComplexityTier.SIMPLE  # low conf → veto does not fire

    def test_allows_ml_escalation(self):
        """If ML recommends higher than blended, veto does not interfere."""
        rec = self._rec()
        with patch("model_plane.ml.recommender.settings") as s:
            s.load_routing_config.return_value = {"ml_guards": {"downgrade_veto_conf": 0.75}}
            result = rec._apply_downgrade_veto(
                ml_tier=ComplexityTier.REASONING,
                blended_tier=ComplexityTier.COMPLEX,
                classifier_conf=0.90,
            )
        assert result == ComplexityTier.REASONING  # escalation allowed


# ── Guard integration via predict() ──────────────────────────────────────────

class TestGuardsInPredict:
    """Verify guards fire end-to-end through the full predict() path."""

    def test_upgrade_cap_applied_in_predict(self):
        """Full predict(): scorer=SIMPLE, clf_conf=0.85 → reasoning capped to medium."""
        rec = _make_tier_recommender()
        feats = _make_features(simple=0.9, token=0.03)  # very simple
        scorer_res = _scorer(ComplexityTier.SIMPLE, conf=0.85)

        with patch("model_plane.ml.recommender.settings") as s:
            s.ml_output_mode = "tier"
            s.load_routing_config.return_value = {
                "ml_guards": {
                    "upgrade_cap_conf": 0.80,
                    "downgrade_veto_conf": 0.75,
                    "skip_ml_classifier_conf": 0.90,
                    "skip_ml_scorer_conf": 0.45,
                }
            }
            result = rec.predict(
                feats, scorer_res,
                blended_tier=ComplexityTier.SIMPLE,
                classifier_conf=0.85,
            )

        # Even if the model would say REASONING, the guard caps it at MEDIUM
        assert isinstance(result.predicted_tier, ComplexityTier)
        assert result.predicted_tier <= ComplexityTier.MEDIUM

    def test_downgrade_veto_applied_in_predict(self):
        """Full predict(): blended=COMPLEX, ml_conf=0.85, clf_conf=0.80 → no downgrade below COMPLEX."""
        rec = _make_tier_recommender()
        feats = _make_features(reasoning=0.8, code=0.9)  # complex signals
        scorer_res = _scorer(ComplexityTier.COMPLEX, conf=0.80)

        with patch("model_plane.ml.recommender.settings") as s:
            s.ml_output_mode = "tier"
            s.load_routing_config.return_value = {
                "ml_guards": {
                    "upgrade_cap_conf": 0.80,
                    "downgrade_veto_conf": 0.75,
                    "skip_ml_classifier_conf": 0.90,
                    "skip_ml_scorer_conf": 0.45,
                }
            }
            result = rec.predict(
                feats, scorer_res,
                blended_tier=ComplexityTier.COMPLEX,
                classifier_conf=0.80,
            )

        # Result must not be below COMPLEX (veto prevents downgrade)
        assert isinstance(result.predicted_tier, ComplexityTier)
        assert result.predicted_tier >= ComplexityTier.COMPLEX
