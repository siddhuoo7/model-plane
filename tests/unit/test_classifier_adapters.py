"""Tests for Sub-Task 2.1 — ClassifierAdapter protocol, BertClassifierAdapter,
LayaClassifier, and the get_classifier() factory.

All tests run without transformers or laya installed — they mock the underlying
SDKs and verify adapter behaviour in isolation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_plane.classifier.features import RequestFeatures, extract_features
from model_plane.classifier.taxonomy import ComplexityTier, TaskType


# ── helpers ───────────────────────────────────────────────────────────────────

def _features(text: str) -> RequestFeatures:
    return extract_features({"messages": [{"role": "user", "content": text}]})


def _run(coro):
    """Run a coroutine in a fresh private event loop (Python 3.11 compatible)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── factory tests ─────────────────────────────────────────────────────────────

def _make_factory_settings(**kwargs) -> MagicMock:
    """Return a MagicMock settings with sensible classifier defaults."""
    s = MagicMock()
    s.classifier_model = kwargs.get("classifier_model", "regex")
    s.bert_classifier_enabled = kwargs.get("bert_classifier_enabled", False)
    s.bert_classifier_model = kwargs.get("bert_classifier_model", "llm-semantic-router/mmbert32k-intent-classifier-merged")
    s.classifier_min_confidence = kwargs.get("classifier_min_confidence", 0.60)
    s.db_overrides_enabled = False
    return s


class TestGetClassifierFactory:
    """get_classifier() returns the right adapter based on settings."""

    def test_regex_returns_none(self):
        from model_plane.classifier.factory import get_classifier
        s = _make_factory_settings(classifier_model="regex", bert_classifier_enabled=False)
        with patch("model_plane.classifier.factory.settings", s):
            assert get_classifier() is None

    def test_unknown_value_returns_none(self):
        from model_plane.classifier.factory import get_classifier
        s = _make_factory_settings(classifier_model="unknown_backend", bert_classifier_enabled=False)
        with patch("model_plane.classifier.factory.settings", s):
            assert get_classifier() is None

    def test_mbert_returns_bert_adapter(self):
        from model_plane.classifier.bert_classifier import BertClassifierAdapter
        from model_plane.classifier.factory import get_classifier
        s = _make_factory_settings(classifier_model="mbert")
        with patch("model_plane.classifier.factory.settings", s):
            adapter = get_classifier()
            assert isinstance(adapter, BertClassifierAdapter)

    def test_laya_returns_laya_classifier(self):
        from model_plane.classifier.factory import get_classifier
        from model_plane.classifier.laya_classifier import LayaClassifier
        s = _make_factory_settings(classifier_model="laya")
        with patch("model_plane.classifier.factory.settings", s):
            adapter = get_classifier()
            assert isinstance(adapter, LayaClassifier)

    def test_bert_enabled_alias_maps_to_mbert(self):
        """Deprecated BERT_CLASSIFIER_ENABLED=true maps to mbert backend."""
        from model_plane.classifier.bert_classifier import BertClassifierAdapter
        from model_plane.classifier.factory import get_classifier
        s = _make_factory_settings(classifier_model="regex", bert_classifier_enabled=True)
        with patch("model_plane.classifier.factory.settings", s):
            adapter = get_classifier()
            assert isinstance(adapter, BertClassifierAdapter)

    def test_classifier_model_wins_over_bert_alias(self):
        """When CLASSIFIER_MODEL=laya and BERT_CLASSIFIER_ENABLED=true, laya wins."""
        from model_plane.classifier.factory import get_classifier
        from model_plane.classifier.laya_classifier import LayaClassifier
        s = _make_factory_settings(classifier_model="laya", bert_classifier_enabled=True)
        with patch("model_plane.classifier.factory.settings", s):
            adapter = get_classifier()
            assert isinstance(adapter, LayaClassifier)


# ── BertClassifierAdapter tests ───────────────────────────────────────────────

class TestBertClassifierAdapter:
    """BertClassifierAdapter wraps predict() and sets task_type only."""

    def _make_adapter(self, min_confidence: float = 0.60):
        from model_plane.classifier.bert_classifier import BertClassifierAdapter
        return BertClassifierAdapter(
            model_name="llm-semantic-router/mmbert32k-intent-classifier-merged",
            min_confidence=min_confidence,
        )

    def test_task_type_overridden_on_high_confidence(self):
        from model_plane.classifier.bert_classifier import BertClassificationResult
        adapter = self._make_adapter(min_confidence=0.60)
        features = _features("write a function to sort a list")

        mock_result = BertClassificationResult(
            task_type=TaskType.CODE_GENERATION,
            confidence=0.90,
        )
        with patch(
            "model_plane.classifier.bert_classifier.predict",
            return_value=mock_result,
        ):
            result = _run(adapter.classify(features))

        assert result is not None
        assert result.task_type == TaskType.CODE_GENERATION
        assert result.confidence == 0.90
        assert result.signals.get("_source") == 1.0

    def test_tier_is_re_derived_from_bert_task(self):
        """Tier should come from _derive_tier(bert_task) not from bert output."""
        from model_plane.classifier.bert_classifier import BertClassificationResult
        adapter = self._make_adapter(min_confidence=0.60)
        features = _features("debug why my test is failing")

        mock_result = BertClassificationResult(
            task_type=TaskType.CODE_DEBUGGING,
            confidence=0.85,
        )
        with patch(
            "model_plane.classifier.bert_classifier.predict",
            return_value=mock_result,
        ):
            result = _run(adapter.classify(features))

        assert result is not None
        # CODE_DEBUGGING maps to COMPLEX in TASK_DEFAULT_TIER
        assert result.complexity_tier == ComplexityTier.COMPLEX

    def test_returns_none_on_low_confidence(self):
        from model_plane.classifier.bert_classifier import BertClassificationResult
        adapter = self._make_adapter(min_confidence=0.80)
        features = _features("what is python?")

        mock_result = BertClassificationResult(
            task_type=TaskType.SIMPLE_QA,
            confidence=0.55,  # below 0.80 threshold
        )
        with patch(
            "model_plane.classifier.bert_classifier.predict",
            return_value=mock_result,
        ):
            result = _run(adapter.classify(features))

        assert result is None

    def test_returns_none_when_predict_returns_none(self):
        adapter = self._make_adapter()
        features = _features("hello world")

        with patch(
            "model_plane.classifier.bert_classifier.predict",
            return_value=None,
        ):
            result = _run(adapter.classify(features))

        assert result is None


# ── LayaClassifier tests ──────────────────────────────────────────────────────

class TestLayaClassifier:
    """LayaClassifier sets BOTH task_type AND complexity_tier from one call."""

    def _make_laya_result(
        self,
        task_label: str = "code_debugging",
        task_score: float = 0.85,
        tier_label: str = "complex",
        laya_model: str = "laya-v1",
    ) -> dict:
        """Build a mock Laya result matching the real laya response format.

        laya score questions return:
          {"type": "score", "score": <float>, "probabilities": {"0": p0, "1": p1, ...}, ...}

        TASK_TYPE_LABELS is an ordered list; we find the index for task_label and
        assign the highest probability to it, distributing 0.05 across the rest.
        """
        from model_plane.classifier.laya_classifier import TASK_TYPE_LABELS
        winner_idx = TASK_TYPE_LABELS.index(task_label) if task_label in TASK_TYPE_LABELS else 0
        noise = round((1.0 - task_score) / max(len(TASK_TYPE_LABELS) - 1, 1), 4)
        probabilities = {
            str(i): (task_score if i == winner_idx else noise)
            for i in range(len(TASK_TYPE_LABELS))
        }
        return {
            "answers": {
                "task_type": {
                    "type": "score",
                    "score": float(winner_idx),
                    "probabilities": probabilities,
                    "confidence": task_score,
                },
                "complexity_tier": {"choice": tier_label},
            },
            "routing": {"model": laya_model},
        }

    def test_task_type_and_tier_both_overridden(self):
        from model_plane.classifier.laya_classifier import LayaClassifier
        adapter = LayaClassifier(min_confidence=0.50)
        features = _features("debug the failing test")

        mock_router = MagicMock()
        mock_router.predict.return_value = self._make_laya_result(
            task_label="code_debugging",
            task_score=0.88,
            tier_label="complex",
        )

        with patch("model_plane.classifier.laya_classifier._router", mock_router):
            result = _run(adapter.classify(features))

        assert result is not None
        assert result.task_type == TaskType.CODE_DEBUGGING
        assert result.complexity_tier == ComplexityTier.COMPLEX
        assert result.signals.get("_source") == 2.0
        assert "_tier_index" in result.signals

    def test_tier_score_to_tier_simple(self):
        from model_plane.classifier.laya_classifier import LayaClassifier
        adapter = LayaClassifier(min_confidence=0.50)
        features = _features("what is 2+2?")

        mock_router = MagicMock()
        mock_router.predict.return_value = self._make_laya_result(
            task_label="simple_qa",
            task_score=0.92,
            tier_label="simple",
        )

        with patch("model_plane.classifier.laya_classifier._router", mock_router):
            result = _run(adapter.classify(features))

        assert result is not None
        assert result.complexity_tier == ComplexityTier.SIMPLE

    def test_tier_score_to_tier_reasoning(self):
        from model_plane.classifier.laya_classifier import LayaClassifier
        adapter = LayaClassifier(min_confidence=0.50)
        features = _features("prove the halting problem by reduction")

        mock_router = MagicMock()
        mock_router.predict.return_value = self._make_laya_result(
            task_label="mathematical_reasoning",
            task_score=0.91,
            tier_label="reasoning",
        )

        with patch("model_plane.classifier.laya_classifier._router", mock_router):
            result = _run(adapter.classify(features))

        assert result is not None
        assert result.complexity_tier == ComplexityTier.REASONING

    def test_returns_none_when_router_unavailable(self):
        from model_plane.classifier.laya_classifier import LayaClassifier
        adapter = LayaClassifier(min_confidence=0.60)
        features = _features("translate this to French")

        with patch("model_plane.classifier.laya_classifier._router", None):
            # _get_router() will also return None because laya is not installed
            with patch("model_plane.classifier.laya_classifier._get_router", return_value=None):
                result = _run(adapter.classify(features))

        assert result is None

    def test_returns_none_on_low_confidence(self):
        """Low task score → low confidence → adapter returns None when threshold is high."""
        from model_plane.classifier.laya_classifier import LayaClassifier
        adapter = LayaClassifier(min_confidence=0.95)  # very high threshold
        features = _features("summarize this document")

        mock_router = MagicMock()
        # task_score=0.60 → confidence clamped to 0.60, below 0.95 threshold
        mock_router.predict.return_value = self._make_laya_result(
            task_label="summarization",
            task_score=0.60,
            tier_label="medium",
        )

        with patch("model_plane.classifier.laya_classifier._router", mock_router):
            result = _run(adapter.classify(features))

        assert result is None

    def test_unknown_task_label_maps_to_unknown(self):
        from model_plane.classifier.laya_classifier import LayaClassifier
        adapter = LayaClassifier(min_confidence=0.50)
        features = _features("do something weird")

        mock_router = MagicMock()
        # "unknown" is the last entry in TASK_TYPE_LABELS and maps to TaskType.UNKNOWN
        mock_router.predict.return_value = self._make_laya_result(
            task_label="unknown",
            task_score=0.80,
            tier_label="medium",
        )

        with patch("model_plane.classifier.laya_classifier._router", mock_router):
            result = _run(adapter.classify(features))

        assert result is not None
        assert result.task_type == TaskType.UNKNOWN

    def test_inference_exception_returns_none(self):
        from model_plane.classifier.laya_classifier import LayaClassifier
        adapter = LayaClassifier(min_confidence=0.50)
        features = _features("write a poem")

        mock_router = MagicMock()
        mock_router.predict.side_effect = RuntimeError("laya internal error")

        with patch("model_plane.classifier.laya_classifier._router", mock_router):
            result = _run(adapter.classify(features))

        assert result is None


# ── classify() integration — adapter path ─────────────────────────────────────

class TestClassifyWithAdapter:
    """Ensure classifier.classify() correctly delegates to and overrides from adapters."""

    def test_regex_only_path_unchanged(self):
        """With CLASSIFIER_MODEL=regex (factory returns None), classify() is pure regex."""
        from model_plane.classifier.classifier import classify

        features = _features("debug my failing test")
        s = _make_factory_settings(classifier_model="regex", bert_classifier_enabled=False)
        with patch("model_plane.classifier.factory.settings", s):
            result = classify(features)

        assert result.task_type == TaskType.CODE_DEBUGGING
        assert result.signals.get("_source") == 0.0

    def test_adapter_result_overrides_regex_task_type(self):
        """When adapter fires with high confidence, task_type is replaced."""
        from model_plane.classifier.classifier import ClassificationResult, classify

        features = _features("debug my failing test")

        override = ClassificationResult(
            task_type=TaskType.TECHNICAL_REASONING,
            complexity_tier=ComplexityTier.REASONING,
            confidence=0.92,
            signals={"_source": 1.0},
        )

        mock_adapter = MagicMock()
        mock_adapter.classify = AsyncMock(return_value=override)

        # patch both the factory settings (so factory returns mock_adapter) and
        # the factory function itself in the classifier module's local import scope.
        with patch("model_plane.classifier.factory.get_classifier", return_value=mock_adapter):
            result = classify(features)

        assert result.task_type == TaskType.TECHNICAL_REASONING
        assert result.complexity_tier == ComplexityTier.REASONING
        assert result.signals.get("_source") == 1.0

    def test_adapter_exception_falls_back_to_regex(self):
        """If adapter raises, regex result is used unchanged."""
        from model_plane.classifier.classifier import classify

        features = _features("summarize this article")

        mock_adapter = MagicMock()
        mock_adapter.classify = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("model_plane.classifier.factory.get_classifier", return_value=mock_adapter):
            result = classify(features)

        # regex fires and returns a sensible result
        assert result.task_type == TaskType.SUMMARIZATION
        assert result.signals.get("_source") == 0.0


# ── _score_to_tier boundary tests ─────────────────────────────────────────────

class TestScoreToTier:
    """Verify all four threshold boundaries of _score_to_tier."""

    def test_score_0_is_simple(self):
        from model_plane.classifier.laya_classifier import _score_to_tier
        assert _score_to_tier(0.0) == ComplexityTier.SIMPLE

    def test_score_just_below_0_33_is_simple(self):
        from model_plane.classifier.laya_classifier import _score_to_tier
        assert _score_to_tier(0.329) == ComplexityTier.SIMPLE

    def test_score_0_33_is_medium(self):
        from model_plane.classifier.laya_classifier import _score_to_tier
        assert _score_to_tier(0.33) == ComplexityTier.MEDIUM

    def test_score_0_55_is_complex(self):
        from model_plane.classifier.laya_classifier import _score_to_tier
        assert _score_to_tier(0.55) == ComplexityTier.COMPLEX

    def test_score_0_78_is_reasoning(self):
        from model_plane.classifier.laya_classifier import _score_to_tier
        assert _score_to_tier(0.78) == ComplexityTier.REASONING

    def test_score_1_0_is_reasoning(self):
        from model_plane.classifier.laya_classifier import _score_to_tier
        assert _score_to_tier(1.0) == ComplexityTier.REASONING
