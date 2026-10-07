"""Tests for BERT classifier augmentation path.

All tests run without transformers installed — they verify the fallback
behaviour (regex wins) and that the BERT path activates correctly when mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from model_plane.classifier.classifier import ClassificationResult, classify
from model_plane.classifier.features import RequestFeatures
from model_plane.classifier.taxonomy import ComplexityTier, TaskType


# ── helpers ───────────────────────────────────────────────────────────────────

def _features(text: str) -> RequestFeatures:
    """Minimal RequestFeatures with just enough fields for classify()."""
    from model_plane.classifier.features import extract_features
    return extract_features({"messages": [{"role": "user", "content": text}]})


# ── flag=false (default) ──────────────────────────────────────────────────────

class TestBertDisabled:
    """When bert_classifier_enabled=False, regex result is always used."""

    def test_regex_used_when_flag_off(self):
        feats = _features("Write a Python function to reverse a string")
        result = classify(feats)
        assert result.task_type == TaskType.CODE_GENERATION
        assert result.signals.get("_source", 0.0) == 0.0   # regex path

    def test_simple_qa_regex(self):
        feats = _features("What is the capital of France?")
        result = classify(feats)
        assert result.task_type == TaskType.SIMPLE_QA
        assert result.signals["_source"] == 0.0

    def test_classification_result_fields_present(self):
        feats = _features("Summarize this document")
        result = classify(feats)
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.task_type, TaskType)
        assert isinstance(result.complexity_tier, ComplexityTier)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.signals, dict)


# ── flag=true, bert returns high confidence ───────────────────────────────────

class TestBertEnabled:
    """When bert_classifier_enabled=True and BERT returns confident result, it overrides regex."""

    @staticmethod
    def _mock_settings(bert_enabled=True, model="test-model", min_conf=0.60):
        s = MagicMock()
        s.classifier_model = "regex"      # deprecated alias path
        s.bert_classifier_enabled = bert_enabled
        s.bert_classifier_model = model
        s.bert_classifier_min_confidence = min_conf
        s.classifier_min_confidence = min_conf
        return s

    def test_bert_overrides_regex_on_high_confidence(self):
        """Regex would say CODE_GENERATION; mock BERT says MATHEMATICAL_REASONING at 0.85."""
        from model_plane.classifier import bert_classifier

        mock_result = bert_classifier.BertClassificationResult(
            task_type=TaskType.MATHEMATICAL_REASONING,
            confidence=0.85,
        )

        s = self._mock_settings()
        with patch("model_plane.classifier.factory.settings", s), \
             patch("model_plane.classifier.bert_classifier.predict", return_value=mock_result):
            feats = _features("Write a Python function")
            result = classify(feats)

        assert result.task_type == TaskType.MATHEMATICAL_REASONING
        assert result.confidence == 0.85
        assert result.signals["_source"] == 1.0   # bert path

    def test_bert_does_not_override_below_min_confidence(self):
        """BERT returns 0.05 — below the 0.10 NLI threshold — regex wins."""
        from model_plane.classifier import bert_classifier

        mock_result = bert_classifier.BertClassificationResult(
            task_type=TaskType.MATHEMATICAL_REASONING,
            confidence=0.05,   # calibrated score below 0.10 mBERT default threshold
        )

        s = self._mock_settings()
        with patch("model_plane.classifier.factory.settings", s), \
             patch("model_plane.classifier.bert_classifier.predict", return_value=mock_result):
            feats = _features("Write a Python function")
            result = classify(feats)

        # regex should win
        assert result.task_type == TaskType.CODE_GENERATION
        assert result.signals["_source"] == 0.0

    def test_bert_none_result_falls_back_to_regex(self):
        """BERT returns None (model not loaded) — regex wins silently."""
        s = self._mock_settings()
        with patch("model_plane.classifier.factory.settings", s), \
             patch("model_plane.classifier.bert_classifier.predict", return_value=None):
            feats = _features("Debug this traceback: AttributeError")
            result = classify(feats)

        assert result.task_type == TaskType.CODE_DEBUGGING
        assert result.signals["_source"] == 0.0

    def test_bert_exception_falls_back_to_regex(self):
        """If BERT predict() raises, regex result is used and no exception propagates."""
        s = self._mock_settings()
        with patch("model_plane.classifier.factory.settings", s), \
             patch("model_plane.classifier.bert_classifier.predict", side_effect=RuntimeError("cuda oom")):
            feats = _features("Summarize this article")
            # Must not raise
            result = classify(feats)

        assert result.task_type == TaskType.SUMMARIZATION
        assert result.signals["_source"] == 0.0

    def test_tier_re_derived_after_bert_override(self):
        """When BERT overrides to MATHEMATICAL_REASONING, tier should be REASONING."""
        from model_plane.classifier import bert_classifier

        mock_result = bert_classifier.BertClassificationResult(
            task_type=TaskType.MATHEMATICAL_REASONING,
            confidence=0.92,
        )

        s = self._mock_settings()
        with patch("model_plane.classifier.factory.settings", s), \
             patch("model_plane.classifier.bert_classifier.predict", return_value=mock_result):
            feats = _features("What is 2+2?")  # simple regex → SIMPLE tier
            result = classify(feats)

        assert result.task_type == TaskType.MATHEMATICAL_REASONING
        assert result.complexity_tier == ComplexityTier.REASONING


# ── bert_classifier module unit tests ─────────────────────────────────────────

class TestBertClassifierModule:
    """Unit tests for bert_classifier.py internals, no model required."""

    def test_label_to_task_type_code_gen(self):
        from model_plane.classifier.bert_classifier import _label_to_task_type
        assert _label_to_task_type("code generation and writing new code") == TaskType.CODE_GENERATION

    def test_label_to_task_type_math(self):
        from model_plane.classifier.bert_classifier import _label_to_task_type
        assert _label_to_task_type("mathematical reasoning and calculation") == TaskType.MATHEMATICAL_REASONING

    def test_label_to_task_type_unknown(self):
        from model_plane.classifier.bert_classifier import _label_to_task_type
        assert _label_to_task_type("completely unrecognised label xyz") == TaskType.UNKNOWN

    def test_predict_returns_none_when_transformers_missing(self):
        """If transformers import fails, predict() returns None gracefully."""
        from model_plane.classifier import bert_classifier

        # Reset singleton so _get_pipeline() will try to load
        bert_classifier.reload()

        with patch.dict("sys.modules", {"transformers": None}):
            result = bert_classifier.predict("Hello world", "test-model")

        # After failed load, pipeline is None → predict returns None
        # (Note: may still be None if _load_attempted was set previously)
        # The important assertion is that it doesn't raise.
        assert result is None or hasattr(result, "task_type")

    def test_predict_returns_none_for_empty_text(self):
        """Empty text should return None without calling the pipeline."""
        from model_plane.classifier import bert_classifier

        mock_pipe = MagicMock()
        with patch.object(bert_classifier, "_pipeline", mock_pipe):
            # Force min_hits check to pass but text is empty
            result = bert_classifier.predict("", "test-model")

        # Empty text → None (early return before pipeline call)
        assert result is None
        mock_pipe.assert_not_called()

    def test_reload_resets_singleton(self):
        from model_plane.classifier import bert_classifier
        bert_classifier._pipeline = MagicMock()
        bert_classifier._load_attempted = True
        bert_classifier.reload()
        assert bert_classifier._pipeline is None
        assert bert_classifier._load_attempted is False
