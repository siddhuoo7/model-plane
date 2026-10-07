"""Unit tests for Sub-Task 4.3 — ML Task Classifier."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from model_plane.classifier.ml_task_classifier import (
    MLTaskClassifier,
    SIGNAL_KEYS,
    N_FEATURES,
    _softmax,
    _dot,
    train_and_save,
)


# ── helper builders ───────────────────────────────────────────────────────────

def _make_model(labels=("code_generation", "question_answering"), n_features=N_FEATURES):
    """Build a minimal valid model dict."""
    n = len(labels)
    return {
        "labels": list(labels),
        "weights": [[0.0] * n_features for _ in range(n)],
        "biases": [0.0] * n,
        "mean": [0.0] * n_features,
        "std": [1.0] * n_features,
        "signal_keys": SIGNAL_KEYS,
        "n_features": n_features,
        "n_samples": 100,
        "n_classes": n,
        "training_accuracy": 0.9,
    }


def _saved_model(tmp_path, model):
    p = tmp_path / "model.json"
    p.write_text(json.dumps(model))
    return p


# ── _softmax ──────────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_sums_to_one(self):
        probs = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_monotone(self):
        probs = _softmax([1.0, 2.0, 3.0])
        assert probs[0] < probs[1] < probs[2]

    def test_uniform_for_equal_logits(self):
        probs = _softmax([1.0, 1.0, 1.0])
        for p in probs:
            assert abs(p - 1 / 3) < 1e-9


# ── _dot ──────────────────────────────────────────────────────────────────────

class TestDot:
    def test_basic(self):
        assert _dot([1.0, 2.0], [3.0, 4.0]) == pytest.approx(11.0)

    def test_zero_vector(self):
        assert _dot([0.0, 0.0], [5.0, 6.0]) == 0.0


# ── MLTaskClassifier ──────────────────────────────────────────────────────────

class TestMLTaskClassifier:
    def test_not_available_when_model_missing(self, tmp_path):
        clf = MLTaskClassifier(tmp_path / "nonexistent.json")
        assert not clf.is_available

    def test_available_when_model_exists(self, tmp_path):
        model = _make_model()
        p = _saved_model(tmp_path, model)
        clf = MLTaskClassifier(p)
        assert clf.is_available

    def test_vectorise_correct_length(self, tmp_path):
        model = _make_model()
        p = _saved_model(tmp_path, model)
        clf = MLTaskClassifier(p)
        signals = {k: 0.0 for k in SIGNAL_KEYS}
        signals["code_gen"] = 1.0
        vec = clf._vectorise(signals)
        assert len(vec) == N_FEATURES
        assert vec[SIGNAL_KEYS.index("code_gen")] == 1.0

    def test_vectorise_missing_keys_default_zero(self, tmp_path):
        model = _make_model()
        p = _saved_model(tmp_path, model)
        clf = MLTaskClassifier(p)
        vec = clf._vectorise({})
        assert all(v == 0.0 for v in vec)

    @pytest.mark.asyncio
    async def test_classify_returns_result_for_code_signals(self, tmp_path):
        """With a biased model, code_gen signal should return code_generation."""
        labels = ["code_generation", "question_answering"]
        n = len(labels)
        # Make the first class (code_generation) very likely by biasing its logit up
        model = _make_model(labels)
        model["biases"] = [10.0, -10.0]  # strongly biased to first class
        p = _saved_model(tmp_path, model)
        clf = MLTaskClassifier(p)

        from unittest.mock import patch, MagicMock
        from model_plane.classifier.features import RequestFeatures
        features = RequestFeatures(
            last_user_text="write a Python function",
            full_text="write a Python function",
            code_presence=0.5,
            has_tools=0.0,
            simple_indicators=0.0,
            token_count_signal=0.0,
        )

        result = await clf.classify(features)
        assert result is not None
        assert result.task_type.value == "code_generation"
        assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_classify_returns_none_when_not_available(self, tmp_path):
        clf = MLTaskClassifier(tmp_path / "nonexistent.json")
        from unittest.mock import MagicMock
        result = await clf.classify(MagicMock())
        assert result is None


# ── train_and_save ────────────────────────────────────────────────────────────

class TestTrainAndSave:
    def _make_csv(self, tmp_path, n_per_class=30):
        """Generate a synthetic CSV with perfectly separated classes."""
        import csv
        labels = ["code_generation", "question_answering", "summarization"]
        rows = []
        for i, label in enumerate(labels):
            for _ in range(n_per_class):
                row = {k: "0.0" for k in SIGNAL_KEYS}
                # Each class has one perfectly set signal
                if i == 0:
                    row["code_gen"] = "1.0"
                elif i == 1:
                    row["tool"] = "1.0"
                else:
                    row["summarize"] = "1.0"
                row["task_type"] = label
                rows.append(row)

        p = tmp_path / "train.csv"
        with open(p, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SIGNAL_KEYS + ["task_type"])
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_produces_valid_json(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "model.json"
        meta = train_and_save(csv_path, out)
        assert out.exists()
        model = json.loads(out.read_text())
        assert "labels" in model
        assert "weights" in model
        assert "biases" in model

    def test_returns_accuracy_metadata(self, tmp_path):
        csv_path = self._make_csv(tmp_path)
        out = tmp_path / "model.json"
        meta = train_and_save(csv_path, out)
        assert "accuracy" in meta
        assert 0.0 <= meta["accuracy"] <= 1.0
        assert meta["n_classes"] == 3

    def test_high_accuracy_on_separable_data(self, tmp_path):
        csv_path = self._make_csv(tmp_path, n_per_class=60)
        out = tmp_path / "model.json"
        meta = train_and_save(csv_path, out)
        # Linearly separable data should reach near-perfect accuracy
        assert meta["accuracy"] >= 0.90

    def test_empty_csv_raises(self, tmp_path):
        import csv
        p = tmp_path / "empty.csv"
        with open(p, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SIGNAL_KEYS + ["task_type"])
            writer.writeheader()
        with pytest.raises(ValueError, match="Empty"):
            train_and_save(p, tmp_path / "out.json")


# ── factory integration ───────────────────────────────────────────────────────

class TestFactoryMlTask:
    def test_factory_returns_ml_task_classifier_when_flag_set(self, tmp_path):
        from unittest.mock import patch, MagicMock
        mock_settings = MagicMock()
        mock_settings.classifier_model = "ml_task"
        mock_settings.bert_classifier_enabled = False
        mock_settings.ml_task_classifier_path = str(tmp_path / "nonexistent.json")

        with patch("model_plane.classifier.factory.settings", mock_settings):
            from model_plane.classifier.factory import get_classifier
            clf = get_classifier()
        assert clf is not None
        assert isinstance(clf, MLTaskClassifier)

    def test_factory_returns_none_for_regex(self):
        from unittest.mock import patch, MagicMock
        mock_settings = MagicMock()
        mock_settings.classifier_model = "regex"
        mock_settings.bert_classifier_enabled = False

        with patch("model_plane.classifier.factory.settings", mock_settings):
            from model_plane.classifier.factory import get_classifier
            clf = get_classifier()
        assert clf is None
