"""ML Task Classifier — Sub-Task 4.3.

A lightweight logistic-regression classifier that maps the 14 regex signal
booleans produced by ``_regex_classify`` to a ``TaskType``.

Architecture
------------
- Input: 14-dimensional binary vector (one entry per regex signal key).
- Output: TaskType (15 classes).
- Model: multi-class logistic regression trained on ``training_data.csv``.
- Storage: JSON file (weights matrix + bias + label list + z-score stats).
  No joblib / pickle — fully portable, no sklearn at inference time.

Feature-flag
------------
``MODEL_PLANE_ML_TASK_CLASSIFIER_ENABLED=true`` (or
``ML_TASK_CLASSIFIER_ENABLED=true``) enables this classifier in the factory.
When enabled it runs after the regex baseline and overrides ``task_type`` when
its confidence exceeds ``settings.classifier_min_confidence``.

Training
--------
Call ``train_and_save(csv_path, output_path)`` once to produce the JSON model.
The CSV must have columns matching the 14 signal keys plus a ``task_type`` label
column.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from model_plane.classifier.base import ClassifierAdapter
from model_plane.classifier.taxonomy import ComplexityTier, TaskType

if TYPE_CHECKING:
    from model_plane.classifier.features import RequestFeatures
    from model_plane.classifier.classifier import ClassificationResult

# Fixed signal key order — must match the order used during training.
SIGNAL_KEYS: list[str] = [
    "code_debug",
    "code_gen",
    "code_edit",
    "repo_search",
    "code_presence",
    "math",
    "tech_reason",
    "extract",
    "summarize",
    "translate",
    "creative",
    "plan",
    "tool",
    "long_ctx",
]

N_FEATURES = len(SIGNAL_KEYS)


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class MLTaskClassifier(ClassifierAdapter):
    """Logistic-regression task classifier loaded from a JSON model file.

    Implements ``ClassifierAdapter`` so it can be registered in ``factory.py``.
    The weights are stored as a plain JSON dict — no sklearn at inference time.
    """

    name = "ml_task"

    def __init__(self, model_path: str | Path) -> None:
        self._joblib_model: Any | None = None   # sklearn estimator
        self._joblib_class_names: list[str] = []
        self._json_model: dict | None = None    # JSON logistic-regression backup
        self._path = Path(model_path)
        self._load()

    def _load(self) -> None:
        p = self._path
        suffix = p.suffix.lower()
        if suffix in (".joblib", ".pkl"):
            self._load_joblib(p)
        elif suffix == ".json":
            self._load_json(p)
        else:
            # Auto-detect: try joblib first, fall back to JSON
            self._load_joblib(p) or self._load_json(p)

    def _load_joblib(self, p: Path) -> bool:
        if not p.exists():
            return False
        try:
            import joblib as _joblib
            artefact = _joblib.load(p)
            if isinstance(artefact, dict):
                self._joblib_model = artefact.get("model")
                self._joblib_class_names = list(artefact.get("class_names", []))
            else:
                self._joblib_model = artefact
                self._joblib_class_names = list(getattr(self._joblib_model, "classes_", []))
            return True
        except Exception:
            return False

    def _load_json(self, p: Path) -> bool:
        if not p.exists():
            return False
        try:
            with open(p) as fh:
                self._json_model = json.load(fh)
            return True
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._joblib_model is not None or self._json_model is not None

    def _vectorise_from_features(self, features: "RequestFeatures") -> list[float]:
        """Build the feature vector from RequestFeatures.

        Produces 14 dims (numeric features) when the model was trained without
        a query column, or 28 dims (14 numeric + 14 query regex signals) when
        the model was trained with the query column present in the CSV.
        The dimensionality is inferred from the stored joblib model's n_features_in_.
        """
        from model_plane.ml.trainer import FEATURE_COLUMNS, QUERY_SIGNAL_COLUMNS
        feature_map = {
            "reasoning_markers": features.reasoning_markers,
            "code_presence": features.code_presence,
            "simple_indicators": features.simple_indicators,
            "multi_step_patterns": features.multi_step_patterns,
            "technical_terms": features.technical_terms,
            "token_count_signal": features.token_count_signal,
            "creative_markers": features.creative_markers,
            "question_complexity": features.question_complexity,
            "constraint_count": features.constraint_count,
            "imperative_verbs": features.imperative_verbs,
            "output_format": features.output_format,
            "domain_specificity": features.domain_specificity,
            "reference_complexity": features.reference_complexity,
            "negation_complexity": features.negation_complexity,
        }
        vec = [float(feature_map.get(col, 0.0)) for col in FEATURE_COLUMNS]

        # Check if the model expects 28 dims (trained with query signals)
        expected_dims = None
        if self._joblib_model is not None and hasattr(self._joblib_model, "n_features_in_"):
            expected_dims = int(self._joblib_model.n_features_in_)
        if expected_dims is not None and expected_dims > len(FEATURE_COLUMNS):
            # Append 14 query-derived regex boolean signals
            from model_plane.classifier.classifier import (
                _CODE_DEBUG, _CODE_GEN, _CODE_EDIT, _REPO_SEARCH,
                _MATH, _TECH_REASON, _EXTRACT, _SUMMARIZE,
                _TRANSLATE, _CREATIVE, _PLAN, _TOOL, _LONG_CTX,
            )
            from model_plane.classifier.features import _RE_SIMPLE
            t = features.last_user_text
            vec.extend([
                1.0 if _CODE_DEBUG.search(t) else 0.0,
                1.0 if _CODE_GEN.search(t)   else 0.0,
                1.0 if _CODE_EDIT.search(t)  else 0.0,
                1.0 if _REPO_SEARCH.search(t) else 0.0,
                1.0 if _MATH.search(t)       else 0.0,
                1.0 if _TECH_REASON.search(t) else 0.0,
                1.0 if _EXTRACT.search(t)    else 0.0,
                1.0 if _SUMMARIZE.search(t)  else 0.0,
                1.0 if _TRANSLATE.search(t)  else 0.0,
                1.0 if _CREATIVE.search(t)   else 0.0,
                1.0 if _PLAN.search(t)       else 0.0,
                1.0 if _TOOL.search(t)       else 0.0,
                1.0 if _LONG_CTX.search(t)   else 0.0,
                1.0 if _RE_SIMPLE.search(t)  else 0.0,
            ])
        return vec

    def _vectorise(self, signals: dict[str, float]) -> list[float]:
        """Extract the fixed-order feature vector from a regex signals dict (legacy)."""
        return [float(signals.get(k, 0.0)) for k in SIGNAL_KEYS]

    def _predict_proba_joblib(self, x: list[float]) -> tuple[list[str], list[float]]:
        """Run sklearn predict_proba and return (labels, probs)."""
        import numpy as np
        assert self._joblib_model is not None
        arr = np.array([x])
        probs = self._joblib_model.predict_proba(arr)[0].tolist()
        # class_names from artefact dict; fallback to model's own classes_
        labels = self._joblib_class_names or [str(c) for c in self._joblib_model.classes_]
        return labels, probs

    def _predict_proba_json(self, x: list[float]) -> tuple[list[str], list[float]]:
        """Run JSON logistic-regression inference."""
        assert self._json_model is not None
        mean = self._json_model.get("mean") or self._json_model.get("normMeans", [0.0] * N_FEATURES)
        std  = self._json_model.get("std")  or self._json_model.get("normStds",  [1.0] * N_FEATURES)
        x_norm = [(xi - m) / (s or 1.0) for xi, m, s in zip(x, mean, std)]
        labels: list[str] = self._json_model.get("labels") or self._json_model.get("classNames", [])
        weights: list[list[float]] = self._json_model["weights"]
        biases: list[float] = self._json_model["biases"]
        logits = [_dot(weights[i], x_norm) + biases[i] for i in range(len(labels))]
        probs = _softmax(logits)
        return labels, probs

    def _predict_proba(self, x: list[float]) -> tuple[list[str], list[float]]:
        """Return (label_list, probability_list) using joblib if available, else JSON."""
        if self._joblib_model is not None:
            return self._predict_proba_joblib(x)
        return self._predict_proba_json(x)

    async def classify(self, features: "RequestFeatures") -> "ClassificationResult | None":
        """Classify a request using the ML task classifier.

        Returns ``None`` if the model is not loaded or confidence is too low
        (caller falls back to the next classifier in the chain).
        """
        if not self.is_available:
            return None

        from model_plane.classifier.classifier import _regex_classify, ClassificationResult

        # Run regex baseline for tier + signals (always needed for fallback tier)
        _, tier, _, signals = _regex_classify(features)

        # Joblib model: use 14-dim feature vector directly from RequestFeatures.
        # JSON model: also uses 14-dim features (not the 14 signal-key booleans).
        x = self._vectorise_from_features(features)

        labels, probs = self._predict_proba(x)
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_label = labels[best_idx]
        confidence = probs[best_idx]

        try:
            task = TaskType(best_label)
        except ValueError:
            return None  # unknown label

        return ClassificationResult(
            task_type=task,
            complexity_tier=tier,
            confidence=confidence,
            signals={**signals, "_source": 1.0},
        )


# ── Training helper ───────────────────────────────────────────────────────────

def train_and_save(csv_path: str | Path, output_path: str | Path) -> dict:
    """Train a logistic regression classifier and save weights as JSON.

    Args:
        csv_path: Path to training CSV. Must contain columns for each key in
            ``SIGNAL_KEYS`` plus a ``task_type`` column.
        output_path: Destination ``.json`` path for the model weights.

    Returns:
        Dict with training metadata (accuracy, n_samples, n_classes).
    """
    import csv as csv_mod

    rows: list[dict] = []
    with open(csv_path) as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Empty training CSV: {csv_path}")

    # Build X and y
    labels_set: list[str] = sorted({r["task_type"] for r in rows if r.get("task_type")})
    label_index = {lbl: i for i, lbl in enumerate(labels_set)}
    n_classes = len(labels_set)

    X: list[list[float]] = []
    y: list[int] = []
    for row in rows:
        if not row.get("task_type"):
            continue
        x = [float(row.get(k, 0.0)) for k in SIGNAL_KEYS]
        X.append(x)
        y.append(label_index[row["task_type"]])

    n = len(X)
    if n == 0:
        raise ValueError("No valid rows after filtering.")

    # Z-score normalise
    mean = [sum(X[i][j] for i in range(n)) / n for j in range(N_FEATURES)]
    std = [
        math.sqrt(sum((X[i][j] - mean[j]) ** 2 for i in range(n)) / max(n - 1, 1))
        for j in range(N_FEATURES)
    ]
    X_norm = [[(X[i][j] - mean[j]) / (std[j] or 1.0) for j in range(N_FEATURES)] for i in range(n)]

    # Mini-batch SGD logistic regression (pure Python, no sklearn)
    W = [[0.0] * N_FEATURES for _ in range(n_classes)]
    b = [0.0] * n_classes
    lr, epochs, reg = 0.1, 200, 0.01

    for _epoch in range(epochs):
        for i in range(n):
            xi, yi = X_norm[i], y[i]
            logits = [_dot(W[c], xi) + b[c] for c in range(n_classes)]
            probs = _softmax(logits)
            for c in range(n_classes):
                delta = probs[c] - (1.0 if c == yi else 0.0)
                for j in range(N_FEATURES):
                    W[c][j] -= lr * (delta * xi[j] + reg * W[c][j])
                b[c] -= lr * delta

    # Evaluate training accuracy
    correct = 0
    for i in range(n):
        logits = [_dot(W[c], X_norm[i]) + b[c] for c in range(n_classes)]
        pred = max(range(n_classes), key=lambda c: logits[c])
        if pred == y[i]:
            correct += 1
    accuracy = correct / n

    model = {
        "labels": labels_set,
        "weights": W,
        "biases": b,
        "mean": mean,
        "std": std,
        "signal_keys": SIGNAL_KEYS,
        "n_features": N_FEATURES,
        "n_samples": n,
        "n_classes": n_classes,
        "training_accuracy": accuracy,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(model, fh, indent=2)

    return {"accuracy": accuracy, "n_samples": n, "n_classes": n_classes}
