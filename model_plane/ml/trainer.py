"""ML model trainer — offline training script for the routing recommender.

Two models are trained from training_data.csv:

  1. Task classifier (task_type → JSON weights, like strata's classifier_ml.json)
     Trained on 14 numeric feature dimensions, outputs 14-15 TaskType classes.
     Stored as JSON — no sklearn at inference time (MLTaskClassifier loads it).

     Run:
         python -m model_plane.ml.trainer --mode task \\
             --output models/classifier_ml.json

  2. Tier classifier (complexity_tier → joblib, like strata's router_ml.json)
     Same 14 features, outputs 4 ComplexityTier classes.
     Stored as joblib (sklearn RandomForest/GradientBoosting).

     Run:
         python -m model_plane.ml.trainer --mode tier \\
             --output models/router_ml_tier_v1.joblib

  Legacy deployment model (rollback):
         python -m model_plane.ml.trainer --mode tier \\
             --label selected_deployment --output models/router_ml.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

# Valid tier label strings — must match ComplexityTier enum values.
TIER_LABELS = ["simple", "medium", "complex", "reasoning"]

# Feature column order — must match features.py RequestFeatures.as_vector() and
# strata's router_ml.json normMeans/normStds dimension order.
FEATURE_COLUMNS: list[str] = [
    "reasoning_markers",
    "code_presence",
    "simple_indicators",
    "multi_step_patterns",
    "technical_terms",
    "token_count_signal",
    "creative_markers",
    "question_complexity",
    "constraint_count",
    "imperative_verbs",
    "output_format",
    "domain_specificity",
    "reference_complexity",
    "negation_complexity",
]

# Extra signal columns derived from the query text via the regex classifier.
# When a training row has a non-empty "query" column, these are computed and
# appended to the feature vector (dims 14–27). This gives the model the same
# boolean signals strata's ml-classifier.ts uses, dramatically improving
# accuracy on short text examples where the 14 numeric features are near-zero.
QUERY_SIGNAL_COLUMNS: list[str] = [
    "sig_code_debug", "sig_code_gen", "sig_code_edit", "sig_repo_search",
    "sig_math", "sig_tech_reason", "sig_extract", "sig_summarize",
    "sig_translate", "sig_creative", "sig_plan", "sig_tool",
    "sig_long_ctx", "sig_simple",
]


def load_dataset(
    csv_path: Path,
    label_column: str = "complexity_tier",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load training rows from *csv_path*.

    Args:
        csv_path:      Path to the training CSV file.
        label_column:  Name of the column to use as the target label.
                       Use ``"complexity_tier"`` (or its CSV alias ``"tier"``) for
                       the tier model, ``"task_type"`` for the task classifier,
                       or ``"selected_deployment"`` for the legacy deployment model.

    Returns:
        (X, y, class_names) where class_names is the ordered list of label strings.
    """
    class_names: list[str] = []
    label_to_idx: dict[str, int] = {}

    # "complexity_tier" is the canonical name in the plan; "tier" is the actual
    # column header in the training CSV.  Accept either.
    _is_tier_label = label_column in ("complexity_tier", "tier")
    _csv_col = "tier" if _is_tier_label else label_column

    # For the tier label, pre-populate with a canonical order so the saved model's
    # class indices are stable across training runs (SIMPLE=0, MEDIUM=1, …).
    if _is_tier_label:
        for t in TIER_LABELS:
            label_to_idx[t] = len(class_names)
            class_names.append(t)

    # Lazy import of the regex classifier signals extractor.
    # We extract 14 boolean signal features from the query text (when present)
    # and append them as dims 14–27 to give the model the same discriminative
    # signals that strata's ml-classifier.ts uses.
    def _query_signals(text: str) -> list[float]:
        """Return 14 regex signal floats from query text (0.0 or 1.0 each)."""
        import re as _re
        from model_plane.classifier.classifier import (
            _CODE_DEBUG, _CODE_GEN, _CODE_EDIT, _REPO_SEARCH,
            _MATH, _TECH_REASON, _EXTRACT, _SUMMARIZE,
            _TRANSLATE, _CREATIVE, _PLAN, _TOOL, _LONG_CTX,
        )
        from model_plane.classifier.features import _RE_SIMPLE
        t = text
        return [
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
        ]

    X_rows, y_rows = [], []
    has_query_col: bool | None = None  # detected from first row

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row.get(_csv_col, "").strip()
            # final_success may be "1.0", "1", "True" — treat any truthy non-zero string as success
            success_val = row.get("final_success", "")
            if not label or not success_val or success_val in ("0", "0.0", "False", "false"):
                continue

            # Detect whether the CSV has a query column (once, on first row)
            if has_query_col is None:
                has_query_col = "query" in row

            features = [float(row.get(col, 0)) for col in FEATURE_COLUMNS]

            # Append query-derived boolean signals when a query column is present
            if has_query_col:
                query_text = row.get("query", "").strip()
                features.extend(_query_signals(query_text) if query_text else [0.0] * len(QUERY_SIGNAL_COLUMNS))

            if label not in label_to_idx:
                label_to_idx[label] = len(class_names)
                class_names.append(label)
            X_rows.append(features)
            y_rows.append(label_to_idx[label])

    return np.array(X_rows), np.array(y_rows), class_names


# ── Shared JSON logistic regression trainer ───────────────────────────────────

def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _train_json_logreg(
    rows: list[dict],
    label_column: str,
    class_names: list[str],
    model_name: str,
    output_path: Path,
    epochs: int = 300,
    lr: float = 0.1,
    reg: float = 0.01,
) -> dict:
    """Shared SGD logistic regression trainer used by both task and tier classifiers.

    Produces the same portable JSON format (strata-compatible):
      classNames / weights / biases / normMeans / normStds /
      trainedOn / trainingSamples / accuracy / featureColumns

    Args:
        rows:         Pre-filtered list of CSV row dicts.
        label_column: Which key in each row is the target label.
        class_names:  Ordered list of class labels (stable ordering matters).
        model_name:   Human label used in the print summary.
        output_path:  Destination .json path.
    """
    n_features = len(FEATURE_COLUMNS)
    label_index = {lbl: i for i, lbl in enumerate(class_names)}
    n_classes = len(class_names)

    X: list[list[float]] = [[float(r.get(col, 0)) for col in FEATURE_COLUMNS] for r in rows]
    y: list[int] = [label_index[r[label_column]] for r in rows]
    n = len(X)

    # Z-score normalisation — mirrors strata normMeans / normStds.
    means = [sum(X[i][j] for i in range(n)) / n for j in range(n_features)]
    stds = [
        math.sqrt(sum((X[i][j] - means[j]) ** 2 for i in range(n)) / max(n - 1, 1))
        for j in range(n_features)
    ]
    X_norm = [
        [(X[i][j] - means[j]) / (stds[j] or 1.0) for j in range(n_features)]
        for i in range(n)
    ]

    W = [[0.0] * n_features for _ in range(n_classes)]
    b = [0.0] * n_classes

    for _epoch in range(epochs):
        for i in range(n):
            xi, yi = X_norm[i], y[i]
            logits = [sum(W[c][j] * xi[j] for j in range(n_features)) + b[c] for c in range(n_classes)]
            probs = _softmax(logits)
            for c in range(n_classes):
                delta = probs[c] - (1.0 if c == yi else 0.0)
                for j in range(n_features):
                    W[c][j] -= lr * (delta * xi[j] + reg * W[c][j])
                b[c] -= lr * delta

    correct = sum(
        1 for i in range(n)
        if max(range(n_classes), key=lambda c: sum(W[c][j] * X_norm[i][j] for j in range(n_features)) + b[c]) == y[i]
    )
    accuracy = correct / n

    model_json = {
        "classNames": class_names,
        "weights": W,
        "biases": b,
        "normMeans": means,
        "normStds": stds,
        "featureColumns": FEATURE_COLUMNS,   # self-describing — inference can verify order
        "trainedOn": datetime.now(timezone.utc).isoformat(),
        "trainingSamples": n,
        "accuracy": accuracy,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(model_json, fh, indent=2)

    print(f"{model_name} saved → {output_path}  (accuracy={accuracy:.3f}, classes={n_classes}, samples={n})")
    return {"accuracy": accuracy, "n_samples": n, "n_classes": n_classes}


# ── Task classifier (task_type) — primary: sklearn joblib, backup: JSON ───────

def train_task_classifier(
    data_path: Path,
    output_path: Path,
    model_type: str = "gradient_boost",
) -> dict:
    """Train a task-type classifier and save as a sklearn joblib artefact.

    Primary format — matches the tier classifier pattern so both models are
    consistent (sklearn GBT/RF, joblib storage, feature importances in UI).

    Input CSV column: ``task_type``
    Default output:   models/classifier_task.joblib

    The JSON logistic-regression backup (``train_task_classifier_json``) is still
    available for portable/sklearn-free environments.
    """
    return train(data_path, output_path, model_type=model_type, label_column="task_type")


def train_task_classifier_json(data_path: Path, output_path: Path) -> dict:
    """Backup: portable JSON logistic-regression task-type classifier.

    No sklearn at inference. Use when joblib is unavailable or for inspection.
    Output: models/classifier_ml.json (strata-compatible format)
    """
    rows: list[dict] = []
    with open(data_path, newline="") as fh:
        for row in csv.DictReader(fh):
            sv = row.get("final_success", "")
            if row.get("task_type") and sv not in ("0", "0.0", "False", "false"):
                rows.append(row)
    if len(rows) < 20:
        raise ValueError(f"Insufficient training data: {len(rows)} rows (need ≥ 20)")
    class_names = sorted({r["task_type"] for r in rows})
    return _train_json_logreg(rows, "task_type", class_names, "Task classifier (JSON backup)", output_path)


# ── Tier classifier (complexity_tier, JSON) ───────────────────────────────────

def train_tier_classifier_json(data_path: Path, output_path: Path) -> dict:
    """Train a logistic-regression complexity-tier classifier and save as JSON.

    Same portable format as the task classifier — no sklearn/joblib dependency
    at inference time.

    Input CSV column: ``tier``
    Output: models/router_ml_tier_v1.json

    Why JSON instead of joblib for this path?
      RandomForest/GBT (train() below) gives slightly higher accuracy but requires
      sklearn at inference.  This JSON version is deployable anywhere and is the
      default when CLASSIFIER_MODEL=ml_task or in lightweight environments.
      Both formats are supported; the joblib path remains for sklearn environments.
    """
    rows: list[dict] = []
    with open(data_path, newline="") as fh:
        for row in csv.DictReader(fh):
            sv = row.get("final_success", "")
            tier = row.get("tier", "").strip()
            if tier in TIER_LABELS and sv not in ("0", "0.0", "False", "false"):
                rows.append(row)
    if len(rows) < 20:
        raise ValueError(f"Insufficient training data: {len(rows)} rows (need ≥ 20)")
    # Canonical tier order: stable class indices across retrains.
    return _train_json_logreg(rows, "tier", TIER_LABELS, "Tier classifier (JSON)", output_path,
                               epochs=400, lr=0.15)


def train(
    data_path: Path,
    output_path: Path,
    model_type: str = "random_forest",
    label_column: str = "complexity_tier",
) -> dict:
    """Train a classifier and save it as a joblib artefact.

    Args:
        data_path:    Path to the training CSV.
        output_path:  Destination for the joblib file.
        model_type:   ``"random_forest"`` (default) or ``"gradient_boost"``.
        label_column: Column to use as the target label.  Default is
                      ``"complexity_tier"`` (4 classes); use
                      ``"selected_deployment"`` for the legacy model.

    Returns:
        ``classification_report`` output dict.
    """
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    X, y, class_names = load_dataset(data_path, label_column=label_column)
    if len(X) < 20:
        raise ValueError(f"Insufficient training data: {len(X)} rows (need >= 20)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == "gradient_boost":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)

    report = classification_report(
        y_test, clf.predict(X_test), target_names=class_names, output_dict=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "class_names": class_names,
            "label_column": label_column,
        },
        output_path,
    )
    print(f"Model saved to {output_path}")
    print(classification_report(y_test, clf.predict(X_test), target_names=class_names))

    return report


# Public aliases used by retrain endpoints.
train_model = train                                    # sklearn joblib (tier classifier)
train_task_model = train_task_classifier               # sklearn joblib (task classifier)
train_task_json_model = train_task_classifier_json     # JSON logreg backup (task)
train_tier_json_model = train_tier_classifier_json     # JSON logreg backup (tier)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train routing ML models")
    parser.add_argument(
        "--mode",
        choices=["tier", "tier-json", "task", "task-json"],
        default="tier",
        help=(
            "tier      → sklearn GBT/RF tier classifier (joblib)\n"
            "tier-json → portable JSON LR tier classifier\n"
            "task      → sklearn GBT/RF task classifier (joblib)\n"
            "task-json → portable JSON LR task classifier (backup)"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("model_plane/data/training_data.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults vary by mode.",
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "gradient_boost"],
        default="gradient_boost",
        help="sklearn estimator for --mode tier or task",
    )
    args = parser.parse_args()

    if args.mode == "task":
        out = args.output or Path("models/classifier_task.joblib")
        report = train_task_classifier(args.data, out, args.model)
    elif args.mode == "task-json":
        out = args.output or Path("models/classifier_ml.json")
        report = train_task_classifier_json(args.data, out)
    elif args.mode == "tier-json":
        out = args.output or Path("models/router_ml_tier_v1.json")
        report = train_tier_classifier_json(args.data, out)
    else:
        out = args.output or Path("models/router_ml_tier_v1.joblib")
        report = train(args.data, out, args.model, label_column="complexity_tier")

    print(json.dumps({k: v for k, v in report.items() if k in ("accuracy", "n_samples", "n_classes")}, indent=2))


if __name__ == "__main__":
    main()
