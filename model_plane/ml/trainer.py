"""ML model trainer — offline training script for the routing recommender.

Run:
    python -m model_plane.ml.trainer --data data/training_data.csv --output models/router_ml.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from model_plane.classifier.taxonomy import ComplexityTier

    tier_to_idx = {t.value: i for i, t in enumerate([
        ComplexityTier.SIMPLE,
        ComplexityTier.MEDIUM,
        ComplexityTier.COMPLEX,
        ComplexityTier.REASONING,
    ])}

    X_rows, y_rows = [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Skip rows without final_success or where routing failed
            if not row.get("tier") or not row.get("final_success"):
                continue
            features = [
                float(row.get("reasoning_markers", 0)),
                float(row.get("code_presence", 0)),
                float(row.get("simple_indicators", 0)),
                float(row.get("multi_step_patterns", 0)),
                float(row.get("technical_terms", 0)),
                float(row.get("token_count_signal", 0)),
                float(row.get("creative_markers", 0)),
                float(row.get("question_complexity", 0)),
                float(row.get("constraint_count", 0)),
                float(row.get("imperative_verbs", 0)),
                float(row.get("output_format", 0)),
                float(row.get("domain_specificity", 0)),
                float(row.get("reference_complexity", 0)),
                float(row.get("negation_complexity", 0)),
            ]
            tier = row["tier"]
            if tier not in tier_to_idx:
                continue
            X_rows.append(features)
            y_rows.append(tier_to_idx[tier])

    return np.array(X_rows), np.array(y_rows), list(tier_to_idx.keys())


def train(data_path: Path, output_path: Path, model_type: str = "random_forest") -> dict:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    X, y, class_names = load_dataset(data_path)
    if len(X) < 20:
        raise ValueError(f"Insufficient training data: {len(X)} rows (need >= 20)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if model_type == "gradient_boost":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)

    report = classification_report(y_test, clf.predict(X_test), target_names=class_names, output_dict=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"Model saved to {output_path}")
    print(classification_report(y_test, clf.predict(X_test), target_names=class_names))

    return report


# Public alias used by the retrain endpoint
train_model = train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train routing ML model")
    parser.add_argument("--data", type=Path, default=Path("model_plane/data/training_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/router_ml.joblib"))
    parser.add_argument("--model", choices=["random_forest", "gradient_boost"], default="gradient_boost")
    args = parser.parse_args()

    report = train(args.data, args.output, args.model)
    print(json.dumps({k: v for k, v in report.items() if k in ("accuracy", "log_loss", "mean_max_proba")}, indent=2))


if __name__ == "__main__":
    main()
