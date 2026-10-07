"""Convert raw-text training examples (strata-style JSONL) into training_data.csv rows.

This bridges the two training architectures:

  STRATA:       text → textToSignals() → 26 binary signals → logistic regression weights
  MODEL-PLANE:  text → extract_features() → 14 float features → GBT/RF joblib model

Both systems do the same thing — convert text to a feature vector, then learn weights.
The difference is WHERE the text lives:
  • Strata keeps raw text in classifier_training.jsonl and featurises at train time.
  • Model-plane stores pre-computed feature floats in training_data.csv.

This script lets you add training examples the strata way (raw text + label) and outputs
CSV rows that can be appended to training_data.csv or uploaded via the Admin UI.

Input JSONL format (one JSON object per line, strata-compatible):
    {"text": "Fix this Python error: NameError", "task_type": "code_debugging", "tier": "complex"}
    {"text": "Hello, how are you?",              "task_type": "simple_qa",       "tier": "simple"}

Alternative: use "labels" key (strata native) — labels[0] is the task_type signal name,
which is automatically mapped to the model-plane TaskType enum value:
    {"text": "Fix this Python error", "labels": ["code_debug", "code_presence"]}
    → task_type="code_debugging", tier derived from task_type default

Usage:
    # Convert a JSONL file to CSV rows and append to training_data.csv:
    python -m model_plane.ml.generate_from_text \\
        --input path/to/examples.jsonl \\
        --output model_plane/data/training_data.csv \\
        --merge

    # Dry-run: print CSV to stdout without writing:
    python -m model_plane.ml.generate_from_text --input examples.jsonl --dry-run

    # Convert strata's own classifier_training.jsonl:
    python -m model_plane.ml.generate_from_text \\
        --input /path/to/classifier_training.jsonl \\
        --output model_plane/data/training_data.csv \\
        --merge
"""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── strata signal name → model-plane task_type enum value ────────────────────
# Strata's classifier_training.jsonl uses short signal names like "code_debug".
# Model-plane uses full TaskType enum values like "code_debugging".
# Only labels[0] is used as the task_type; secondary labels are ignored.
_SIGNAL_TO_TASK: dict[str, str] = {
    "code_debug":     "code_debugging",
    "code_gen":       "code_generation",
    "code_edit":      "code_editing",
    "repo_search":    "repository_search",
    "math":           "mathematical_reasoning",
    "tech_reason":    "technical_reasoning",
    "extract":        "structured_extraction",
    "summarize":      "summarization",
    "translate":      "translation",
    "creative":       "creative_writing",
    "plan":           "planning",
    "tool":           "tool_call_interpretation",
    "long_ctx":       "long_context_synthesis",
    "simple":         "simple_qa",
    "simple_qa":      "simple_qa",
    "greeting":       "simple_qa",    # greetings → simple_qa
    "explain_simple": "simple_qa",
    "unknown":        "unknown",
    # model-plane native values pass through unchanged
    "code_debugging":            "code_debugging",
    "code_generation":           "code_generation",
    "code_editing":              "code_editing",
    "repository_search":         "repository_search",
    "mathematical_reasoning":    "mathematical_reasoning",
    "technical_reasoning":       "technical_reasoning",
    "structured_extraction":     "structured_extraction",
    "summarization":             "summarization",
    "translation":               "translation",
    "creative_writing":          "creative_writing",
    "planning":                  "planning",
    "tool_call_interpretation":  "tool_call_interpretation",
    "long_context_synthesis":    "long_context_synthesis",
    "simple_qa":                 "simple_qa",
}

# Default tier for each task_type (mirrors TASK_DEFAULT_TIER in taxonomy.py)
_TASK_DEFAULT_TIER: dict[str, str] = {
    "simple_qa":                "simple",
    "summarization":            "medium",
    "translation":              "medium",
    "creative_writing":         "medium",
    "structured_extraction":    "medium",
    "planning":                 "complex",
    "code_generation":          "complex",
    "code_editing":             "complex",
    "code_debugging":           "complex",
    "repository_search":        "complex",
    "tool_call_interpretation":  "medium",
    "technical_reasoning":      "complex",
    "mathematical_reasoning":   "reasoning",
    "long_context_synthesis":   "complex",
    "unknown":                  "medium",
}

# Canonical CSV column order — matches training_data.csv header
_FEATURE_COLUMNS = [
    "reasoning_markers", "code_presence", "simple_indicators", "multi_step_patterns",
    "technical_terms", "token_count_signal", "creative_markers", "question_complexity",
    "constraint_count", "imperative_verbs", "output_format", "domain_specificity",
    "reference_complexity", "negation_complexity",
]

_CSV_COLUMNS = [
    "timestamp", "request_id", "task_type", "tier",
    *_FEATURE_COLUMNS,
    "selected_deployment", "latency_ms", "cost_usd", "validation_result",
    "fallback_used", "final_success", "feedback_source", "quality_score",
]


def _text_to_csv_row(text: str, task_type: str, tier: str) -> dict:
    """Run extract_features() on text and return a CSV-ready row dict."""
    from model_plane.classifier.features import extract_features

    feats = extract_features({"messages": [{"role": "user", "content": text}]})

    return {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "request_id":         f"gen-{uuid.uuid4().hex[:8]}",
        "task_type":          task_type,
        "tier":               tier,
        "reasoning_markers":  round(feats.reasoning_markers, 4),
        "code_presence":      round(feats.code_presence, 4),
        "simple_indicators":  round(feats.simple_indicators, 4),
        "multi_step_patterns": round(feats.multi_step_patterns, 4),
        "technical_terms":    round(feats.technical_terms, 4),
        "token_count_signal": round(feats.token_count_signal, 4),
        "creative_markers":   round(feats.creative_markers, 4),
        "question_complexity": round(feats.question_complexity, 4),
        "constraint_count":   round(feats.constraint_count, 4),
        "imperative_verbs":   round(feats.imperative_verbs, 4),
        "output_format":      round(feats.output_format, 4),
        "domain_specificity": round(feats.domain_specificity, 4),
        "reference_complexity": round(feats.reference_complexity, 4),
        "negation_complexity": round(feats.negation_complexity, 4),
        "selected_deployment": "",
        "latency_ms":         "",
        "cost_usd":           "",
        "validation_result":  "ok",
        "fallback_used":      "False",
        "final_success":      "1.0",
        "feedback_source":    "generated_from_text",
        "quality_score":      "0.90",
    }


def convert(
    input_path: Path,
    output_path: Path | None = None,
    merge: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    """Read a JSONL file of text examples and produce CSV rows.

    Args:
        input_path:  Path to the input .jsonl file.
        output_path: Destination CSV path. If None, returns rows only (dry-run friendly).
        merge:       When True, append to existing CSV; when False, replace it.
        dry_run:     If True, print rows to stdout and do not write.

    Returns:
        List of generated CSV row dicts.
    """
    rows: list[dict] = []
    skipped = 0

    with open(input_path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [line {lineno}] JSON parse error: {exc} — skipped")
                skipped += 1
                continue

            text: str = obj.get("text", "").strip()
            if not text:
                print(f"  [line {lineno}] No 'text' key — skipped")
                skipped += 1
                continue

            # Resolve task_type: prefer explicit "task_type" key, then labels[0]
            task_raw: str = ""
            if "task_type" in obj:
                task_raw = str(obj["task_type"]).strip().lower()
            elif "labels" in obj and obj["labels"]:
                task_raw = str(obj["labels"][0]).strip().lower()

            task_type = _SIGNAL_TO_TASK.get(task_raw, "")
            if not task_type:
                print(f"  [line {lineno}] Unknown task label '{task_raw}' — skipped")
                skipped += 1
                continue

            # Resolve tier: prefer explicit "tier" key, fall back to task default
            tier = str(obj.get("tier", "")).strip().lower()
            if tier not in ("simple", "medium", "complex", "reasoning"):
                tier = _TASK_DEFAULT_TIER.get(task_type, "medium")

            row = _text_to_csv_row(text, task_type, tier)
            rows.append(row)

    print(f"Converted {len(rows)} rows ({skipped} skipped)")

    if dry_run:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return rows

    if output_path is None:
        return rows

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if merge and output_path.exists():
        with open(output_path, newline="") as fh:
            existing = list(csv.DictReader(fh))
        combined = existing + rows
    else:
        combined = rows

    # Derive the full column union (existing columns first, then any new ones)
    if merge and output_path.exists():
        with open(output_path, newline="") as fh:
            reader = csv.DictReader(fh)
            existing_cols = list(reader.fieldnames or _CSV_COLUMNS)
        extra = [c for c in _CSV_COLUMNS if c not in existing_cols]
        final_cols = existing_cols + extra
    else:
        final_cols = _CSV_COLUMNS

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=final_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)

    print(f"Written {len(combined)} rows to {output_path}  (was {len(combined) - len(rows)}, added {len(rows)})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert text-labelled JSONL examples to training_data.csv rows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  "-i", type=Path, required=True, help="Input .jsonl file")
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("model_plane/data/training_data.csv"),
        help="Output CSV path (default: model_plane/data/training_data.csv)",
    )
    parser.add_argument(
        "--merge", action="store_true", default=True,
        help="Append to existing CSV (default: True)",
    )
    parser.add_argument(
        "--replace", dest="merge", action="store_false",
        help="Replace existing CSV instead of appending",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated CSV rows to stdout, do not write to disk",
    )
    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=None if args.dry_run else args.output,
        merge=args.merge,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
