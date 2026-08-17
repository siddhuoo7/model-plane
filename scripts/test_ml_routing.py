"""
Quick verification of ML active routing.
Run from repo root:  python scripts/test_ml_routing.py
"""
from __future__ import annotations
import os, sys

os.environ["MODEL_PLANE_ML_ROUTING_ENABLED"] = "true"
os.environ["MODEL_PLANE_ML_CONFIDENCE_THRESHOLD"] = "0.65"
os.environ["MODEL_PLANE_LOG_LEVEL"] = "warning"   # suppress debug noise

# Force ML singleton reload so it picks up the new joblib + threshold
import model_plane.ml.recommender as _r
_r._recommender = None

from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline
from model_plane.ml.recommender import get_recommender
from model_plane.classifier.features import extract_features

CASES = [
    ("SIMPLE",              "What is the capital of France?",
     "simple_qa"),
    ("CODE_GEN",            "Write a Python class that implements a binary search tree with insert, delete, and in-order traversal methods.",
     "code_generation"),
    ("MATH",                "Prove that the sum of the first n odd numbers equals n squared using mathematical induction. Show every step.",
     "mathematical_reasoning"),
    ("TECHNICAL",           "Explain the trade-offs between B-tree and LSM-tree storage engines for a write-heavy workload. Consider compaction, read amplification, and write amplification.",
     "technical_reasoning"),
    ("STRUCTURED_EXTRACT",  "Extract all entities from the following text and return them as a JSON object with fields: people, organisations, dates, locations.\n\nText: On 14 March 2024, IBM CEO Arvind Krishna announced a partnership with the European Central Bank in Frankfurt.",
     "structured_extraction"),
    ("LONG_CTX",            "I have a 50,000-word document. Synthesise the key themes, contradictions, and conclusions across all sections into a structured executive summary.",
     "long_context_synthesis"),
    ("CODE_DEBUG",          "Debug this Python function:\n```python\ndef fib(n):\n    if n == 0: return 1\n    return fib(n-1) + fib(n-2)\n```\nIt gives the wrong answer. Explain the bug and fix it.",
     "code_debugging"),
]

# ── Load model and show calibrated probas ────────────────────────────────────
rec = get_recommender()
print(f"\nML model loaded : {rec._loaded}")
print(f"Model path      : {rec._model_path}")
print(f"Threshold       : 0.65\n")

# Show calibrated probas for a clear simple case vs a clear complex case
for label, text in [("simple_qa", "What is the capital of France?"),
                    ("code_generation", "Write a Python BST class with insert and delete.")]:
    f = extract_features({"messages": [{"role": "user", "content": text}]})
    probas = rec._model.predict_proba([f.as_vector()])[0]
    labels = ["simple", "medium", "complex", "reasoning"]
    best_conf = max(probas)
    best_tier = labels[list(probas).index(best_conf)]
    print(f"Calibrated probas [{label}]:")
    for l, p in zip(labels, probas):
        bar = "█" * int(p * 30)
        marker = " ◀ FIRES (≥0.65)" if p >= 0.65 else ""
        print(f"  {l:12s}  {p:.4f}  {bar}{marker}")
    print()

# ── Run all 7 cases ──────────────────────────────────────────────────────────
W = [6, 38, 28, 10, 7, 9, 16, 9, 7, 9, 22, 12]  # col widths
HDR = ["#", "Case", "Task ✓", "ClfTier", "BlendTier", "Score", "ML-Conf", "Source", "Deployment"]
fmt = "{:>2}  {:<22}  {:<28}  {:<9}  {:<12}  {:<7}  {:<9}  {:<18}  {}"
print(fmt.format(*HDR))
print("─" * 130)

ml_fired = 0
results = []
for i, (label, prompt, expected_task) in enumerate(CASES):
    ctx = build_routing_context({"messages": [{"role": "user", "content": prompt}], "max_tokens": 512})
    dec = run_routing_pipeline(ctx)

    task    = ctx.classification.task_type.value if ctx.classification else "?"
    clf_t   = ctx.classification.complexity_tier.value if ctx.classification else "?"
    blend_t = ctx.scorer_result.tier.value if ctx.scorer_result else "?"
    score   = round(ctx.scorer_result.raw_score, 4) if ctx.scorer_result else 0.0
    ml_conf = round(ctx.ml_recommendation_confidence, 3)
    src     = ctx.routing_source
    dep     = dec.deployment.name

    task_ok  = f"✅ {task}" if task == expected_task else f"⚠  {task} (expected {expected_task})"
    blend_up = f"{blend_t} ⬆" if blend_t != clf_t else blend_t
    src_fmt  = f"🤖 ml_active" if src == "ml_active" else f"   {src}"
    if src == "ml_active":
        ml_fired += 1

    print(fmt.format(i+1, label, task_ok, clf_t, blend_up, score, ml_conf, src_fmt, dep))
    results.append((label, task, expected_task, ml_conf, src, dep))

print()
print(f"{'─'*60}")
print(f"ML fired (source=ml_active): {ml_fired} / {len(CASES)} cases")
correct = sum(1 for _, t, e, _, _, _ in results if t == e)
print(f"Task classification correct: {correct} / {len(CASES)}")
print(f"{'─'*60}")

# ── Summary per case ─────────────────────────────────────────────────────────
print("\nPer-case verdict:")
for label, task, expected, ml_conf, src, dep in results:
    ml_tag  = f"ML conf={ml_conf} → 🤖 ACTIVE" if src == "ml_active" else f"ML conf={ml_conf} → below gate"
    print(f"  {label:<22}  {task:<28}  {dep:<30}  {ml_tag}")
