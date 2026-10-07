"""Laya-based classifier — runs fully in-process, no network call required.

Uses the ``laya`` SDK (``pip install laya``) to answer BOTH the task_type
classification question AND the complexity_tier scoring question in a single
``Router.predict()`` call.  This collapses the two-model architecture into one
Laya call while the regex classifier still runs as a fast-path and fallback.

Enable by setting:  CLASSIFIER_MODEL=laya

Expected p99 latency: 200–300 ms on CPU.
``asyncio.to_thread`` is used because ``Router.predict()`` is synchronous CPU work
that would block the FastAPI event loop without it.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING, Any

from model_plane.classifier.taxonomy import ComplexityTier, TaskType
from model_plane.logging_setup import get_logger

if TYPE_CHECKING:
    from model_plane.classifier.classifier import ClassificationResult
    from model_plane.classifier.features import RequestFeatures

log = get_logger(__name__)

# ── Laya question definitions ─────────────────────────────────────────────────

# Ordered list of (label, description) pairs.
# Order matters: laya 'score' indexes levels 0..N-1, and probabilities[i] maps back
# to TASK_TYPE_LABELS[i].  Add new types at the END to preserve existing model indices.
_TASK_TYPE_LEVELS: list[tuple[str, str]] = [
    ("simple_qa",               "Simple factual questions, definitions, yes/no lookups"),
    ("summarization",           "Summarise or condense a document or passage"),
    ("translation",             "Translate text between languages"),
    ("creative_writing",        "Creative writing: poems, stories, fiction, imaginative tasks"),
    ("structured_extraction",   "Extract structured data: JSON, tables, key-value pairs"),
    ("planning",                "Create project plans, roadmaps, milestones, action items"),
    ("code_generation",         "Write new code: functions, classes, scripts, APIs"),
    ("code_editing",            "Refactor, rename, optimise, or clean up existing code"),
    ("code_debugging",          "Debug errors, fix failing tests, trace exceptions"),
    ("repository_search",       "Find or locate files, functions, or patterns in a codebase"),
    ("tool_call_interpretation", "Interpret tool outputs or decide which tool to call"),
    ("technical_reasoning",     "Compare architectures, evaluate trade-offs, explain designs"),
    ("mathematical_reasoning",  "Solve equations, proofs, statistics, or numerical problems"),
    ("long_context_synthesis",  "Analyse or synthesise content across very long documents"),
    ("unknown",                 "Does not clearly fit any of the above categories"),
]

# Parallel structures derived from _TASK_TYPE_LEVELS — do not edit independently.
TASK_TYPE_LABELS: list[str] = [label for label, _ in _TASK_TYPE_LEVELS]
# laya score criteria: a list of level descriptions, index 0 first.
TASK_TYPE_SCORE_CRITERIA: list[str] = [desc for _, desc in _TASK_TYPE_LEVELS]

TIER_CRITERIA: list[str] = ["simple", "medium", "complex", "reasoning"]

LAYA_QUESTIONS: dict[str, Any] = {
    # score: criteria must be a list of level descriptions (index 0 first).
    # Returns probabilities[i] per level; we pick argmax to get the dominant task type.
    # Using score (not choice) means the model outputs a continuous distribution across
    # all levels, so combined requests (e.g. code + math) surface the dominant type
    # via the probability mass rather than a hard single pick.
    "task_type": {
        "type": "score",
        "instructions": (
            "Rate the task types that describe the user's primary request. "
            "A combined request (e.g. code and maths) should score highest on its dominant type."
        ),
        "criteria": TASK_TYPE_SCORE_CRITERIA,
    },
    # choice: directly picks the complexity tier — avoids threshold ambiguity.
    "complexity_tier": {
        "type": "choice",
        "instructions": (
            "Pick the single complexity tier that best describes this request. "
            "simple: factual lookups, greetings. "
            "medium: explanations, summaries, standard code. "
            "complex: multi-step design, debugging, architecture. "
            "reasoning: proofs, mathematical derivations, deep inference."
        ),
        "criteria": TIER_CRITERIA,
    },
}

# ── Tier score → ComplexityTier mapping ───────────────────────────────────────
# score < threshold → assign this tier  (thresholds configurable in routing.yaml)

TIER_SCORE_THRESHOLDS: dict[ComplexityTier, float] = {
    ComplexityTier.SIMPLE:    0.33,
    ComplexityTier.MEDIUM:    0.55,
    ComplexityTier.COMPLEX:   0.78,
    ComplexityTier.REASONING: 1.01,   # catch-all upper bound
}


def _score_to_tier(score: float) -> ComplexityTier:
    """Map a Laya ``score`` float [0, 1] to a ``ComplexityTier``."""
    for tier, threshold in TIER_SCORE_THRESHOLDS.items():
        if score < threshold:
            return tier
    return ComplexityTier.REASONING


def _confidence_from_tier_score(tier_score: float) -> float:
    """Map a tier score to a confidence value in [0.60, 0.95].

    A score near 0 or near 1 means Laya is decisive (high confidence).
    A score near 0.5 (threshold boundary) signals uncertainty (low confidence).
    Formula: 0.60 + 0.35 * |score - 0.5| * 2  → range [0.60, 0.95]
    """
    midpoint = 0.5
    return round(0.60 + 0.35 * abs(tier_score - midpoint) * 2, 3)


# ── Laya Router singleton ─────────────────────────────────────────────────────

_router: Any = None   # laya.Router, lazy-loaded


def _get_router() -> Any:
    """Lazy-load the Laya Router singleton (thread-safe first use)."""
    global _router
    if _router is not None:
        return _router
    try:
        from laya import Router  # type: ignore[import]
        # Laya ships checkpoints with invalid temperature parameters that trigger a
        # RuntimeWarning from within Laya's own inference code. The warning is benign
        # (Laya handles it internally) but pollutes startup logs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="laya")
            _router = Router()
        log.info("laya_router_loaded")
    except ImportError:
        log.warning(
            "laya_unavailable",
            reason="laya not installed",
            hint="pip install laya",
        )
    return _router


# ── ClassifierAdapter implementation ─────────────────────────────────────────

class LayaClassifier:
    """Local Laya classifier that answers BOTH task_type AND complexity_tier.

    A single ``Router.predict()`` call fills the full ``ClassificationResult``.
    Source tag ``_source=2.0`` is stored in signals to distinguish from BERT
    (``_source=1.0``) and regex (``_source=0.0``).
    """

    def __init__(self, min_confidence: float = 0.60) -> None:
        self._min_confidence = min_confidence

    async def classify(self, features: "RequestFeatures") -> "ClassificationResult | None":
        from model_plane.classifier.classifier import ClassificationResult  # local import

        router = _get_router()
        if router is None:
            return None

        state = _build_state(features)

        try:
            result = await asyncio.to_thread(router.predict, state, LAYA_QUESTIONS)
        except Exception as exc:
            log.warning("laya_inference_failed", error=str(exc))
            return None

        try:
            # task_type is "score" — laya returns per-level probabilities keyed by str(index).
            # Pick the index with the highest probability, then map to the label name.
            task_answer: dict = result["answers"]["task_type"]
            task_probs: dict = task_answer.get("probabilities", {})
            if task_probs:
                best_idx = max(task_probs, key=lambda i: float(task_probs[i]))
                task_label: str = TASK_TYPE_LABELS[int(best_idx)]
                task_score_val: float = float(task_probs[best_idx])
            else:
                # Fallback: use the scalar score as an index into the label list
                raw_score = float(task_answer.get("score", 0.0))
                task_label = TASK_TYPE_LABELS[min(int(round(raw_score)), len(TASK_TYPE_LABELS) - 1)]
                task_score_val = 0.75
            # complexity_tier is "choice" — returns the selected tier name directly
            tier_label: str = result["answers"]["complexity_tier"]["choice"]
            laya_model: str = result.get("routing", {}).get("model", "")
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("laya_result_parse_failed", error=str(exc))
            return None

        try:
            task_type = TaskType(task_label)
        except ValueError:
            log.warning("laya_unknown_task_label", label=task_label)
            task_type = TaskType.UNKNOWN

        # Map tier label to ComplexityTier enum
        tier_map = {
            "simple":    ComplexityTier.SIMPLE,
            "medium":    ComplexityTier.MEDIUM,
            "complex":   ComplexityTier.COMPLEX,
            "reasoning": ComplexityTier.REASONING,
        }
        tier = tier_map.get(tier_label.lower(), ComplexityTier.MEDIUM)

        # Confidence: use the winning task_type probability as a proxy
        confidence = round(min(max(task_score_val, 0.60), 0.98), 3)

        if confidence < self._min_confidence:
            return None

        log.debug(
            "laya_classifier_result",
            task_type=task_type.value,
            tier=tier.value,
            tier_label=tier_label,
            confidence=confidence,
            laya_model=laya_model,
        )

        # signals must be dict[str, float]; encode tier as its index (0–3)
        _TIER_INDEX = {"simple": 0.0, "medium": 1.0, "complex": 2.0, "reasoning": 3.0}
        return ClassificationResult(
            task_type=task_type,
            complexity_tier=tier,
            confidence=confidence,
            signals={
                "_source":      2.0,
                "_tier_index":  _TIER_INDEX.get(tier_label.lower(), 1.0),
                "_laya_model":  1.0 if laya_model else 0.0,
                # top task probability so the simulator can show it
                "_task_score":  round(task_score_val, 3),
            },
        )


def _build_state(features: "RequestFeatures") -> dict:
    """Build the Laya state dict from extracted ``RequestFeatures``."""
    return {
        "prompt":        features.last_user_text,
        "full_context":  features.full_text[:2000],
        "has_code":      features.code_presence > 0.1,
        "has_math":      features.reasoning_markers > 0.3,
        "token_count":   features.total_tokens,
        "multi_turn":    features.user_message_count > 2,
        "has_tools":     features.has_tools,
        "output_format": features.output_format > 0.1,
    }
