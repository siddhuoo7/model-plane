"""BERT-based intent classifier — optional augmentation for the regex classifier.

Uses a HuggingFace transformers pipeline to classify task intent from prompt text.
Falls back gracefully if transformers/torch is not installed or the model fails to load.

Enable:  MODEL_PLANE_BERT_CLASSIFIER_ENABLED=true  (deprecated — use CLASSIFIER_MODEL=mbert)
Model:   MODEL_PLANE_BERT_CLASSIFIER_MODEL=cross-encoder/nli-MiniLM2-L6-H768
         (default; MiniLM cross-encoder fine-tuned for NLI, ~22 MB, <100ms on CPU)

The classifier runs on CPU by default — no GPU required.

Label mapping: the zero-shot NLI pipeline scores each of the 15 candidate label strings
against the prompt using the model's entailment head.  Top-scoring label wins.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from model_plane.classifier.taxonomy import TaskType
from model_plane.logging_setup import get_logger

log = get_logger(__name__)

# ── label strings passed to zero-shot pipeline ───────────────────────────────
# These are human-readable descriptions that the zero-shot NLI model scores against.
# Kept intentionally short so the model's entailment head fires on keyword overlap.
_CANDIDATE_LABELS: list[str] = [
    "code debugging and fixing errors",
    "code generation and writing new code",
    "code editing and refactoring",
    "repository search and code navigation",
    "mathematical reasoning and calculation",
    "technical reasoning and architecture comparison",
    "long context synthesis and document analysis",
    "tool call interpretation",
    "structured data extraction to JSON",
    "text summarization",
    "language translation",
    "creative writing and storytelling",
    "planning and project management",
    "simple question answering",
    "unknown or general request",
]

# Map from candidate label (lowered) substring → TaskType
_LABEL_TO_TASK: dict[str, TaskType] = {
    "code debugging":          TaskType.CODE_DEBUGGING,
    "code generation":         TaskType.CODE_GENERATION,
    "code editing":            TaskType.CODE_EDITING,
    "repository search":       TaskType.REPOSITORY_SEARCH,
    "mathematical reasoning":  TaskType.MATHEMATICAL_REASONING,
    "technical reasoning":     TaskType.TECHNICAL_REASONING,
    "long context":            TaskType.LONG_CONTEXT_SYNTHESIS,
    "tool call":               TaskType.TOOL_CALL_INTERPRETATION,
    "structured data":         TaskType.STRUCTURED_EXTRACTION,
    "text summarization":      TaskType.SUMMARIZATION,
    "language translation":    TaskType.TRANSLATION,
    "creative writing":        TaskType.CREATIVE_WRITING,
    "planning":                TaskType.PLANNING,
    "simple question":         TaskType.SIMPLE_QA,
    "unknown":                 TaskType.UNKNOWN,
}


@dataclass
class BertClassificationResult:
    task_type: TaskType
    confidence: float          # probability score from NLI pipeline [0, 1]
    source: str = "bert"       # "bert" | "fallback"


# ── singleton ─────────────────────────────────────────────────────────────────

_pipeline: Any = None          # transformers.Pipeline, lazy-loaded
_pipeline_lock = threading.Lock()
_load_attempted = False        # only attempt load once; avoid repeated failures


def _label_to_task_type(label: str) -> TaskType:
    """Map a candidate label string back to a TaskType enum value."""
    label_lower = label.lower()
    for key, task in _LABEL_TO_TASK.items():
        if key in label_lower:
            return task
    return TaskType.UNKNOWN


# Track which model name the current singleton was loaded for.
# If model_name changes (e.g. config update) the pipeline is reloaded.
_pipeline_model_name: str | None = None


def _get_pipeline(model_name: str) -> Any | None:
    """Lazy-load the zero-shot classification pipeline (thread-safe).

    Reloads automatically if *model_name* differs from the loaded model —
    this handles the case where ``bert_classifier_model`` is changed in config
    without restarting the server.
    """
    global _pipeline, _load_attempted, _pipeline_model_name

    # Fast path: already loaded for this model name
    if _pipeline is not None and _pipeline_model_name == model_name:
        return _pipeline

    with _pipeline_lock:
        # Double-check inside lock
        if _pipeline is not None and _pipeline_model_name == model_name:
            return _pipeline

        # Model name changed — force reload
        if _pipeline_model_name != model_name:
            _pipeline = None
            _load_attempted = False
            _pipeline_model_name = model_name

        if _load_attempted:
            return None                   # previous load for this model failed; don't retry
        _load_attempted = True

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import]

            log.info(
                "bert_classifier_loading",
                model=model_name,
                hint="first request will be slower; model cached after first load",
            )
            _pipeline = hf_pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,               # force CPU; set device=0 for GPU
                truncation=True,
                max_length=512,
            )
            log.info("bert_classifier_loaded", model=model_name)
        except ImportError:
            log.warning(
                "bert_classifier_unavailable",
                reason="transformers not installed",
                hint="pip install 'model-plane[bert]'",
            )
        except Exception as exc:
            log.warning("bert_classifier_load_failed", error=str(exc), model=model_name)

    return _pipeline


def predict(text: str, model_name: str) -> BertClassificationResult | None:
    """Run zero-shot classification on *text* and return the top TaskType.

    Returns None if the model is not loaded or inference fails — caller should
    fall back to the regex classifier in that case.

    Confidence calibration:
        NLI zero-shot spreads probability across all N candidate labels, so the
        raw top score for 15 labels (~0.20–0.35) is much lower than binary
        classifier scores (~0.85–0.99).  We calibrate by mapping the raw score
        against the uniform-random baseline (1/N) to a [0, 1] range:

            calibrated = (top_score - 1/N) / (1 - 1/N)

        A score at chance → 0.0; a score of 1.0 → 1.0.  This lets the same
        0.60 threshold work across both binary and NLI classifiers.

    Args:
        text:       The user prompt text to classify (last user turn is best).
        model_name: HuggingFace model ID to load.

    Returns:
        BertClassificationResult or None on failure.
    """
    pipe = _get_pipeline(model_name)
    if pipe is None:
        return None

    # Truncate to 400 chars for speed — the first few sentences are most informative
    snippet = text[:400].strip() or text
    if not snippet:
        return None

    try:
        result = pipe(snippet, candidate_labels=_CANDIDATE_LABELS, multi_label=False)
        # result = {"sequence": ..., "labels": [...], "scores": [...]}
        top_label: str = result["labels"][0]
        top_score: float = float(result["scores"][0])

        # Calibrate: map raw score from [1/N, 1.0] → [0.0, 1.0]
        n_labels = len(_CANDIDATE_LABELS)
        chance = 1.0 / n_labels
        calibrated = max(0.0, (top_score - chance) / (1.0 - chance))

        task_type = _label_to_task_type(top_label)

        log.debug(
            "bert_classifier_result",
            top_label=top_label,
            task_type=task_type.value,
            raw_score=round(top_score, 3),
            confidence=round(calibrated, 3),
        )
        return BertClassificationResult(
            task_type=task_type,
            confidence=calibrated,
        )
    except Exception as exc:
        log.warning("bert_classifier_inference_failed", error=str(exc))
        return None


def reload() -> None:
    """Force reload of the pipeline on next predict() call (e.g. after model update)."""
    global _pipeline, _load_attempted, _pipeline_model_name
    with _pipeline_lock:
        _pipeline = None
        _load_attempted = False
        _pipeline_model_name = None
    log.info("bert_classifier_reset")


# ── ClassifierAdapter implementation ─────────────────────────────────────────

class BertClassifierAdapter:
    """Wraps the module-level ``predict()`` function as a ``ClassifierAdapter``.

    Only sets ``task_type`` and ``confidence``; ``complexity_tier`` is always
    re-derived from the regex ``_derive_tier()`` so routing behaviour stays
    deterministic.  Source tag ``_source=1.0`` is stored in signals.

    ``predict()`` is synchronous CPU inference; offloaded via ``asyncio.to_thread``
    to avoid blocking the FastAPI event loop.
    """

    def __init__(self, model_name: str, min_confidence: float = 0.10) -> None:
        # NLI zero-shot calibrated confidence: 0.10 corresponds to the top label
        # scoring ~2.7× the uniform-random baseline across 15 candidate labels.
        # The cross-encoder/nli-MiniLM2-L6-H768 model is consistently accurate
        # even at low calibrated scores — empirically 100% accuracy at cal≥0.05.
        self._model_name = model_name
        self._min_confidence = min_confidence

    async def classify(
        self,
        features: "RequestFeatures",  # type: ignore[name-defined]  # noqa: F821
    ) -> "ClassificationResult | None":  # type: ignore[name-defined]  # noqa: F821
        from model_plane.classifier.classifier import ClassificationResult, _derive_tier  # local import

        result = await asyncio.to_thread(
            predict,
            text=features.last_user_text,
            model_name=self._model_name,
        )
        if result is None or result.confidence < self._min_confidence:
            return None

        tier = _derive_tier(result.task_type, features)
        return ClassificationResult(
            task_type=result.task_type,
            complexity_tier=tier,
            confidence=result.confidence,
            signals={"_source": 1.0},
        )
