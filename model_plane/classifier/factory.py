"""Classifier factory — returns the correct ``ClassifierAdapter`` for the current config.

Selection logic (single ``CLASSIFIER_MODEL`` env var):

  CLASSIFIER_MODEL=regex      -> None  (regex only; zero deps)
  CLASSIFIER_MODEL=ml_task    -> MLTaskClassifier (Sub-Task 4.3, flag-gated)
  CLASSIFIER_MODEL=mbert      -> BertClassifierAdapter
  CLASSIFIER_MODEL=laya       -> LayaClassifier
  anything else               -> None  (same as regex)

Backward-compat:  ``BERT_CLASSIFIER_ENABLED=true`` is treated as an alias for
``CLASSIFIER_MODEL=mbert``.  If both are set, ``CLASSIFIER_MODEL`` wins.

Sub-Task 4.3:  ``ML_TASK_CLASSIFIER_ENABLED=true`` enables
``CLASSIFIER_MODEL=ml_task`` as an alternative lightweight classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Module-level import so tests can patch 'model_plane.classifier.factory.settings'
# or the canonical 'model_plane.config.settings' and the factory will see it.
from model_plane.config import settings
from model_plane.logging_setup import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from model_plane.classifier.base import ClassifierAdapter


# ── availability check ────────────────────────────────────────────────────────

def check_classifier_availability() -> dict[str, bool]:
    """Return availability of each optional classifier backend.

    Called at startup (from app.py) so missing deps are logged immediately.
    Also used by the classifiers admin API to surface install_hint per classifier.

    Returns a dict: {"laya": bool, "mbert": bool}
    """
    availability: dict[str, bool] = {}

    # Check laya
    try:
        import laya  # noqa: F401
        availability["laya"] = True
    except ImportError:
        availability["laya"] = False
        log.warning(
            "classifier_unavailable",
            classifier="laya",
            reason="laya not installed",
            fix="pip install 'model-plane[laya]'  OR  pnpm install",
        )

    # Check mbert (transformers + torch)
    try:
        import transformers  # noqa: F401
        availability["mbert"] = True
    except ImportError:
        availability["mbert"] = False
        log.warning(
            "classifier_unavailable",
            classifier="mbert",
            reason="transformers not installed",
            fix="pip install 'model-plane[bert]'  OR  pnpm install",
        )

    return availability


def get_classifier(owner_id: str = "") -> "ClassifierAdapter | None":
    """Return the active ``ClassifierAdapter``, or ``None`` for regex-only mode.

    Reads SQLite routing_overrides (namespace='classifier') for *owner_id* / __global__,
    falling back to ``settings.classifier_model``.
    Always safe to call — returns ``None`` when no ML classifier is configured.
    """
    model = (settings.classifier_model or "regex").strip().lower()

    # Check DB override if enabled/configured
    if getattr(settings, "db_overrides_enabled", True):
        try:
            import model_plane.db as _db
            for _k in (owner_id, "__global__"):
                if _k:
                    _clf_db = _db.load_routing_override("classifier", _k)
                    if _clf_db.get("active"):
                        model = str(_clf_db["active"]).strip().lower()
                        break
        except Exception:
            pass

    # Backward-compat: BERT_CLASSIFIER_ENABLED=true → mbert (if not already set)
    if model == "regex" and getattr(settings, "bert_classifier_enabled", False):
        model = "mbert"

    if model == "ml_task":
        from model_plane.classifier.ml_task_classifier import MLTaskClassifier
        ml_model_path = getattr(settings, "ml_task_classifier_path", "models/classifier_task.joblib")
        return MLTaskClassifier(model_path=ml_model_path)

    if model == "mbert":
        from model_plane.classifier.bert_classifier import BertClassifierAdapter
        bert_model = getattr(
            settings,
            "bert_classifier_model",
            "cross-encoder/nli-MiniLM2-L6-H768",
        )
        # NLI zero-shot calibrated scores are much lower than binary classifier scores
        # (top label out of 15 ≈ 0.10–0.50 calibrated vs 0.85+ for binary).
        # Use 0.10 as the mBERT default; only override if the user has set a
        # lower-than-default value (allowing explicit tightening via env var).
        general_conf = getattr(settings, "classifier_min_confidence", 0.60)
        min_conf = min(general_conf, 0.10)  # mBERT default caps at 0.10
        return BertClassifierAdapter(model_name=bert_model, min_confidence=min_conf)

    if model == "laya":
        from model_plane.classifier.laya_classifier import LayaClassifier
        min_conf = getattr(settings, "classifier_min_confidence", 0.60)
        return LayaClassifier(min_confidence=min_conf)

    # "regex" or unknown value → no ML augmentation
    return None
