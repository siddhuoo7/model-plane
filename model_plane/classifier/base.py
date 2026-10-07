"""ClassifierAdapter — protocol that every ML classifier backend must implement.

Three mutually exclusive backends are selected at deploy time via CLASSIFIER_MODEL:
  regex   — pure regex chain (default, zero deps)
  mbert   — regex + BERT NLI: overrides task_type, re-derives tier
  laya    — regex + Laya:     overrides BOTH task_type AND complexity_tier

Adapters return None when the model is unavailable or confidence is below the
minimum threshold, causing the regex result to stand unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from model_plane.classifier.classifier import ClassificationResult
from model_plane.classifier.features import RequestFeatures


@runtime_checkable
class ClassifierAdapter(Protocol):
    """Protocol for ML classifier backends that augment the regex baseline."""

    async def classify(self, features: RequestFeatures) -> ClassificationResult | None:
        """Classify a request from its extracted feature vector.

        Args:
            features: The extracted ``RequestFeatures`` for this request.

        Returns:
            A ``ClassificationResult`` when the model fires with sufficient
            confidence, or ``None`` to signal that the regex result should stand.
        """
        ...
