"""Similarity-based router — routes to the deployment that worked best for
similar past prompts.

Phase 3 implementation:
- Embeds prompt text with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- Maintains an in-memory index of (embedding, deployment_name, success_score) tuples
- At query time: cosine-similarity top-k retrieval → weighted vote over deployments
- Gets smarter with every request via the feedback / post-call update path
- Optional FAISS index for large histories (falls back to numpy dot-product)

Enable:   MODEL_PLANE_SIMILARITY_ROUTING_ENABLED=true
Threshold: MODEL_PLANE_SIMILARITY_CONFIDENCE_THRESHOLD=0.55  (cosine similarity)
Min hits:  MODEL_PLANE_SIMILARITY_MIN_HITS=3  (index must have this many entries)
K:         MODEL_PLANE_SIMILARITY_TOP_K=5
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model_plane.logging_setup import get_logger

log = get_logger(__name__)

# ── index entry ───────────────────────────────────────────────────────────────

@dataclass
class IndexEntry:
    embedding: np.ndarray        # (384,) float32
    deployment_name: str
    task_type: str
    success_score: float         # 1.0 = good outcome, 0.0 = bad, 0.5 = unknown
    feedback_score: float | None = None  # explicit user override (1.0 / 0.0)


@dataclass
class SimilarityResult:
    deployment_name: str
    confidence: float            # top cosine similarity score
    source: str = "similarity"


# ── router ────────────────────────────────────────────────────────────────────

class SimilarityRouter:
    """In-memory cosine-similarity router backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        confidence_threshold: float = 0.55,
        min_hits: int = 3,
        index_path: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._threshold = confidence_threshold
        self._min_hits = min_hits
        self._index_path = index_path or Path("data/similarity_index.npy")
        self._entries: list[IndexEntry] = []
        self._lock = threading.RLock()
        self._encoder: Any = None   # sentence_transformers.SentenceTransformer, lazy-loaded
        self._faiss_index: Any = None  # optional faiss.IndexFlatIP

        self._load_index()

    # ── encode ────────────────────────────────────────────────────────────────

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                self._encoder = SentenceTransformer(self._model_name)
                log.info("similarity_encoder_loaded", model=self._model_name)
            except ImportError:
                log.warning(
                    "sentence_transformers_not_installed",
                    hint="pip install sentence-transformers",
                )
        return self._encoder

    def _encode(self, text: str) -> np.ndarray | None:
        enc = self._get_encoder()
        if enc is None:
            return None
        vec = enc.encode([text], normalize_embeddings=True)[0].astype(np.float32)
        return vec

    # ── index persistence ─────────────────────────────────────────────────────

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = np.load(self._index_path, allow_pickle=True).item()
            entries = data.get("entries", [])
            with self._lock:
                self._entries = entries
            log.info("similarity_index_loaded", entries=len(entries))
        except Exception as exc:
            log.warning("similarity_index_load_failed", error=str(exc))

    def _save_index(self) -> None:
        try:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                np.save(self._index_path, {"entries": self._entries})
        except Exception as exc:
            log.warning("similarity_index_save_failed", error=str(exc))

    # ── query ─────────────────────────────────────────────────────────────────

    def predict(self, prompt_text: str) -> SimilarityResult | None:
        """Return the best deployment for this prompt, or None if not confident."""
        with self._lock:
            n = len(self._entries)

        if n < self._min_hits:
            return None

        query_vec = self._encode(prompt_text)
        if query_vec is None:
            return None

        with self._lock:
            entries = list(self._entries)

        # Stack all embeddings and compute cosine similarity (they are normalised)
        matrix = np.stack([e.embedding for e in entries])  # (N, 384)
        sims = (matrix @ query_vec).astype(float)          # (N,) cosine sim

        top_k = min(self._top_k, n)
        top_idxs = np.argsort(sims)[::-1][:top_k]
        top_sim = float(sims[top_idxs[0]])

        if top_sim < self._threshold:
            log.debug(
                "similarity_below_threshold",
                top_sim=round(top_sim, 3),
                threshold=self._threshold,
            )
            return None

        # Weighted vote: deployment → sum of (similarity × success_score)
        votes: dict[str, float] = {}
        for idx in top_idxs:
            e = entries[idx]
            weight = float(sims[idx])
            effective_score = e.feedback_score if e.feedback_score is not None else e.success_score
            votes[e.deployment_name] = votes.get(e.deployment_name, 0.0) + weight * effective_score

        best_dep = max(votes, key=lambda d: votes[d])
        log.debug(
            "similarity_prediction",
            top_sim=round(top_sim, 3),
            top_k_deployments=list(votes.keys()),
            winner=best_dep,
        )
        return SimilarityResult(deployment_name=best_dep, confidence=top_sim)

    # ── update ────────────────────────────────────────────────────────────────

    def add_entry(
        self,
        prompt_text: str,
        deployment_name: str,
        task_type: str,
        success_score: float = 0.5,
    ) -> None:
        """Add a new prompt→deployment outcome to the index."""
        vec = self._encode(prompt_text)
        if vec is None:
            return
        entry = IndexEntry(
            embedding=vec,
            deployment_name=deployment_name,
            task_type=task_type,
            success_score=success_score,
        )
        with self._lock:
            self._entries.append(entry)
            # Evict oldest entries beyond 10 000 to prevent unbounded growth
            if len(self._entries) > 10_000:
                self._entries = self._entries[-10_000:]
        self._save_index()

    def apply_feedback(self, prompt_text: str, correct_deployment: str) -> int:
        """
        Apply user feedback: find the most similar indexed entry and set its
        feedback_score. Returns number of entries updated.
        """
        vec = self._encode(prompt_text)
        if vec is None:
            return 0

        updated = 0
        with self._lock:
            if not self._entries:
                return 0
            matrix = np.stack([e.embedding for e in self._entries])
            sims = matrix @ vec
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= 0.7:
                self._entries[best_idx].feedback_score = (
                    1.0 if self._entries[best_idx].deployment_name == correct_deployment else 0.0
                )
                updated = 1

        if updated:
            self._save_index()
        return updated


# ── singleton ─────────────────────────────────────────────────────────────────

_router: SimilarityRouter | None = None


def get_similarity_router() -> SimilarityRouter | None:
    """Return the singleton SimilarityRouter, or None if not enabled."""
    from model_plane.config import settings

    if not getattr(settings, "similarity_routing_enabled", False):
        return None

    global _router
    if _router is None:
        _router = SimilarityRouter(
            top_k=getattr(settings, "similarity_top_k", 5),
            confidence_threshold=getattr(settings, "similarity_confidence_threshold", 0.55),
            min_hits=getattr(settings, "similarity_min_hits", 3),
        )
    return _router
