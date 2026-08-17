"""Session and cache state management.

Tracks:
- Session state (session_id → anchor deployment, message prefix hash, cache expiry)
- Cache warmth evaluation (should we stick to current model or switch?)
- Model-switch penalty calculation
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from model_plane.config import settings
from model_plane.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class SessionState:
    session_id: str
    provider: str
    model: str  # deployment name
    prefix_hash: str
    context_tokens: int
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    cache_expiry: float = 0.0
    last_compaction: float = 0.0
    model_switches: int = 0

    def is_expired(self) -> bool:
        if self.cache_expiry and time.time() > self.cache_expiry:
            return True
        return False

    def update(self, prefix_hash: str, context_tokens: int) -> None:
        self.prefix_hash = prefix_hash
        self.context_tokens = context_tokens
        self.last_updated = time.time()


def _compute_prefix_hash(messages: list[dict]) -> str:
    """Compute a stable hash of the non-last-turn message prefix."""
    prefix = messages[:-1] if len(messages) > 1 else messages
    serialized = json.dumps(prefix, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:16]


class SessionCache:
    """In-process cache; can be backed by Redis when configured."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._redis_client: Any = None
        self._use_redis = bool(settings.redis_url)
        if self._use_redis:
            self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis

            self._redis_client = redis.from_url(settings.redis_url)
            self._redis_client.ping()
            log.info("session_cache_redis_connected", url=settings.redis_url)
        except Exception as exc:
            log.warning("session_cache_redis_unavailable", error=str(exc))
            self._use_redis = False

    def get(self, session_id: str) -> SessionState | None:
        if self._use_redis and self._redis_client:
            return self._redis_get(session_id)
        return self._sessions.get(session_id)

    def put(self, state: SessionState) -> None:
        if self._use_redis and self._redis_client:
            self._redis_put(state)
        else:
            self._sessions[state.session_id] = state

    def _redis_get(self, session_id: str) -> SessionState | None:
        try:
            raw = self._redis_client.get(f"session:{session_id}")
            if raw:
                data = json.loads(raw)
                return SessionState(**data)
        except Exception as exc:
            log.warning("redis_get_failed", error=str(exc))
        return None

    def _redis_put(self, state: SessionState) -> None:
        try:
            data = json.dumps(state.__dict__)
            self._redis_client.setex(
                f"session:{state.session_id}",
                settings.cache_ttl_seconds,
                data,
            )
        except Exception as exc:
            log.warning("redis_put_failed", error=str(exc))


class CacheAwareRouter:
    """Evaluates whether to stick with the current session model or switch."""

    def __init__(self, cache: SessionCache) -> None:
        self._cache = cache

    def evaluate(
        self,
        session_id: str | None,
        messages: list[dict],
        proposed_deployment: "Any",  # DeploymentConfig
    ) -> tuple[str | None, bool]:
        """
        Returns (recommended_deployment_name | None, cache_warm).
        None means: proceed with proposed_deployment.
        """
        if not session_id:
            return None, False

        state = self._cache.get(session_id)
        if not state or state.is_expired():
            return None, False

        prefix_hash = _compute_prefix_hash(messages)
        cache_warm = state.prefix_hash == prefix_hash

        if not cache_warm:
            return None, False

        # Same provider/model → cache is warm, stay
        if state.model == proposed_deployment.name:
            return state.model, True

        # Different model proposed: calculate switch penalty
        # Simple heuristic: if context_tokens > 2000 tokens cached, prefer staying
        if state.context_tokens > 2000:
            log.debug(
                "cache_continuity_preserved",
                session=session_id,
                anchor=state.model,
                proposed=proposed_deployment.name,
                context_tokens=state.context_tokens,
            )
            return state.model, True

        return None, False

    def update_session(
        self,
        session_id: str,
        deployment_name: str,
        provider: str,
        messages: list[dict],
        context_tokens: int,
    ) -> None:
        prefix_hash = _compute_prefix_hash(messages)
        existing = self._cache.get(session_id)

        if existing:
            model_switched = existing.model != deployment_name
            existing.update(prefix_hash, context_tokens)
            existing.model = deployment_name
            existing.provider = provider
            if model_switched:
                existing.model_switches += 1
            self._cache.put(existing)
        else:
            state = SessionState(
                session_id=session_id,
                provider=provider,
                model=deployment_name,
                prefix_hash=prefix_hash,
                context_tokens=context_tokens,
                cache_expiry=time.time() + settings.cache_ttl_seconds,
            )
            self._cache.put(state)


# ── singletons ────────────────────────────────────────────────────────────────

_session_cache: SessionCache | None = None
_cache_router: CacheAwareRouter | None = None


def get_session_cache() -> SessionCache:
    global _session_cache
    if _session_cache is None:
        _session_cache = SessionCache()
    return _session_cache


def get_cache_router() -> CacheAwareRouter:
    global _cache_router
    if _cache_router is None:
        _cache_router = CacheAwareRouter(get_session_cache())
    return _cache_router
