"""Context reuse scoring for cache-aware routing.

Implements Stage 6e (soft re-rank) from the cache-aware routing plan.

Design (§7, §9, §10, §11 of research doc):
  The ContextReuseScore is a normalised [0, 1] value that estimates how much
  benefit a candidate deployment offers by preserving existing KV-cache /
  prefix-cache locality for the current session.

  It is a SOFT signal — it adds to the candidate sort key but cannot override
  the security hard-filter or force a routing decision on its own.

  CacheCapability abstraction (§10):
    The same gateway must work with providers that expose zero cache visibility
    (OpenAI, Anthropic) and providers with full prefix-cache or KV-event
    visibility (vLLM, Dynamo).  Rather than requiring all providers to expose
    the same interface, each deployment declares its cache capability tier.
    The scoring formula gives progressively higher ProviderCacheSignal to
    deployments with more visibility.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING

from model_plane.logging_setup import get_logger

if TYPE_CHECKING:
    from model_plane.cache.session import SessionState
    from model_plane.registry.catalog import DeploymentConfig
    from model_plane.routing.context import RoutingContext

log = get_logger(__name__)


# ── Cache capability abstraction (§10) ───────────────────────────────────────

class CacheCapability(str, Enum):
    NONE      = "none"       # opaque provider — no cache visibility at all
    RESPONSE  = "response"   # exact / semantic response cache available
    PREFIX    = "prefix"     # prefix/prompt cache (vLLM, Anthropic prompt cache)
    KV_EVENTS = "kv_events"  # full KV event stream (vLLM with Dynamo integration)


# ProviderCacheSignal weight per capability level
_CACHE_SIGNAL_WEIGHT: dict[CacheCapability, float] = {
    CacheCapability.NONE:      0.0,
    CacheCapability.RESPONSE:  0.25,
    CacheCapability.PREFIX:    0.70,
    CacheCapability.KV_EVENTS: 1.0,
}

# Default token threshold for CacheValueWeight component.
# Overridden by settings.context_reuse_token_threshold when available.
_DEFAULT_TOKEN_THRESHOLD = 2000


# ── ContextReuseScore formula weights (§11) ──────────────────────────────────

_W_SESSION_CONTINUITY       = 0.25
_W_PREFIX_SIMILARITY        = 0.25
_W_CONVERSATION_CONTINUITY  = 0.15
_W_PROVIDER_CACHE_SIGNAL    = 0.10
_W_CACHE_VALUE              = 0.20
_W_CACHE_FRESHNESS          = 0.05


def _provider_cache_signal(dep: "DeploymentConfig") -> float:
    """Return normalised [0, 1] cache-signal weight for *dep*."""
    cap_str: str = getattr(dep, "cache_capability", "none") or "none"
    try:
        cap = CacheCapability(cap_str)
    except ValueError:
        cap = CacheCapability.NONE
    return _CACHE_SIGNAL_WEIGHT[cap]


def _cache_freshness(session_state: "SessionState", ttl_seconds: int) -> float:
    """Return 1.0 for a brand-new session, decaying to 0.0 at TTL."""
    if ttl_seconds <= 0:
        return 1.0
    elapsed = time.time() - session_state.last_updated
    return max(0.0, 1.0 - elapsed / ttl_seconds)


def score_context_reuse(
    session_state: "SessionState | None",
    dep: "DeploymentConfig",
    ctx: "RoutingContext",
    *,
    token_threshold: int = _DEFAULT_TOKEN_THRESHOLD,
    ttl_seconds: int = 3600,
) -> float:
    """Compute a normalised context-reuse score for *dep* given the current session.

    Returns a float in [0.0, 1.0].  Higher means more expected cache / context
    benefit from routing to this deployment.

    A score of 0.0 means: no session, or session expired, or prefix mismatch.
    A score of 1.0 would require: active session, matching prefix, large
    accumulated token history, and a deployment with full KV-event visibility.

    Args:
        session_state: Current session state from SessionCache, or None.
        dep: Candidate deployment being evaluated.
        ctx: Current RoutingContext (used for prefix_hash).
        token_threshold: Tokens above which CacheValueWeight saturates at 1.0.
        ttl_seconds: Session TTL used to compute CacheFreshness.

    Fail-safe: any exception returns 0.0 — the candidate is not penalised,
    it just receives no cache-reuse bonus.
    """
    try:
        return _score_impl(session_state, dep, ctx, token_threshold, ttl_seconds)
    except Exception as exc:
        log.debug("context_reuse_score_failed", deployment=dep.name, error=str(exc))
        return 0.0


def _score_impl(
    session_state: "SessionState | None",
    dep: "DeploymentConfig",
    ctx: "RoutingContext",
    token_threshold: int,
    ttl_seconds: int,
) -> float:
    # ── SessionContinuity ─────────────────────────────────────────────────────
    # 1.0 if session is active and not expired; 0.0 otherwise.
    if session_state is None or session_state.is_expired():
        return 0.0  # early-exit: no session → all components are 0
    session_continuity = 1.0

    # ── PrefixSimilarity ──────────────────────────────────────────────────────
    # 1.0 if the current request prefix hash matches the stored hash (cache hit
    # is likely); 0.5 if this candidate was previously warm (partial credit);
    # 0.0 otherwise.
    current_hash = ctx.prefix_hash or ""
    if current_hash and session_state.prefix_hash == current_hash:
        prefix_similarity = 1.0
    elif dep.name in (session_state.warm_model_ids or set()):
        prefix_similarity = 0.5
    else:
        prefix_similarity = 0.0

    # ── ConversationContinuity ────────────────────────────────────────────────
    # Proxy: 1.0 if the session has had zero model switches (same model all
    # along), 0.5 if it has had exactly one switch, 0.0 for two or more.
    switches = getattr(session_state, "model_switches", 0)
    if switches == 0:
        conv_continuity = 1.0
    elif switches == 1:
        conv_continuity = 0.5
    else:
        conv_continuity = 0.0

    # ── ProviderCacheSignal ───────────────────────────────────────────────────
    # Higher weight for deployments with richer cache-visibility capability.
    # Only counts if the candidate is the session's current warm model.
    if dep.name == session_state.model:
        provider_cache_signal = _provider_cache_signal(dep)
    else:
        # A different deployment gets half the signal — it could share the
        # same provider's prefix cache but we have no guarantee.
        provider_cache_signal = _provider_cache_signal(dep) * 0.5

    # ── CacheValueWeight ──────────────────────────────────────────────────────
    # Grows with accumulated_tokens up to token_threshold, then saturates.
    accumulated = getattr(session_state, "accumulated_tokens", 0)
    cache_value = min(1.0, accumulated / max(token_threshold, 1))

    # ── CacheFreshness ────────────────────────────────────────────────────────
    freshness = _cache_freshness(session_state, ttl_seconds)

    score = (
        _W_SESSION_CONTINUITY      * session_continuity
        + _W_PREFIX_SIMILARITY     * prefix_similarity
        + _W_CONVERSATION_CONTINUITY * conv_continuity
        + _W_PROVIDER_CACHE_SIGNAL * provider_cache_signal
        + _W_CACHE_VALUE           * cache_value
        + _W_CACHE_FRESHNESS       * freshness
    )

    # Clamp to [0, 1] in case of floating-point noise
    return max(0.0, min(1.0, score))
