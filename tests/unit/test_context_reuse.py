"""Unit tests for model_plane/routing/context_reuse.py.

Covers:
  - CacheCapability enum values
  - score_context_reuse: no session → 0.0
  - score_context_reuse: expired session → 0.0
  - score_context_reuse: warm session + matching prefix → score ≥ 0.50
  - score_context_reuse: PREFIX capability adds signal
  - score_context_reuse: KV_EVENTS capability adds maximum signal
  - score_context_reuse: different deployment (not warm model) gets half signal
  - score_context_reuse: score always in [0.0, 1.0]
  - score_context_reuse: fail-safe on exception returns 0.0
"""

from __future__ import annotations

import time

import pytest

from model_plane.routing.context_reuse import CacheCapability, score_context_reuse


# ── helpers ───────────────────────────────────────────────────────────────────

def _dep(name: str, cache_capability: str = "none") -> object:
    class _Dep:
        pass
    d = _Dep()
    d.name = name
    d.cache_capability = cache_capability
    return d


def _ctx(prefix_hash: str = "") -> object:
    class _Ctx:
        pass
    c = _Ctx()
    c.prefix_hash = prefix_hash
    return c


def _session(
    model: str = "dep-a",
    prefix_hash: str = "abc123",
    accumulated_tokens: int = 3000,
    model_switches: int = 0,
    warm_model_ids: set | None = None,
    last_updated: float | None = None,
    cache_expiry: float = 0.0,
) -> object:
    class _Session:
        def is_expired(self):
            if self.cache_expiry and time.time() > self.cache_expiry:
                return True
            return False
    s = _Session()
    s.model = model
    s.prefix_hash = prefix_hash
    s.accumulated_tokens = accumulated_tokens
    s.model_switches = model_switches
    s.warm_model_ids = warm_model_ids or {model}
    s.last_updated = last_updated or time.time()
    s.cache_expiry = cache_expiry
    return s


# ── CacheCapability ───────────────────────────────────────────────────────────

class TestCacheCapability:
    def test_values(self):
        assert CacheCapability.NONE.value == "none"
        assert CacheCapability.RESPONSE.value == "response"
        assert CacheCapability.PREFIX.value == "prefix"
        assert CacheCapability.KV_EVENTS.value == "kv_events"

    def test_from_string(self):
        assert CacheCapability("prefix") == CacheCapability.PREFIX


# ── score_context_reuse ───────────────────────────────────────────────────────

class TestScoreContextReuse:

    def test_no_session_returns_zero(self):
        dep = _dep("dep-a")
        ctx = _ctx("abc123")
        score = score_context_reuse(None, dep, ctx)
        assert score == 0.0

    def test_expired_session_returns_zero(self):
        sess = _session(cache_expiry=time.time() - 1)  # already expired
        dep = _dep("dep-a")
        ctx = _ctx("abc123")
        score = score_context_reuse(sess, dep, ctx)
        assert score == 0.0

    def test_warm_session_matching_prefix_high_score(self):
        sess = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=5000)
        dep = _dep("dep-a", cache_capability="prefix")
        ctx = _ctx("abc123")
        score = score_context_reuse(sess, dep, ctx, token_threshold=2000)
        assert score >= 0.50

    def test_score_always_in_range(self):
        sess = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=500)
        dep = _dep("dep-a")
        ctx = _ctx("abc123")
        score = score_context_reuse(sess, dep, ctx)
        assert 0.0 <= score <= 1.0

    def test_prefix_capability_increases_score(self):
        sess = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=3000)
        ctx = _ctx("abc123")

        dep_none   = _dep("dep-a", cache_capability="none")
        dep_prefix = _dep("dep-a", cache_capability="prefix")

        score_none   = score_context_reuse(sess, dep_none, ctx, token_threshold=2000)
        score_prefix = score_context_reuse(sess, dep_prefix, ctx, token_threshold=2000)

        assert score_prefix > score_none

    def test_kv_events_capability_highest(self):
        sess = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=3000)
        ctx = _ctx("abc123")

        dep_prefix = _dep("dep-a", cache_capability="prefix")
        dep_kv     = _dep("dep-a", cache_capability="kv_events")

        score_prefix = score_context_reuse(sess, dep_prefix, ctx, token_threshold=2000)
        score_kv     = score_context_reuse(sess, dep_kv, ctx, token_threshold=2000)

        assert score_kv >= score_prefix

    def test_different_deployment_gets_partial_signal(self):
        """A deployment that is not the warm model should score lower than the warm one."""
        sess = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=3000)
        ctx = _ctx("abc123")

        dep_warm  = _dep("dep-a", cache_capability="prefix")
        dep_other = _dep("dep-b", cache_capability="prefix")

        score_warm  = score_context_reuse(sess, dep_warm, ctx, token_threshold=2000)
        score_other = score_context_reuse(sess, dep_other, ctx, token_threshold=2000)

        assert score_warm > score_other

    def test_prefix_mismatch_lowers_score(self):
        sess = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=3000)
        dep = _dep("dep-a", cache_capability="prefix")

        ctx_match    = _ctx("abc123")
        ctx_mismatch = _ctx("different_hash")

        score_match    = score_context_reuse(sess, dep, ctx_match, token_threshold=2000)
        score_mismatch = score_context_reuse(sess, dep, ctx_mismatch, token_threshold=2000)

        assert score_match > score_mismatch

    def test_model_switches_lowers_conv_continuity(self):
        sess_clean    = _session(model="dep-a", model_switches=0, accumulated_tokens=3000)
        sess_switched = _session(model="dep-a", model_switches=2, accumulated_tokens=3000)
        dep = _dep("dep-a")
        ctx = _ctx("abc123")

        score_clean    = score_context_reuse(sess_clean, dep, ctx, token_threshold=2000)
        score_switched = score_context_reuse(sess_switched, dep, ctx, token_threshold=2000)

        assert score_clean > score_switched

    def test_low_tokens_lower_cache_value(self):
        sess_few  = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=100)
        sess_many = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=5000)
        dep = _dep("dep-a")
        ctx = _ctx("abc123")

        score_few  = score_context_reuse(sess_few, dep, ctx, token_threshold=2000)
        score_many = score_context_reuse(sess_many, dep, ctx, token_threshold=2000)

        assert score_many > score_few

    def test_fail_safe_returns_zero(self, monkeypatch):
        """Exception inside scoring must return 0.0, never propagate."""
        import model_plane.routing.context_reuse as cr_mod
        monkeypatch.setattr(cr_mod, "_score_impl", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

        sess = _session()
        dep = _dep("dep-a")
        ctx = _ctx("abc123")
        assert score_context_reuse(sess, dep, ctx) == 0.0

    def test_unknown_cache_capability_treated_as_none(self):
        sess = _session(model="dep-a", prefix_hash="abc123", accumulated_tokens=3000)
        dep = _dep("dep-a", cache_capability="unknown_future_value")
        ctx = _ctx("abc123")
        # Should not raise
        score = score_context_reuse(sess, dep, ctx, token_threshold=2000)
        assert 0.0 <= score <= 1.0
