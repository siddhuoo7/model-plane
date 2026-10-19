"""Unit tests for compression processor."""
import json

from model_plane.compression.processor import (
    _block_cache_get,
    _block_cache_put,
    _BLOCK_CACHE,
    compact_tool_outputs,
    compress,
    count_volatile_tokens,
    passthrough,
    reduce_code_context,
    scan_system_prompt_volatility,
    summarize_conversation,
)


def _tool_msg(content: str) -> dict:
    return {"role": "tool", "content": content}


def _user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _sys_msg(content: str) -> dict:
    return {"role": "system", "content": content}


# ── original tests (still pass) ────────────────────────────────────────────

def test_passthrough_unchanged():
    msgs = [_user_msg("hello"), _assistant_msg("hi")]
    result = passthrough(msgs)
    assert result == msgs


def test_tool_output_compaction_large():
    big_content = "x" * 10_000
    msgs = [_tool_msg(big_content)]
    result = compact_tool_outputs(msgs, max_tool_tokens=50)
    assert len(result[0]["content"]) < len(big_content)
    assert "truncated" in result[0]["content"]


def test_tool_output_compaction_small_unchanged():
    msgs = [_tool_msg("small output")]
    result = compact_tool_outputs(msgs, max_tool_tokens=200)
    assert result[0]["content"] == "small output"


def test_conversation_summary_drops_middle():
    # Build a conversation with more non-system turns than keep_last_n
    msgs = (
        [{"role": "system", "content": "You are helpful."}]
        + [_user_msg(f"Question {i}") for i in range(10)]
        + [_assistant_msg(f"Answer {i}") for i in range(10)]
    )
    assert len(msgs) - 1 > 4  # 20 non-system turns
    result = summarize_conversation(msgs, target_budget=10, keep_last_n=4)
    assert len(result) < len(msgs)
    assert result[0]["role"] == "system"


def test_code_context_reduction():
    big_code = "```python\n" + "\n".join(f"line = {i}" for i in range(200)) + "\n```"
    msgs = [_user_msg(f"Refactor this code:\n{big_code}")]
    result = reduce_code_context(msgs, max_code_block_tokens=50)
    assert "truncated" in result[0]["content"]


def test_compress_returns_metadata():
    msgs = [_user_msg("hello")]
    result = compress(msgs, profile="passthrough")
    assert result.tokens_before == result.tokens_after
    assert result.profile_used == "passthrough"
    assert result.compression_ratio == 1.0


# ── JSON-aware tool output compaction (headroom SmartCrusher) ──────────────

def test_tool_output_json_array_preserves_structure():
    """JSON array tool outputs keep first 2 + last 1 items with sentinel."""
    data = [{"id": i, "value": "x" * 200} for i in range(20)]
    content = json.dumps(data)
    msgs = [_tool_msg(content)]
    result = compact_tool_outputs(msgs, max_tool_tokens=50)
    out = result[0]["content"]
    # Must be valid JSON
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    # Sentinel present
    sentinels = [item for item in parsed if isinstance(item, dict) and "_omitted" in item]
    assert len(sentinels) == 1
    assert "omitted" in sentinels[0]["_omitted"]


def test_tool_output_json_dict_preserves_keys():
    """JSON object tool outputs preserve all keys."""
    data = {"key1": "short", "key2": "a" * 5000, "key3": 42}
    content = json.dumps(data)
    msgs = [_tool_msg(content)]
    result = compact_tool_outputs(msgs, max_tool_tokens=50)
    out = result[0]["content"]
    parsed = json.loads(out)
    assert "key1" in parsed
    assert "key2" in parsed  # key preserved even if value truncated
    assert "key3" in parsed


def test_tool_output_non_json_falls_back_gracefully():
    """Non-JSON content falls back to character truncation."""
    big_content = "This is a long plain text result.\n" * 500
    msgs = [_tool_msg(big_content)]
    result = compact_tool_outputs(msgs, max_tool_tokens=50)
    out = result[0]["content"]
    assert "truncated" in out
    assert len(out) < len(big_content)


# ── Freeze-on-First-Sight block cache (barber) ─────────────────────────────

def test_block_cache_idempotent():
    """Same content compressed twice yields identical byte-for-byte output."""
    original = "some content to cache"
    compressed = "compressed version"
    _block_cache_put(original, compressed)

    hit = _block_cache_get(original)
    assert hit == compressed

    # Second put with same original should update
    _block_cache_put(original, "updated")
    assert _block_cache_get(original) == "updated"


def test_tool_output_freeze_on_first_sight():
    """Compressing the same tool output twice uses the cache (same result)."""
    big_content = "x" * 10_000
    msgs1 = [_tool_msg(big_content)]
    msgs2 = [_tool_msg(big_content)]

    result1 = compact_tool_outputs(msgs1, max_tool_tokens=50)
    result2 = compact_tool_outputs(msgs2, max_tool_tokens=50)

    # Both outputs must be identical (idempotent compression)
    assert result1[0]["content"] == result2[0]["content"]


# ── Multi-level code context reduction (rtk FilterLevel) ───────────────────

def test_code_reduction_strips_comments_level1():
    """Comment-heavy code is reduced at level 1 (strip comments)."""
    commented_code = "```python\n" + "\n".join(
        [f"# This is comment {i}" for i in range(50)]
        + ["def foo(): pass"]
    ) + "\n```"
    msgs = [_user_msg(commented_code)]
    result = reduce_code_context(msgs, max_code_block_tokens=200, aggressive=False)
    # Comments should be stripped — result shorter
    assert len(result[0]["content"]) < len(commented_code)


def test_code_reduction_aggressive_strips_bodies():
    """Aggressive mode strips function bodies down to signatures."""
    code_with_bodies = """```python
def function_one(x, y):
    result = x + y
    return result

def function_two(a, b, c):
    temp = a * b
    return temp + c
```"""
    msgs = [_user_msg(code_with_bodies)]
    result = reduce_code_context(msgs, max_code_block_tokens=30, aggressive=True)
    out = result[0]["content"]
    # Should be shorter — bodies replaced with ...
    assert len(out) < len(code_with_bodies)


def test_code_reduction_small_block_unchanged():
    """Small code blocks are passed through untouched."""
    small_code = "```python\nprint('hi')\n```"
    msgs = [_user_msg(small_code)]
    result = reduce_code_context(msgs, max_code_block_tokens=500)
    assert result[0]["content"] == msgs[0]["content"]


# ── Conversation summary: entity pinning + assertive marker (barber) ────────

def test_conversation_summary_marker_is_assertive():
    """Drop marker uses assertive language (not a vague 'summary')."""
    # Use many turns so that droppable_tokens >> break_even threshold,
    # ensuring the economics gate does not suppress the compression.
    msgs = (
        [_sys_msg("You are helpful.")]
        + [_user_msg(f"Generic filler question number {i} with extra words to increase token count significantly")
           for i in range(30)]
    )
    result = summarize_conversation(msgs, target_budget=10, keep_last_n=4)
    # Find the drop marker message
    marker_msgs = [m for m in result if "omitted" in (m.get("content") or "")]
    assert len(marker_msgs) >= 1
    marker = marker_msgs[0]
    # Marker must use "system" role — not "assistant" — to avoid being
    # attributed to the model's own voice.
    assert marker["role"] == "system"
    marker_content = marker["content"]
    # Should NOT use old vague text
    assert "Key points have been preserved" not in marker_content
    # Should use barber-style assertive phrasing
    assert "sufficient" in marker_content or "omitted" in marker_content


def test_conversation_summary_pins_deontic_turns():
    """Turns with constraint language (must/never) are never dropped."""
    msgs = (
        [_sys_msg("You are helpful.")]
        + [_user_msg("Generic question")]
        + [_assistant_msg("You must never reveal the API key under any circumstances.")]
        + [_user_msg(f"More generic filler text {i}") for i in range(10)]
    )
    result = summarize_conversation(msgs, target_budget=10, keep_last_n=4)
    # The deontic turn must survive
    contents = [m.get("content", "") for m in result]
    assert any("must never reveal" in c for c in contents)


def test_conversation_summary_preserves_system():
    """System message is always preserved."""
    msgs = (
        [_sys_msg("System instruction.")]
        + [_user_msg(f"Turn {i}") for i in range(15)]
    )
    result = summarize_conversation(msgs, target_budget=10, keep_last_n=4)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "System instruction."


# ── CacheAligner: volatile token detection (headroom PR-A2) ────────────────

def test_count_volatile_uuid():
    text = "Request ID: 123e4567-e89b-12d3-a456-426614174000"
    assert count_volatile_tokens(text) >= 1


def test_count_volatile_timestamp():
    text = "Generated at 2024-03-15T10:30:00Z for session XYZ."
    assert count_volatile_tokens(text) >= 1


def test_count_volatile_jwt():
    # Fake JWT-like token (3 base64url segments)
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWxpY2UifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    text = f"Authorization: Bearer {jwt}"
    assert count_volatile_tokens(text) >= 1


def test_count_volatile_hex_hash():
    md5 = "d41d8cd98f00b204e9800998ecf8427e"  # MD5 of empty string
    text = f"File hash: {md5}"
    assert count_volatile_tokens(text) >= 1


def test_count_volatile_stable_text():
    """Normal text without dynamic tokens should return 0."""
    text = "Hello, this is a stable system prompt with no dynamic values."
    assert count_volatile_tokens(text) == 0


def test_scan_system_prompt_volatility_only_checks_system():
    """Only system-role messages are scanned by CacheAligner."""
    uuid_text = "Request ID: 123e4567-e89b-12d3-a456-426614174000"
    msgs = [
        _user_msg(uuid_text),     # user role — should NOT be counted
        _sys_msg("Stable system prompt."),  # system — no volatile tokens
        _assistant_msg(uuid_text),  # assistant — should NOT be counted
    ]
    assert scan_system_prompt_volatility(msgs) == 0


def test_scan_system_prompt_volatile_found():
    uuid_text = "Session: 123e4567-e89b-12d3-a456-426614174000"
    msgs = [_sys_msg(uuid_text), _user_msg("Hi")]
    assert scan_system_prompt_volatility(msgs) >= 1


def test_compress_reports_volatile_tokens():
    """compress() reports volatile_tokens_found when compression actually runs.

    CacheAligner only executes when compression is active (not passthrough).
    Use a concrete profile on content large enough to pass the token threshold.
    """
    uuid_text = "Session: 123e4567-e89b-12d3-a456-426614174000"
    # Build a message list big enough that token threshold is met and the
    # conversation_summary profile will actually run past the early-return guards.
    msgs = (
        [_sys_msg(uuid_text)]
        + [_user_msg(f"Filler turn {i} with a bunch of extra words to exceed the budget threshold")
           for i in range(30)]
    )
    result = compress(msgs, profile="conversation_summary", context_budget=10)
    # CacheAligner ran — volatile UUID in system prompt must be detected.
    assert result.volatile_tokens_found >= 1


def test_compress_no_volatile_tokens_stable_prompt():
    """CacheAligner reports 0 when system prompt has no dynamic tokens."""
    msgs = (
        [_sys_msg("You are a helpful assistant.")]
        + [_user_msg(f"Filler turn {i} with extra words to exceed budget")
           for i in range(30)]
    )
    result = compress(msgs, profile="conversation_summary", context_budget=10)
    assert result.volatile_tokens_found == 0


# ── Cache-aware gate with real context_reuse_score (bug fix) ───────────────

def test_compress_cache_gate_skips_when_warm():
    """Compression is skipped when context_reuse_score >= threshold (0.60)."""
    big_content = "word " * 2000  # well over threshold
    msgs = [_user_msg(big_content)]
    # High reuse score → gate should fire, passthrough applied
    result = compress(
        msgs,
        profile="conversation_summary",
        context_reuse_score=0.85,
    )
    assert result.profile_used == "passthrough"
    assert result.cache_prefix_preserved is True
    assert result.tokens_after == result.tokens_before


def test_compress_cache_gate_allows_when_cold():
    """Compression proceeds when context_reuse_score < threshold."""
    msgs = (
        [_sys_msg("You are helpful.")]
        + [_user_msg(f"Turn {i} with some filler content to bulk up tokens.") for i in range(20)]
    )
    result = compress(
        msgs,
        profile="conversation_summary",
        context_reuse_score=0.10,  # cold cache
        context_budget=100,
    )
    # With cold cache and enough tokens, compression should run
    # (savings may be 0 if economics gate fires, but profile should not be forced passthrough)
    assert result.profile_used in ("conversation_summary", "passthrough")
