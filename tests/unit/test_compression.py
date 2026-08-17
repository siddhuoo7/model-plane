"""Unit tests for compression processor."""
from model_plane.compression.processor import (
    compact_tool_outputs,
    compress,
    passthrough,
    reduce_code_context,
    summarize_conversation,
)


def _tool_msg(content: str) -> dict:
    return {"role": "tool", "content": content}


def _user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


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
    # keep_last_n=4 means we need > 4 non-system turns to drop middle
    assert len(msgs) - 1 > 4  # 20 non-system turns
    # Force token budget very low so summary is triggered
    result = summarize_conversation(msgs, target_budget=10, keep_last_n=4)
    # Should be shorter than original
    assert len(result) < len(msgs)
    # System message preserved
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
