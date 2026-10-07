"""Compression profiles and processors.

Phase 6:
- pass-through (default, always safe)
- tool_output_compaction
- conversation_summary
- retrieval_selection
- code_context_reduction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from model_plane.config import settings
from model_plane.logging_setup import get_logger

log = get_logger(__name__)

_ENC: tiktoken.Encoding | None = None


def _get_enc() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _token_count(text: str) -> int:
    return len(_get_enc().encode(text))


def _msg_tokens(msg: dict) -> int:
    content = msg.get("content", "") or ""
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return _token_count(content)


@dataclass
class CompressionResult:
    messages: list[dict]
    tokens_before: int
    tokens_after: int
    profile_used: str
    warnings: list[str] = field(default_factory=list)
    cache_prefix_preserved: bool = True

    @property
    def savings(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before


# ── individual processors ──────────────────────────────────────────────────


def passthrough(messages: list[dict], **_kwargs: Any) -> list[dict]:
    return messages


def compact_tool_outputs(messages: list[dict], max_tool_tokens: int = 800) -> list[dict]:
    """Truncate oversized tool result messages."""
    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                result.append(msg)
                continue
            tok = _token_count(content)
            if tok > max_tool_tokens:
                # Try to keep JSON structure intact — truncate the stringified value
                truncated = content[:max_tool_tokens * 4]  # ~4 chars/token heuristic
                msg = {**msg, "content": truncated + "\n[...truncated by model-plane...]"}
                log.debug("tool_output_compacted", original_tokens=tok)
        result.append(msg)
    return result


def summarize_conversation(
    messages: list[dict],
    target_budget: int = 3000,
    keep_last_n: int = 6,
) -> list[dict]:
    """
    Drop middle turns to fit within target budget.
    Always keeps: system message + last N turns.
    Inserts a synthetic assistant summary placeholder.
    """
    total = sum(_msg_tokens(m) for m in messages)
    if total <= target_budget:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    tail = non_system[-keep_last_n:] if len(non_system) > keep_last_n else non_system
    middle = non_system[:-keep_last_n] if len(non_system) > keep_last_n else []

    if not middle:
        # Nothing to drop — return as-is
        return messages

    summary_msg = {
        "role": "assistant",
        "content": (
            "[Context summary: prior conversation omitted to reduce token usage. "
            "Key points have been preserved in subsequent messages.]"
        ),
    }

    compressed = system + [summary_msg] + tail
    log.debug("conversation_summarized", original_tokens=total, compressed_turns=len(compressed))
    return compressed


def reduce_code_context(
    messages: list[dict],
    max_code_block_tokens: int = 1200,
) -> list[dict]:
    """Truncate very large code blocks within message content."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            result.append(msg)
            continue
        # Find code blocks and truncate large ones
        def _truncate_block(match: re.Match) -> str:
            block = match.group(0)
            if _token_count(block) > max_code_block_tokens:
                lines = block.splitlines()
                kept = lines[:40]  # keep first 40 lines
                return "\n".join(kept) + "\n# ... [truncated by model-plane] ...\n```"
            return block

        new_content = re.sub(r"```[\s\S]*?```", _truncate_block, content)
        if new_content != content:
            msg = {**msg, "content": new_content}
            log.debug("code_context_reduced")
        result.append(msg)
    return result


# ── profile dispatcher ──────────────────────────────────────────────────────


PROFILES: dict[str, Any] = {
    "passthrough": passthrough,
    "tool_output_compaction": compact_tool_outputs,
    "conversation_summary": summarize_conversation,
    "code_context_reduction": reduce_code_context,
}


def compress(
    messages: list[dict],
    profile: str,
    context_budget: int | None = None,
    task_type: str | None = None,
    context_reuse_score: float = 0.0,
    **kwargs: Any,
) -> CompressionResult:
    """Apply the named compression profile to messages.

    Cache-aware gate (§16 of research doc):
    If the session has a high context_reuse_score the request's prefix is
    likely already cached on the warm provider.  Compressing it would
    invalidate the cache hit, costing more in re-prefill than it saves in
    reduced input tokens.  When context_reuse_score exceeds
    CACHE_AWARE_COMPRESSION_THRESHOLD we pass through unmodified so the
    prefix cache remains intact.
    """
    from model_plane.runtime_overrides import compression_overrides as _ov

    tokens_before = sum(_msg_tokens(m) for m in messages)

    # Resolve live overrides — runtime_overrides dict wins over env/settings.
    _enabled: bool = bool(_ov.get("enabled", settings.compression_enabled))
    _cache_gate: bool = bool(_ov.get("cache_aware_gate_enabled", settings.context_reuse_enabled))
    _cache_threshold: float = float(_ov.get("cache_aware_threshold", 0.60))

    # Honour profile override from the admin UI (replaces the caller-supplied profile).
    _profile_override = str(_ov.get("profile", "")) if _ov.get("profile") else None
    if _profile_override and _profile_override != "auto":
        profile = _profile_override
    elif _profile_override == "auto" and profile not in ("auto", "passthrough"):
        # Caller already resolved to a concrete profile — keep it.
        pass

    # Honour token-threshold override.
    if "token_threshold" in _ov:
        kwargs.setdefault("target_budget", int(_ov["token_threshold"]))

    # Cache-aware gate: skip compression when the session prefix is warm.
    if (
        _cache_gate
        and context_reuse_score >= _cache_threshold
    ):
        log.debug(
            "compression_skipped_cache_warm",
            context_reuse_score=round(context_reuse_score, 3),
            threshold=_cache_threshold,
        )
        return CompressionResult(
            messages=messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            profile_used="passthrough",
            cache_prefix_preserved=True,
        )

    if not _enabled or profile == "passthrough":
        return CompressionResult(
            messages=messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            profile_used="passthrough",
        )

    # Auto-select profile if not specified
    if profile == "auto":
        profile = _auto_select_profile(messages, task_type)

    processor = PROFILES.get(profile, passthrough)
    if context_budget:
        kwargs["target_budget"] = context_budget

    compressed_messages = processor(messages, **kwargs)
    tokens_after = sum(_msg_tokens(m) for m in compressed_messages)

    return CompressionResult(
        messages=compressed_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        profile_used=profile,
    )


def _auto_select_profile(messages: list[dict], task_type: str | None) -> str:
    """Heuristically pick the best compression profile."""
    has_tool_results = any(m.get("role") == "tool" for m in messages)
    code_heavy = any(
        isinstance(m.get("content", ""), str) and "```" in m.get("content", "")
        for m in messages
    )

    if has_tool_results:
        return "tool_output_compaction"
    if task_type in ("code_generation", "code_editing", "code_debugging", "repository_search") and code_heavy:
        return "code_context_reduction"
    total = sum(_msg_tokens(m) for m in messages)
    if total > settings.compression_token_threshold:
        return "conversation_summary"
    return "passthrough"
