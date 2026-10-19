"""Compression profiles and processors.

Profiles:
- passthrough          — no-op
- tool_output_compaction — JSON-aware truncation of oversized tool results
- conversation_summary — drops middle turns with entity pinning and economics gate
- code_context_reduction — multi-level code block reduction (comments → signatures → truncate)
- auto                 — heuristic dispatcher
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
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


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


@dataclass
class CompressionResult:
    messages: list[dict]
    tokens_before: int
    tokens_after: int
    profile_used: str
    warnings: list[str] = field(default_factory=list)
    cache_prefix_preserved: bool = True
    volatile_tokens_found: int = 0  # CacheAligner finding

    @property
    def savings(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before


# ── Block cache — content_hash → compressed string ─────────────────────────
# Keeps compression decisions byte-identical across turns so the provider
# KV-cache prefix is never invalidated by non-deterministic re-compression.
# Uses an OrderedDict so get() promotes the entry to MRU position and eviction
# removes the LRU end — O(1) per operation.

from collections import OrderedDict as _OrderedDict

_BLOCK_CACHE: "_OrderedDict[str, str]" = _OrderedDict()
_BLOCK_CACHE_MAX = 4096
_BLOCK_CACHE_LOCK = threading.Lock()


def _block_cache_get(content: str) -> str | None:
    key = _content_hash(content)
    with _BLOCK_CACHE_LOCK:
        if key not in _BLOCK_CACHE:
            return None
        _BLOCK_CACHE.move_to_end(key)   # promote to MRU
        return _BLOCK_CACHE[key]


def _block_cache_put(original: str, compressed: str) -> None:
    key = _content_hash(original)
    with _BLOCK_CACHE_LOCK:
        if key in _BLOCK_CACHE:
            _BLOCK_CACHE[key] = compressed       # update value
            _BLOCK_CACHE.move_to_end(key)        # promote to MRU
        else:
            if len(_BLOCK_CACHE) >= _BLOCK_CACHE_MAX:
                _BLOCK_CACHE.popitem(last=False)  # evict LRU
            _BLOCK_CACHE[key] = compressed


# ── Volatile-token patterns for system-prompt KV-cache analysis ────────────

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
)
_JWT_RE = re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\b")
_HEXHASH_RE = re.compile(r"\b([0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64})\b", re.IGNORECASE)


def count_volatile_tokens(text: str) -> int:
    """Count dynamic rotating tokens (UUIDs, timestamps, JWTs, hex hashes) in text.

    Stdlib parsing validates regex matches to reduce false positives.
    Returns 0 for stable prompts, > 0 when dynamic values are detected.
    """
    count = 0

    for m in _UUID_RE.finditer(text):
        try:
            uuid.UUID(m.group(0))
            count += 1
        except ValueError:
            pass

    for m in _ISO_DATE_RE.finditer(text):
        try:
            datetime.fromisoformat(m.group(0).rstrip("Z"))
            count += 1
        except ValueError:
            pass

    count += len(_JWT_RE.findall(text))
    count += len(_HEXHASH_RE.findall(text))

    return count


# ── individual processors ──────────────────────────────────────────────────


def passthrough(messages: list[dict], **_kwargs: Any) -> list[dict]:
    return messages


def _try_json_truncate(content: str, max_tokens: int) -> str:
    """Truncate JSON content to max_tokens while preserving structural validity.

    Arrays: keeps head + tail items with an _omitted sentinel in the middle.
    Dicts:  keeps all keys, truncates long string values.
    Other:  falls back to character truncation at a newline boundary.
    """
    stripped = content.strip()
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        truncated = content[:max_chars]
        last_nl = truncated.rfind("\n")
        if last_nl > max_chars * 0.7:
            truncated = truncated[:last_nl]
        return truncated + "\n[...truncated by model-plane...]"

    if isinstance(data, list) and len(data) > 4:
        head = data[:2]
        tail = data[-1:]
        omitted = len(data) - len(head) - len(tail)
        sentinel = {"_omitted": f"{omitted} items omitted to reduce token usage"}
        compressed = json.dumps(head + [sentinel] + tail, ensure_ascii=False, separators=(",", ":"))
        if _token_count(compressed) <= max_tokens:
            return compressed
        compressed = json.dumps([data[0], sentinel], ensure_ascii=False, separators=(",", ":"))
        return compressed

    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str) and _token_count(v) > max_tokens // 4:
                v = v[: max_tokens * 2] + "...[truncated]"
            out[k] = v
        serialised = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
        if _token_count(serialised) <= max_tokens:
            return serialised
        max_chars = max_tokens * 4
        return serialised[:max_chars] + "\n[...truncated by model-plane...]"

    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    max_chars = max_tokens * 4
    if len(raw) > max_chars:
        return raw[:max_chars] + "\n[...truncated by model-plane...]"
    return raw


def compact_tool_outputs(messages: list[dict], max_tool_tokens: int = 800) -> list[dict]:
    """Truncate oversized tool results to max_tool_tokens using JSON-aware truncation."""
    result = []
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                result.append(msg)
                continue
            tok = _token_count(content)
            if tok > max_tool_tokens:
                cached = _block_cache_get(content)
                if cached is not None:
                    truncated = cached
                    log.debug("tool_output_compacted", original_tokens=tok, method="block_cache_hit")
                else:
                    truncated = _try_json_truncate(content, max_tool_tokens)
                    _block_cache_put(content, truncated)
                    log.debug("tool_output_compacted", original_tokens=tok, method="json_aware")
                msg = {**msg, "content": truncated}
        result.append(msg)
    return result


# ── Code context reduction ──────────────────────────────────────────────────

_COMMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*#.*$", re.MULTILINE),            # Python / Shell / Ruby
    re.compile(r"^\s*//.*$", re.MULTILINE),           # JS / TS / Go / Rust / Java / C
    re.compile(r"/\*[\s\S]*?\*/", re.MULTILINE),       # C-style block comments
    re.compile(r"^\s*--.*$", re.MULTILINE),            # SQL / Haskell
]

# Matches Python `def` signatures and their indented bodies.
_PY_BODY_RE = re.compile(
    r"(def\s+\w+\s*\([^)]*\)\s*(?:->[^:\n]+)?:)(\s*\n)((?:[ \t]+.*\n?)*)",
    re.MULTILINE,
)


def _strip_comments(code: str) -> str:
    for pat in _COMMENT_PATTERNS:
        code = pat.sub("", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def _py_stub(m: re.Match) -> str:
    return m.group(1) + m.group(2) + "    ...\n"


def _extract_signatures(code: str) -> str:
    """Replace Python function bodies with `...`, keeping signatures only.

    Only Python `def` bodies are handled; brace-language body extraction
    requires balanced-brace parsing that regex cannot safely provide.
    Falls back to comment-stripped output if no defs are matched.
    """
    result = _PY_BODY_RE.sub(_py_stub, code)
    return result.strip() or _strip_comments(code)


def _reduce_code_block(block: str, max_tokens: int, aggressive: bool = True) -> str:
    """Reduce a fenced code block to fit within max_tokens.

    Tries three levels in order:
      1. Strip comments
      2. Strip Python function bodies (aggressive=True only)
      3. Keep first 40 lines
    """
    cached = _block_cache_get(block)
    if cached is not None:
        return cached

    if _token_count(block) <= max_tokens:
        return block

    header_match = re.match(r"(```\w*\n?)", block)
    header = header_match.group(1) if header_match else "```\n"
    inner = block[len(header):]
    if inner.endswith("```"):
        inner = inner[:-3]

    candidate = header + _strip_comments(inner) + "\n```"
    if _token_count(candidate) <= max_tokens:
        _block_cache_put(block, candidate)
        return candidate

    if aggressive:
        candidate = header + _extract_signatures(inner) + "\n```"
        if _token_count(candidate) <= max_tokens:
            _block_cache_put(block, candidate)
            return candidate

    result = "\n".join(block.splitlines()[:40]) + "\n# ... [truncated by model-plane] ...\n```"
    _block_cache_put(block, result)
    return result


def reduce_code_context(
    messages: list[dict],
    max_code_block_tokens: int = 1200,
    aggressive: bool = True,
) -> list[dict]:
    """Reduce fenced code blocks in messages that exceed max_code_block_tokens."""
    def _process_block(match: re.Match) -> str:
        return _reduce_code_block(match.group(0), max_code_block_tokens, aggressive)

    result = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            result.append(msg)
            continue

        new_content = re.sub(r"```[\s\S]*?```", _process_block, content)
        if new_content != content:
            msg = {**msg, "content": new_content}
            log.debug("code_context_reduced")
        result.append(msg)
    return result


# ── Conversation summary ────────────────────────────────────────────────────

# Turns matching this pattern are pinned and never dropped.
_DEONTIC_RE = re.compile(
    r"\b(must|never|required|always|forbidden|prohibited|mandatory|constraint)\b",
    re.IGNORECASE,
)

_DROP_MARKER = (
    "[… {n} prior turn(s) omitted as not relevant to current task — "
    "the remaining context is sufficient to continue …]"
)


def _tokenize_simple(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", text)}


def _is_pinned(msg: dict, rare_entities: set[str]) -> bool:
    """Return True if this message must not be dropped.

    Pins: system messages, turns with constraint language, turns mentioning
    rare entities (words that appear in ≤ 2 non-system turns).
    """
    role = msg.get("role", "")
    if role == "system":
        return True

    content = msg.get("content", "")
    if not isinstance(content, str):
        return False

    if _DEONTIC_RE.search(content):
        return True

    words = _tokenize_simple(content)
    if words & rare_entities:
        return True

    return False


def _find_rare_entities(messages: list[dict], max_df: int = 2) -> set[str]:
    """Return words appearing in ≤ max_df non-system turns.

    System turns are excluded so common vocabulary in the system prompt
    does not incorrectly pin unrelated conversation turns.
    """
    word_df: Counter[str] = Counter()
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", "") or ""
        if isinstance(content, str):
            words = _tokenize_simple(content)
            for w in words:
                word_df[w] += 1

    return {w for w, df in word_df.items() if 1 <= df <= max_df}


def summarize_conversation(
    messages: list[dict],
    target_budget: int = 3000,
    keep_last_n: int = 6,
) -> list[dict]:
    """Drop middle turns to fit within target_budget.

    Pinned turns (system, deontic language, rare entities) are never dropped.
    Only compresses when savings exceed the prompt-cache re-priming break-even.
    """
    total = sum(_msg_tokens(m) for m in messages)
    if total <= target_budget:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= keep_last_n:
        return messages

    tail = non_system[-keep_last_n:]
    candidate_middle = non_system[:-keep_last_n]

    if not candidate_middle:
        return messages

    rare_entities = _find_rare_entities(messages)

    pinned: list[dict] = []
    droppable: list[dict] = []
    for msg in candidate_middle:
        if _is_pinned(msg, rare_entities):
            pinned.append(msg)
        else:
            droppable.append(msg)

    if not droppable:
        return messages

    # Skip compression when savings do not justify the KV-cache re-priming cost.
    # Break-even: savings_fraction > 1.15 / (1.15 + 0.10 * remaining_turns)
    remaining_turns = len(tail) + len(pinned)
    break_even = 1.15 / (1.15 + 0.10 * max(remaining_turns, 1))
    droppable_tokens = sum(_msg_tokens(m) for m in droppable)
    savings_fraction = droppable_tokens / max(total, 1)
    if savings_fraction < break_even:
        log.debug(
            "conversation_summary_skipped_economics",
            savings_fraction=round(savings_fraction, 3),
            break_even=round(break_even, 3),
        )
        return messages

    n_dropped = len(droppable)
    marker_msg = {
        "role": "system",
        "content": _DROP_MARKER.format(n=n_dropped),
    }

    compressed = system + pinned + [marker_msg] + tail
    log.debug(
        "conversation_summarized",
        original_tokens=total,
        dropped_turns=n_dropped,
        pinned_turns=len(pinned),
        compressed_turns=len(compressed),
    )
    return compressed


def scan_system_prompt_volatility(messages: list[dict]) -> int:
    """Count volatile tokens (UUIDs, timestamps, JWTs, hashes) in system prompts.

    Returns 0 for stable prompts. A positive value means the system prompt
    contains dynamic values that rotate per-request and bust the KV-cache prefix.
    """
    total = 0
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "") or ""
        if isinstance(content, str):
            total += count_volatile_tokens(content)
    return total


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
    """Apply a compression profile to messages.

    Skips compression and returns passthrough when context_reuse_score
    meets the cache-aware threshold (warm prefix cache would be invalidated).
    volatile_tokens_found reports rotating dynamic values in system prompts
    that bust the provider KV-cache.
    """
    from model_plane.runtime_overrides import compression_overrides as _ov

    tokens_before = sum(_msg_tokens(m) for m in messages)

    _enabled: bool = bool(_ov.get("enabled", settings.compression_enabled))
    _cache_gate: bool = bool(_ov.get("cache_aware_gate_enabled", settings.context_reuse_enabled))
    _cache_threshold: float = float(_ov.get("cache_aware_threshold", 0.60))

    _profile_override: str | None = _ov.get("profile") or None
    if _profile_override and _profile_override not in ("auto", ""):
        profile = _profile_override

    if "token_threshold" in _ov:
        kwargs.setdefault("target_budget", int(_ov["token_threshold"]))

    if _cache_gate and context_reuse_score >= _cache_threshold:
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
            volatile_tokens_found=0,
        )

    if not _enabled or profile == "passthrough":
        return CompressionResult(
            messages=messages,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            profile_used="passthrough",
            volatile_tokens_found=0,
        )

    volatile_count = scan_system_prompt_volatility(messages)
    if volatile_count > 0:
        log.debug("cache_aligner_volatile_tokens", count=volatile_count)

    if profile == "auto":
        profile = _auto_select_profile(messages, task_type, tokens_before)

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
        volatile_tokens_found=volatile_count,
    )


def _auto_select_profile(
    messages: list[dict],
    task_type: str | None,
    tokens_before: int,
) -> str:
    """Select the most appropriate compression profile for the request."""
    has_tool_results = any(m.get("role") == "tool" for m in messages)
    code_heavy = any(
        isinstance(m.get("content", ""), str) and "```" in m.get("content", "")
        for m in messages
    )

    if has_tool_results:
        return "tool_output_compaction"
    if task_type in ("code_generation", "code_editing", "code_debugging", "repository_search") and code_heavy:
        return "code_context_reduction"
    if tokens_before > settings.compression_token_threshold:
        return "conversation_summary"
    return "passthrough"
