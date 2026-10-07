"""Request feature extraction — converts raw messages into a named feature vector."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tiktoken

if TYPE_CHECKING:
    from model_plane.classifier.taxonomy import TaskType


@dataclass
class RequestFeatures:
    """Named feature vector extracted from a chat-completion request."""

    # text stats
    total_tokens: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    system_present: bool = False
    last_user_text: str = ""
    full_text: str = ""

    # 14-dimension classifier signals
    reasoning_markers: float = 0.0
    code_presence: float = 0.0
    simple_indicators: float = 0.0
    multi_step_patterns: float = 0.0
    technical_terms: float = 0.0
    token_count_signal: float = 0.0
    creative_markers: float = 0.0
    question_complexity: float = 0.0
    constraint_count: float = 0.0
    imperative_verbs: float = 0.0
    output_format: float = 0.0
    domain_specificity: float = 0.0
    reference_complexity: float = 0.0
    negation_complexity: float = 0.0

    # derived
    has_tools: bool = False
    tool_count: int = 0
    has_images: bool = False
    language_hint: str | None = None

    def as_vector(self, task_type: "TaskType | None" = None) -> list[float]:
        """Return the feature vector for ML models.

        Returns 14 floats by default.  When ``settings.features_15_dim_enabled``
        is ``True`` and *task_type* is provided, a 15th dimension
        (``task_type_tier_signal``) is appended.
        """
        vec = [
            self.reasoning_markers,
            self.code_presence,
            self.simple_indicators,
            self.multi_step_patterns,
            self.technical_terms,
            self.token_count_signal,
            self.creative_markers,
            self.question_complexity,
            self.constraint_count,
            self.imperative_verbs,
            self.output_format,
            self.domain_specificity,
            self.reference_complexity,
            self.negation_complexity,
        ]
        if task_type is not None:
            from model_plane.config import settings  # local import avoids circular dep
            if settings.features_15_dim_enabled:
                vec.append(_get_task_tier_signal().get(task_type, 0.3))
        return vec


# ── 15th-dimension task-tier signal ──────────────────────────────────────────
# Maps each TaskType to a float in [0, 1] representing how much compute the
# task typically demands.  Used as dim[14] when FEATURES_15_DIM_ENABLED=true.
# Values from docs/03-migration-guide.md §2a.

def _build_task_tier_signal() -> "dict[TaskType, float]":
    from model_plane.classifier.taxonomy import TaskType  # deferred to avoid circular import
    return {
        TaskType.SIMPLE_QA:               0.0,
        TaskType.SUMMARIZATION:           0.1,
        TaskType.TRANSLATION:             0.1,
        TaskType.CREATIVE_WRITING:        0.4,
        TaskType.CODE_GENERATION:         0.6,
        TaskType.CODE_EDITING:            0.6,
        TaskType.CODE_DEBUGGING:          0.6,
        TaskType.MATHEMATICAL_REASONING:  0.8,
        TaskType.TECHNICAL_REASONING:     0.8,
        TaskType.REPOSITORY_SEARCH:       0.5,
        TaskType.LONG_CONTEXT_SYNTHESIS:  0.7,
        TaskType.PLANNING:                0.7,
        TaskType.TOOL_CALL_INTERPRETATION: 0.3,
        TaskType.STRUCTURED_EXTRACTION:   0.3,
        TaskType.UNKNOWN:                 0.3,
    }


# Lazily populated on first use so module-level import of taxonomy is not
# required at import time (prevents circular-import issues).
TASK_TIER_SIGNAL: "dict[TaskType, float]" = {}  # type: ignore[assignment]


def _get_task_tier_signal() -> "dict[TaskType, float]":
    global TASK_TIER_SIGNAL
    if not TASK_TIER_SIGNAL:
        TASK_TIER_SIGNAL = _build_task_tier_signal()
    return TASK_TIER_SIGNAL


# ── regex patterns ───────────────────────────────────────────────────────────

_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```|`[^`]+`")
# Raw code patterns: Python/JS/TS keywords and typical constructs that appear
# without markdown fences (e.g. pasted snippets, inline imports).
_RE_RAW_CODE = re.compile(
    r"\b(def |class |import |from \w+ import|async def |await |lambda |"
    r"const |let |var |function |return |yield |raise |except |try:|finally:|"
    r"for .+? in |while |elif |#.*\n|//|/\*|\*\/|printf|std::|->|:=)\b",
    re.I,
)
_RE_REASONING = re.compile(
    r"\b(reason|analyze|explain why|derive|infer|deduce|prove|hypothesis|"
    r"step.by.step|think through|chain of thought|because|therefore|thus|hence)\b",
    re.I,
)
_RE_SIMPLE = re.compile(
    # Exact factual lookups and greetings
    r"\b(what is|what are|who is|who was|when did|when was|where is|where was|"
    r"define |how many|how much|yes or no|true or false|spell out|"
    r"what does .{0,20} stand for|what does .{0,20} mean|"
    r"hi |hello|hey |good morning|good afternoon|how are you|nice to meet)\b",
    re.I,
)
_RE_MULTI_STEP = re.compile(
    r"\b(first|second|third|then|next|finally|step \d|1\.|2\.|3\.|\band\b.*\band\b)\b",
    re.I,
)
_RE_TECHNICAL = re.compile(
    r"\b(algorithm|complexity|architecture|refactor|optimize|latency|throughput|"
    r"concurrency|async|distributed|microservice|vector|embedding|neural|gradient|"
    r"tensor|tokenize|parse|compile|runtime|daemon|mutex|semaphore|"
    r"binary search|linked list|hash table|heap|trie|graph|b-tree|lsm.tree|"
    r"read amplification|write amplification|compaction|induction|"
    r"time complexity|space complexity|big.?o)\b",
    re.I,
)
_RE_CREATIVE = re.compile(
    r"\b(write a story|poem|creative|imagine|fiction|narrative|character|plot|"
    r"metaphor|analogy|brainstorm|generate ideas|fun|playful)\b",
    re.I,
)
_RE_QUESTION = re.compile(r"\?", re.I)
_RE_CONSTRAINT = re.compile(
    r"\b(must|should|only|never|always|do not|don\'t|avoid|require|"
    r"ensure|guarantee|exactly|no more than|at least|limit)\b",
    re.I,
)
_RE_IMPERATIVE = re.compile(
    r"^(write|create|build|implement|generate|fix|debug|refactor|"
    r"explain|summarize|translate|list|find|show|give|make|convert)\b",
    re.I | re.M,
)
_RE_FORMAT = re.compile(
    r"\b(json|yaml|xml|csv|table|markdown|html|code|list|bullet|numbered|"
    r"schema|structured|format)\b",
    re.I,
)
_RE_DOMAIN = re.compile(
    r"\b(kubernetes|terraform|sql|postgres|mongodb|kafka|rabbitmq|graphql|"
    r"react|typescript|fastapi|django|pytorch|tensorflow|scikit|pandas|"
    r"numpy|spark|hadoop|hive|flink|airflow|dbt)\b",
    re.I,
)
_RE_REFERENCE = re.compile(r"(attached|see above|as mentioned|in the context|refer to|based on)", re.I)
_RE_NEGATION = re.compile(r"\b(not|never|no |nor |neither|without|except|unless|but not)\b", re.I)

_ENC: tiktoken.Encoding | None = None


def _get_enc() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _norm(count: int, scale: int = 1) -> float:
    """Normalize a raw count to [0, 1] using a soft cap."""
    return min(count / scale, 1.0)


def extract_features(request: dict[str, Any]) -> RequestFeatures:
    """Extract the full feature set from an OpenAI-compatible chat request dict."""
    messages: list[dict] = request.get("messages", [])
    tools: list = request.get("tools", []) or []

    user_texts: list[str] = []
    all_texts: list[str] = []
    system_present = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if isinstance(content, list):
            # multi-modal content blocks
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
        all_texts.append(content)
        if role == "system":
            system_present = True
        elif role == "user":
            user_texts.append(content)

    full_text = " ".join(all_texts)
    last_user = user_texts[-1] if user_texts else ""

    enc = _get_enc()
    total_tokens = len(enc.encode(full_text))

    has_images = any(
        isinstance(block, dict) and block.get("type") == "image_url"
        for msg in messages
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
    )

    code_blocks = len(_RE_CODE_BLOCK.findall(full_text))

    f = RequestFeatures(
        total_tokens=total_tokens,
        user_message_count=sum(1 for m in messages if m.get("role") == "user"),
        assistant_message_count=sum(1 for m in messages if m.get("role") == "assistant"),
        system_present=system_present,
        last_user_text=last_user,
        full_text=full_text,
        has_tools=bool(tools),
        tool_count=len(tools),
        has_images=has_images,
        # 14 dimensions
        reasoning_markers=_norm(len(_RE_REASONING.findall(last_user)), 5),
        code_presence=_norm(
            code_blocks
            + len(_RE_CODE_BLOCK.findall(last_user))
            + len(_RE_RAW_CODE.findall(full_text)),
            3,
        ),
        simple_indicators=_norm(len(_RE_SIMPLE.findall(last_user)), 3),
        multi_step_patterns=_norm(len(_RE_MULTI_STEP.findall(last_user)), 4),
        technical_terms=_norm(len(_RE_TECHNICAL.findall(full_text)), 6),
        token_count_signal=min(total_tokens / 8000.0, 1.0),
        creative_markers=_norm(len(_RE_CREATIVE.findall(full_text)), 3),
        question_complexity=_norm(len(_RE_QUESTION.findall(last_user)), 3),
        constraint_count=_norm(len(_RE_CONSTRAINT.findall(full_text)), 5),
        imperative_verbs=_norm(len(_RE_IMPERATIVE.findall(last_user)), 3),
        output_format=_norm(len(_RE_FORMAT.findall(full_text)), 4),
        domain_specificity=_norm(len(_RE_DOMAIN.findall(full_text)), 4),
        reference_complexity=_norm(len(_RE_REFERENCE.findall(full_text)), 3),
        negation_complexity=_norm(len(_RE_NEGATION.findall(full_text)), 5),
    )

    # heuristic language hint
    if re.search(r"\b(def |class |import |from \w+ import|->|:\s*\n)\b", full_text):
        f.language_hint = "python"
    elif re.search(r"\b(function |const |let |var |=>|async/await)\b", full_text):
        f.language_hint = "javascript"
    elif re.search(r"\b(fn |let mut|impl |pub struct|cargo)\b", full_text):
        f.language_hint = "rust"

    return f
