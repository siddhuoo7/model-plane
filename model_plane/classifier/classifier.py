"""Task classifier — maps a RequestFeatures vector to a TaskType + confidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from model_plane.classifier.features import RequestFeatures
from model_plane.classifier.taxonomy import ComplexityTier, TaskType


@dataclass
class ClassificationResult:
    task_type: TaskType
    complexity_tier: ComplexityTier
    confidence: float
    signals: dict[str, float]

    def is_code_task(self) -> bool:
        return self.task_type in (
            TaskType.CODE_GENERATION,
            TaskType.CODE_EDITING,
            TaskType.CODE_DEBUGGING,
            TaskType.REPOSITORY_SEARCH,
        )


# ── rule-based classifier ─────────────────────────────────────────────────────

_CODE_DEBUG = re.compile(
    r"\b(debug|traceback|error:|exception|fix the bug|why does this fail|"
    r"stack trace|assertion error|segfault|linker error|failing test|"
    r"why is this (wrong|broken|failing)|what('s| is) wrong with)\b",
    re.I,
)
_CODE_GEN = re.compile(
    # Covers: "write a Python class", "implement a BST", "create a function",
    # "write code to", "write a script", "generate code", "build a class/function"
    r"\b(write\s+(?:\w+\s+){0,4}(function|class|method|script|program|module|api|"
    r"endpoint|decorator|generator|iterator|parser|cli|server|client)|"
    r"implement\s+(?:a\s+|an\s+|the\s+)?\w+|"
    r"create\s+(?:\w+\s+){0,4}(function|class|method|script|program|module)|"
    r"build\s+(?:\w+\s+){0,4}(function|class|api|service|module)|"
    r"generate code|write\s+(?:\w+\s+){0,3}code|scaffold|boilerplate|write a script)\b",
    re.I,
)
_CODE_EDIT = re.compile(
    r"\b(refactor|rename|extract method|optimize this|clean up|"
    r"improve this code|rewrite this|add type hints|add docstring)\b",
    re.I,
)
_REPO_SEARCH = re.compile(
    r"\b(search the repo|find in codebase|where is|which file|"
    r"in the repository|grep for|locate the)\b",
    re.I,
)
_MATH = re.compile(
    r"\b(calculate|compute|solve|equation|integral|derivative|"
    r"probability|statistics|matrix|eigenvalue|differential|"
    r"prove (that|the)|mathematical induction|number theory|geometric series)\b",
    re.I,
)
_TECH_REASON = re.compile(
    r"(\btrade.offs?\b|\btrade off\b|\bcompare\b|\bevaluate\b|\bpros and cons\b|"
    r"\bchoose between\b|\bbest approach\b|\bwhen to use\b|\bvs\b|\bversus\b|"
    r"\bwhat.s the difference\b|how does .{0,30} work|"
    r"\bexplain (the )?(architecture|design|algorithm|protocol|mechanism|trade.offs?)\b)",
    re.I,
)
_EXTRACT = re.compile(
    r"\b(extract|parse|pull out|fill in|json (from|object|output)|"
    r"table from|key.value|return (as|a) json|return (as|a) (dict|object|schema)|"
    r"structured (output|format|data)|fields:)\b",
    re.I,
)
_SUMMARIZE = re.compile(
    r"\b(summarize|summarise|summary|tldr|tl;dr|brief overview|condense|"
    r"key (points|takeaways)|executive summary|in a few sentences)\b",
    re.I,
)
_TRANSLATE = re.compile(r"\b(translate (this |the |to )|in french|in spanish|in german|in japanese|into (french|spanish|german|japanese|chinese|arabic))\b", re.I)
_CREATIVE = re.compile(r"\b(write a poem|write a story|creative writing|fiction|imagine|narrative|haiku|limerick)\b", re.I)
_PLAN = re.compile(r"\b(create a plan|project plan|roadmap|milestones|sprint plan|action items)\b", re.I)
_TOOL = re.compile(r"\b(use tool|call function|run tool|tool output|tool result)\b", re.I)
_LONG_CTX = re.compile(
    r"\b(entire document|whole file|full context|long document|across (all |multiple |the )|"
    r"50[,\s]?000.word|throughout the|across sections)\b",
    re.I,
)


def classify(features: RequestFeatures) -> ClassificationResult:
    """Rule-based task classification with confidence scoring."""
    text = features.last_user_text + " " + features.full_text
    signals: dict[str, float] = {}

    # code signals
    signals["code_debug"] = 1.0 if _CODE_DEBUG.search(text) else 0.0
    signals["code_gen"] = 1.0 if _CODE_GEN.search(text) else 0.0
    signals["code_edit"] = 1.0 if _CODE_EDIT.search(text) else 0.0
    signals["repo_search"] = 1.0 if _REPO_SEARCH.search(text) else 0.0
    signals["code_presence"] = features.code_presence

    # task type signals
    signals["math"] = 1.0 if _MATH.search(text) else 0.0
    signals["tech_reason"] = 1.0 if _TECH_REASON.search(text) else 0.0
    signals["extract"] = 1.0 if _EXTRACT.search(text) else 0.0
    signals["summarize"] = 1.0 if _SUMMARIZE.search(text) else 0.0
    signals["translate"] = 1.0 if _TRANSLATE.search(text) else 0.0
    signals["creative"] = 1.0 if _CREATIVE.search(text) else 0.0
    signals["plan"] = 1.0 if _PLAN.search(text) else 0.0
    signals["tool"] = 1.0 if _TOOL.search(text) else features.has_tools * 0.5
    signals["long_ctx"] = 1.0 if _LONG_CTX.search(text) else 0.0
    signals["simple"] = features.simple_indicators

    # priority order: most specific first
    task, confidence = _pick_task(signals, features)
    tier = _derive_tier(task, features)

    return ClassificationResult(
        task_type=task,
        complexity_tier=tier,
        confidence=confidence,
        signals=signals,
    )


def _pick_task(signals: dict[str, float], f: RequestFeatures) -> tuple[TaskType, float]:
    if signals["code_debug"]:
        return TaskType.CODE_DEBUGGING, 0.85
    if signals["repo_search"]:
        return TaskType.REPOSITORY_SEARCH, 0.82
    if signals["code_gen"]:
        return TaskType.CODE_GENERATION, 0.80
    if signals["code_edit"] and f.code_presence > 0.1:
        return TaskType.CODE_EDITING, 0.78
    if signals["math"]:
        return TaskType.MATHEMATICAL_REASONING, 0.80
    if signals["tech_reason"]:
        return TaskType.TECHNICAL_REASONING, 0.75
    if signals["long_ctx"] or f.token_count_signal > 0.75:
        return TaskType.LONG_CONTEXT_SYNTHESIS, 0.72
    if signals["tool"]:
        return TaskType.TOOL_CALL_INTERPRETATION, 0.75
    if signals["extract"]:
        return TaskType.STRUCTURED_EXTRACTION, 0.78
    if signals["summarize"]:
        return TaskType.SUMMARIZATION, 0.80
    if signals["translate"]:
        return TaskType.TRANSLATION, 0.85
    if signals["creative"]:
        return TaskType.CREATIVE_WRITING, 0.80
    if signals["plan"]:
        return TaskType.PLANNING, 0.75
    if signals["simple"] > 0.3:
        return TaskType.SIMPLE_QA, 0.70
    return TaskType.UNKNOWN, 0.40


def _derive_tier(task: TaskType, f: RequestFeatures) -> ComplexityTier:
    from model_plane.classifier.taxonomy import TASK_DEFAULT_TIER

    base = TASK_DEFAULT_TIER.get(task, ComplexityTier.MEDIUM)

    # escalate to reasoning if heavy reasoning markers
    if f.reasoning_markers > 0.5 and base != ComplexityTier.REASONING:
        if base == ComplexityTier.COMPLEX:
            base = ComplexityTier.REASONING

    # downgrade to simple only when ALL complexity signals are absent
    if f.simple_indicators > 0.6 and f.token_count_signal < 0.1:
        if base in (ComplexityTier.MEDIUM, ComplexityTier.COMPLEX):
            if f.technical_terms < 0.1 and f.reasoning_markers < 0.1 and f.multi_step_patterns < 0.1:
                base = ComplexityTier.SIMPLE

    # escalate for long context
    if f.token_count_signal > 0.6 and base == ComplexityTier.MEDIUM:
        base = ComplexityTier.COMPLEX

    return base
