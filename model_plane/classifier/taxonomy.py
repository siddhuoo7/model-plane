"""Task taxonomy definitions — single source of truth for all task types."""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    SIMPLE_QA = "simple_qa"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    STRUCTURED_EXTRACTION = "structured_extraction"
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    CODE_EDITING = "code_editing"
    CODE_DEBUGGING = "code_debugging"
    REPOSITORY_SEARCH = "repository_search"
    TOOL_CALL_INTERPRETATION = "tool_call_interpretation"
    TECHNICAL_REASONING = "technical_reasoning"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    LONG_CONTEXT_SYNTHESIS = "long_context_synthesis"
    UNKNOWN = "unknown"


# Canonical ordering index for ComplexityTier comparisons.
_TIER_ORDER: dict[str, int] = {
    "simple":    0,
    "medium":    1,
    "complex":   2,
    "reasoning": 3,
}


class ComplexityTier(str, Enum):
    """Coarse routing tier derived from scorer output.

    Supports ordering comparisons (``<``, ``<=``, ``>``, ``>=``) based on
    increasing complexity: SIMPLE < MEDIUM < COMPLEX < REASONING.
    """
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    REASONING = "reasoning"

    def _rank(self) -> int:
        return _TIER_ORDER[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ComplexityTier):
            return NotImplemented
        return self._rank() < other._rank()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, ComplexityTier):
            return NotImplemented
        return self._rank() <= other._rank()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, ComplexityTier):
            return NotImplemented
        return self._rank() > other._rank()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, ComplexityTier):
            return NotImplemented
        return self._rank() >= other._rank()


# Map task types to their natural complexity tier (can be overridden by scorer)
TASK_DEFAULT_TIER: dict[TaskType, ComplexityTier] = {
    TaskType.SIMPLE_QA: ComplexityTier.SIMPLE,
    TaskType.SUMMARIZATION: ComplexityTier.MEDIUM,
    TaskType.TRANSLATION: ComplexityTier.SIMPLE,
    TaskType.CREATIVE_WRITING: ComplexityTier.MEDIUM,
    TaskType.STRUCTURED_EXTRACTION: ComplexityTier.MEDIUM,
    TaskType.PLANNING: ComplexityTier.COMPLEX,
    TaskType.CODE_GENERATION: ComplexityTier.COMPLEX,
    TaskType.CODE_EDITING: ComplexityTier.MEDIUM,
    TaskType.CODE_DEBUGGING: ComplexityTier.COMPLEX,
    TaskType.REPOSITORY_SEARCH: ComplexityTier.MEDIUM,
    TaskType.TOOL_CALL_INTERPRETATION: ComplexityTier.MEDIUM,
    TaskType.TECHNICAL_REASONING: ComplexityTier.REASONING,
    TaskType.MATHEMATICAL_REASONING: ComplexityTier.REASONING,
    TaskType.LONG_CONTEXT_SYNTHESIS: ComplexityTier.COMPLEX,
    TaskType.UNKNOWN: ComplexityTier.MEDIUM,
}
