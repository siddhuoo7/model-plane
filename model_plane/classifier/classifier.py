"""Task classifier — maps a RequestFeatures vector to a TaskType + confidence.

Two modes, controlled by MODEL_PLANE_BERT_CLASSIFIER_ENABLED:

  False (default)  — pure regex priority chain, zero extra dependencies, ~0ms overhead.
  True             — regex runs first; if BERT is loaded and its confidence exceeds
                     BERT_CLASSIFIER_MIN_CONFIDENCE it overrides the task_type (but keeps
                     the regex-derived tier and signals).  Falls back to regex silently if
                     transformers is not installed or the model fails to load.
"""

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
    r"why is this (wrong|broken|failing)|what('s| is) wrong with|"
    # "Fix this Python error: NameError" — colon after error word, or "NameError"
    r"NameError|TypeError|ValueError|KeyError|AttributeError|IndexError|"
    r"ImportError|RuntimeError|SyntaxError|ZeroDivisionError|"
    r"fix this .{0,20} error|fix the .{0,20} error|fix .{0,20} bug|"
    r"why (is|does) (this|my|the) .{0,30} (fail|crash|error|not work))\b",
    re.I,
)
_CODE_GEN = re.compile(
    # Covers: "write a Python class", "implement a BST", "create a function",
    # "write code to", "write a script", "generate code", "build a class/function",
    # "write a Python async web scraper", "write a REST API"
    r"\b(write\s+(?:\w+\s+){0,4}(function|class|method|script|program|module|api|"
    r"endpoint|decorator|generator|iterator|parser|cli|server|client|scraper|crawler|"
    r"bot|hook|plugin|library|sdk|util|utility|tool|pipeline|agent|worker)|"
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
# Exact-word alternations — trailing \b is safe here.
_REPO_SEARCH = re.compile(
    r"\b(search the repo|find in codebase|which file|in the repository|grep for|locate the)\b",
    re.I,
)
# Stem-based alternations — NO trailing \b because the stem (e.g. "implement")
# appears mid-word ("implemented"). Match whole pattern at the start boundary only.
_REPO_SEARCH_STEM = re.compile(
    r"\bwhere is .{0,30}implement|\bwhere is .{0,30}defined|"
    r"\bfind where .{0,50}implement|\bfind where .{0,50}defined|\bfind where .{0,50}called|"
    r"\bwhere (is|are|was|were) .{0,40}implement",
    re.I,
)
_MATH = re.compile(
    r"\b(calculate|compute|solve|equation|integral|derivative|"
    r"probability|statistics|matrix|eigenvalue|differential|"
    # "prove that/the/by" — covers "prove by induction", "prove that X"
    r"prove (that|the|by)|mathematical induction|induction (proof|step)|"
    r"number theory|geometric series|divisib|"
    # arithmetic: "what is 12345 * 6789" — require an operator so "what is 2+2" matches
    # but "what is Python" doesn't. Operator must be adjacent to digits.
    r"\d+\s*[+\-\*×÷/%]\s*\d+|\d+\s*\^\s*\d+)\b",
    re.I,
)
_TECH_REASON = re.compile(
    r"(\btrade.offs?\b|\btrade off\b|\bcompare\b|\bevaluate\b|\bpros and cons\b|"
    r"\bchoose between\b|\bbest approach\b|\bwhen to use\b|\bvs\b|\bversus\b|"
    # "key differences", "the difference", "what's the difference"
    r"\bkey differences?\b|\bwhat.s the differences?\b|\bthe differences? between\b|"
    r"how does .{0,30} work|"
    r"\bexplain (the )?(architecture|design|algorithm|protocol|mechanism|trade.offs?)\b|"
    # "Design a microservices architecture", "design the system", "design an API for ..."
    r"\bdesign\s+(?:a\s+|an\s+|the\s+)?(?:\w+\s+){0,3}"
    r"(microservice|distributed|scalable|event.driven|cloud.native|serverless|modular|"
    r"system|api|database|data|pipeline|service|architecture|schema|workflow|platform|solution)\b)",
    re.I,
)
_EXTRACT = re.compile(
    r"\b(extract|parse|pull out|fill in|json (from|object|output)|"
    r"table from|key.value|return (as|a) json|return (as|a) (dict|object|schema)|"
    r"structured (output|format|data)|fields:)\b",
    re.I,
)
_SUMMARIZE = re.compile(
    # "summarize" alone — but NOT "summarize the key differences" which is tech_reason
    # Anchored to standalone summarisation verbs without "differences/compare/vs" context
    r"\b(summarize|summarise|summary|tldr|tl;dr|brief overview|condense|"
    r"key (points|takeaways)|executive summary|in a few sentences)\b",
    re.I,
)
_TRANSLATE = re.compile(r"\b(translate (this |the |to )|in french|in spanish|in german|in japanese|into (french|spanish|german|japanese|chinese|arabic))\b", re.I)
_CREATIVE = re.compile(r"\b(write a poem|write a story|creative writing|fiction|imagine|narrative|haiku|limerick)\b", re.I)
_PLAN = re.compile(
    r"\b(create a plan|project plan|roadmap|milestones|sprint plan|action items|"
    # "Create a migration plan", "Write a phased rollout plan", "Plan the steps to..."
    r"migration plan|rollout plan|deployment plan|release plan|incident plan|"
    r"plan the (steps|migration|rollout|deployment|release|phases)|"
    r"write a .{0,20} plan|create a .{0,20} plan|build a .{0,20} plan)\b",
    re.I,
)
_TOOL = re.compile(r"\b(use tool|call function|run tool|tool output|tool result)\b", re.I)
_LONG_CTX = re.compile(
    r"\b(entire document|whole file|full context|long document|across (all |multiple |the )|"
    r"50[,\s]?000.word|throughout the|across sections)\b",
    re.I,
)


def classify(features: RequestFeatures, owner_id: str = "") -> ClassificationResult:
    """Classify a request into a TaskType + ComplexityTier.

    Step 1: Always run the regex chain (fast, zero deps, always available).
    Step 2: If CLASSIFIER_MODEL != "regex", the selected ML adapter runs after the
            regex step. If the adapter returns a result with confidence >=
            CLASSIFIER_MIN_CONFIDENCE it overrides task_type (mBERT) or both
            task_type AND complexity_tier (Laya). Falls back to regex silently on
            any error.

    The recommended call site for async FastAPI routes is ``classify_async()``.
    This synchronous wrapper is safe to call from any sync context — it creates
    a fresh, private event loop that is closed before returning, so the caller's
    loop is never affected.
    """
    import asyncio

    result = _classify_sync(features, owner_id=owner_id)
    if result is not None:
        return result

    # Kick off the async adapter from a sync context (e.g. tests, pipeline).
    # Use a dedicated new loop to avoid interfering with any existing loop
    # created by pytest-asyncio or the application's ASGI server.
    try:
        loop = asyncio.get_running_loop()
        # We are already inside a running loop — delegate to a thread pool to
        # avoid deadlock (asyncio.run() cannot be used inside a running loop).
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _classify_async_impl(features, owner_id=owner_id))
            return future.result()
    except RuntimeError:
        # No running loop in this thread — create a private loop, run, then
        # close it.  We do NOT use asyncio.run() here because that also calls
        # loop.close() which corrupts the default loop in Python ≤3.9 test
        # environments where get_event_loop() returns the same object.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_classify_async_impl(features, owner_id=owner_id))
        finally:
            loop.close()


async def classify_async(features: RequestFeatures, owner_id: str = "") -> ClassificationResult:
    """Async variant — preferred entry point from FastAPI route handlers."""
    return await _classify_async_impl(features, owner_id=owner_id)


def _regex_classify(features: RequestFeatures) -> tuple[TaskType, ComplexityTier, float, dict]:
    """Run the regex baseline and return (task, tier, confidence, signals)."""
    text = features.last_user_text + " " + features.full_text
    signals: dict[str, float] = {}

    # code signals
    signals["code_debug"] = 1.0 if _CODE_DEBUG.search(text) else 0.0
    signals["code_gen"] = 1.0 if _CODE_GEN.search(text) else 0.0
    signals["code_edit"] = 1.0 if _CODE_EDIT.search(text) else 0.0
    signals["repo_search"] = 1.0 if (_REPO_SEARCH.search(text) or _REPO_SEARCH_STEM.search(text)) else 0.0
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

    task, confidence = _pick_task(signals, features)
    tier = _derive_tier(task, features)
    return task, tier, confidence, signals


def _classify_sync(features: RequestFeatures, owner_id: str = "") -> ClassificationResult | None:
    """Return a regex-only result immediately if no ML adapter is configured."""
    from model_plane.classifier.factory import get_classifier
    adapter = get_classifier(owner_id=owner_id)
    if adapter is None:
        task, tier, confidence, signals = _regex_classify(features)
        signals["_source"] = 0.0
        return ClassificationResult(
            task_type=task,
            complexity_tier=tier,
            confidence=confidence,
            signals=signals,
        )
    return None  # need async path


async def _classify_async_impl(features: RequestFeatures, owner_id: str = "") -> ClassificationResult:
    """Full classify pipeline including optional async ML adapter."""
    from model_plane.classifier.factory import get_classifier

    task, tier, confidence, signals = _regex_classify(features)

    # ── Step 2: ML adapter augmentation (opt-in) ─────────────────────────────
    adapter = get_classifier(owner_id=owner_id)
    if adapter is not None:
        try:
            adapter_result = await adapter.classify(features)
            if adapter_result is not None:
                # mBERT: overrides task_type only (tier already re-derived inside adapter)
                # Laya:  overrides BOTH task_type AND complexity_tier
                task = adapter_result.task_type
                confidence = adapter_result.confidence
                tier = adapter_result.complexity_tier
                # Merge adapter signals on top of regex signals
                signals.update(adapter_result.signals)
        except Exception:
            # Never let adapter errors break routing — regex result stands
            pass

    signals.setdefault("_source", 0.0)

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
    # Raw code present (pasted snippet) without explicit verb → code_generation
    if f.code_presence > 0.4:
        return TaskType.CODE_GENERATION, 0.72
    if signals["math"]:
        # Guard: trivially simple arithmetic ("What is 2+2?") — strong simple signal
        # AND very short query → classify as simple_qa, not mathematical_reasoning.
        if signals["simple"] > 0.3 and f.total_tokens <= 15 and f.code_presence < 0.1:
            return TaskType.SIMPLE_QA, 0.70
        return TaskType.MATHEMATICAL_REASONING, 0.80
    if signals["tech_reason"]:
        return TaskType.TECHNICAL_REASONING, 0.75
    if signals["long_ctx"] or f.token_count_signal > 0.75:
        return TaskType.LONG_CONTEXT_SYNTHESIS, 0.72
    if signals["tool"]:
        return TaskType.TOOL_CALL_INTERPRETATION, 0.75
    if signals["extract"]:
        return TaskType.STRUCTURED_EXTRACTION, 0.78
    # tech_reason beats summarize: "Summarize the key differences between X and Y"
    # fires both summarize and tech_reason — the intent is comparison, not summarisation.
    if signals["summarize"] and not signals["tech_reason"]:
        return TaskType.SUMMARIZATION, 0.80
    if signals["translate"]:
        return TaskType.TRANSLATION, 0.85
    if signals["creative"]:
        return TaskType.CREATIVE_WRITING, 0.80
    if signals["plan"]:
        return TaskType.PLANNING, 0.75
    # simple_qa: explicit factual lookup OR short greeting with no other signals
    if signals["simple"] > 0.3 and f.code_presence < 0.1:
        return TaskType.SIMPLE_QA, 0.70
    # Catch greetings / very short low-signal inputs ("Hi, how are you?")
    if f.simple_indicators > 0.1 and f.total_tokens < 20 and f.code_presence < 0.1:
        return TaskType.SIMPLE_QA, 0.65
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
