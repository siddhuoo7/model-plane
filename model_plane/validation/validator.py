"""Response validation — structured output, tool-call, and quality checks.

Phase 7: drives validation-triggered escalation / fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from model_plane.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    result_code: str  # ok | schema_fail | tool_call_fail | empty | quality_low | parse_error
    details: str = ""
    should_escalate: bool = False
    should_retry: bool = False


def validate_response(
    response_content: str | None,
    *,
    expect_json: bool = False,
    json_schema: dict | None = None,
    expect_tool_call: bool = False,
    tool_schemas: list[dict] | None = None,
    min_length: int = 1,
) -> ValidationResult:
    """Validate a model response and produce a structured ValidationResult."""

    if not response_content or len(response_content.strip()) < min_length:
        return ValidationResult(
            passed=False,
            result_code="empty",
            details="Response was empty or too short",
            should_retry=True,
        )

    # ── JSON schema validation ────────────────────────────────────────────────
    if expect_json:
        parsed = _extract_json(response_content)
        if parsed is None:
            return ValidationResult(
                passed=False,
                result_code="parse_error",
                details="Could not extract valid JSON from response",
                should_escalate=True,
            )
        if json_schema:
            schema_err = _validate_against_schema(parsed, json_schema)
            if schema_err:
                return ValidationResult(
                    passed=False,
                    result_code="schema_fail",
                    details=schema_err,
                    should_escalate=True,
                )

    # ── Tool-call validation ──────────────────────────────────────────────────
    if expect_tool_call:
        found = _has_tool_call_markers(response_content)
        if not found:
            return ValidationResult(
                passed=False,
                result_code="tool_call_fail",
                details="Expected tool call not found in response",
                should_escalate=True,
            )

    # ── Quality floor ─────────────────────────────────────────────────────────
    if len(response_content.strip()) < 10:
        return ValidationResult(
            passed=False,
            result_code="quality_low",
            details="Response too short to be useful",
            should_retry=True,
        )

    return ValidationResult(passed=True, result_code="ok")


def _extract_json(text: str) -> Any:
    """Try to extract a JSON object from a text response."""
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding a JSON block in markdown
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first `{...}` block
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _validate_against_schema(obj: Any, schema: dict) -> str | None:
    """Minimal required-keys schema check (no jsonschema dep)."""
    required = schema.get("required", [])
    if not isinstance(obj, dict):
        return "Expected JSON object (dict)"
    missing = [k for k in required if k not in obj]
    if missing:
        return f"Missing required fields: {missing}"
    return None


def _has_tool_call_markers(text: str) -> bool:
    return bool(re.search(r'"name"\s*:\s*"[^"]+"', text) or "tool_calls" in text.lower())


# ── Escalation policy ─────────────────────────────────────────────────────────

@dataclass
class EscalationPolicy:
    """Defines how to react to validation failures."""

    max_retries: int = 2
    escalate_on_schema_fail: bool = True
    escalate_on_tool_fail: bool = True
    fallback_deployments: list[str] = field(default_factory=list)  # in order

    def should_escalate(self, result: ValidationResult, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return True
        return result.should_escalate

    def next_fallback(self, attempt: int) -> str | None:
        if attempt < len(self.fallback_deployments):
            return self.fallback_deployments[attempt]
        return None
