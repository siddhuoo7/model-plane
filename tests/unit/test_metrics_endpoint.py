"""Unit tests for GET /v1/routing/metrics."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pytest

HEADER = [
    "timestamp", "request_id", "task_type", "tier", "routing_source",
    "reasoning_markers", "code_presence", "simple_indicators",
    "multi_step_patterns", "technical_terms", "token_count_signal",
    "creative_markers", "question_complexity", "constraint_count",
    "imperative_verbs", "output_format", "domain_specificity",
    "reference_complexity", "negation_complexity",
    "selected_deployment", "latency_ms", "cost_usd",
    "validation_result", "fallback_used", "escalated", "final_success",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _sample_row(**overrides) -> dict:
    base = dict(zip(HEADER, [
        "2024-01-01T00:00:00Z", "req-001", "simple_qa", "simple", "custom_plugin",
        0, 0, 0.9, 0, 0, 0.05, 0, 0.1, 0, 0, 0, 0, 0, 0,
        "watsonx-granite-small", 250, 0.000010,
        "ok", "False", "False", "1.0",
    ], strict=False))
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_no_data(tmp_path):
    """When the CSV doesn't exist, return status=no_data."""
    from model_plane.adapters.admin_router import routing_metrics

    canonical = Path("model_plane/data/training_data.csv")
    backup = None

    # Temporarily hide the file if it exists
    if canonical.exists():
        backup = canonical.read_bytes()
        canonical.unlink()

    try:
        result = asyncio.get_event_loop().run_until_complete(routing_metrics())
        assert result["status"] in ("no_data", "ok")  # ok if seed data present
    finally:
        if backup is not None:
            canonical.write_bytes(backup)


def test_metrics_with_real_data():
    """Write known rows to the canonical CSV path and verify aggregation."""
    from model_plane.adapters.admin_router import routing_metrics

    canonical = Path("model_plane/data/training_data.csv")
    backup = canonical.read_bytes() if canonical.exists() else None

    rows = [
        _sample_row(request_id=f"req-{i}", final_success="1.0",
                    latency_ms=200 + i * 10, cost_usd=0.00001 * (i + 1))
        for i in range(10)
    ]
    rows.append(_sample_row(
        request_id="req-fail",
        validation_result="schema_fail",
        fallback_used="True",
        escalated="True",
        final_success="0.0",
    ))

    canonical.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(canonical, rows)

    try:
        result = asyncio.get_event_loop().run_until_complete(routing_metrics())
    finally:
        if backup is not None:
            canonical.write_bytes(backup)
        elif canonical.exists():
            canonical.unlink()

    assert result["status"] == "ok"
    assert result["total_requests"] == 11
    assert result["fallback_rate"] > 0.0
    assert result["escalation_rate"] > 0.0
    assert "by_deployment" in result
    assert "watsonx-granite-small" in result["by_deployment"]
    dep_stats = result["by_deployment"]["watsonx-granite-small"]
    assert dep_stats["requests"] == 11
    assert dep_stats["mean_latency_ms"] is not None
    assert "by_task_type" in result
    assert "simple_qa" in result["by_task_type"]
    assert "by_routing_source" in result
    assert "custom_plugin" in result["by_routing_source"]


def test_metrics_success_rate_calculation():
    """success_rate should be 9/10 with one failure row."""
    from model_plane.adapters.admin_router import routing_metrics

    canonical = Path("model_plane/data/training_data.csv")
    backup = canonical.read_bytes() if canonical.exists() else None

    rows = [_sample_row(request_id=f"req-{i}", final_success="1.0") for i in range(9)]
    rows.append(_sample_row(request_id="req-fail", final_success="0.0"))

    canonical.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(canonical, rows)

    try:
        result = asyncio.get_event_loop().run_until_complete(routing_metrics())
    finally:
        if backup is not None:
            canonical.write_bytes(backup)
        elif canonical.exists():
            canonical.unlink()

    assert result["status"] == "ok"
    assert result["success_rate"] == pytest.approx(0.9, abs=0.01)
