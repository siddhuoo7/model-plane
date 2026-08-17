"""Prometheus metrics for the model control plane."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_plane.config import settings

if TYPE_CHECKING:
    from model_plane.routing.context import RoutingContext

try:
    from prometheus_client import Counter, Gauge, Histogram

    REQUESTS_TOTAL = Counter(
        "model_plane_requests_total",
        "Total requests processed",
        ["deployment", "task_type", "tier", "routing_source", "success"],
    )
    REQUEST_LATENCY = Histogram(
        "model_plane_request_latency_ms",
        "Request latency in milliseconds",
        ["deployment", "task_type"],
        buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000],
    )
    INPUT_TOKENS = Counter(
        "model_plane_input_tokens_total",
        "Total input tokens",
        ["deployment"],
    )
    OUTPUT_TOKENS = Counter(
        "model_plane_output_tokens_total",
        "Total output tokens",
        ["deployment"],
    )
    COST_USD = Counter(
        "model_plane_cost_usd_total",
        "Total estimated cost in USD",
        ["deployment"],
    )
    COMPRESSION_SAVINGS = Counter(
        "model_plane_compression_tokens_saved_total",
        "Total tokens saved by compression",
        ["profile"],
    )
    FALLBACK_TOTAL = Counter(
        "model_plane_fallback_total",
        "Total fallback routing decisions",
        ["reason"],
    )
    ACTIVE_SESSIONS = Gauge(
        "model_plane_active_sessions",
        "Approximate number of active sessions",
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def record_request(ctx: "RoutingContext", success: bool) -> None:
    if not settings.metrics_enabled or not _PROMETHEUS_AVAILABLE:
        return

    dep_name = ctx.selected_deployment.name if ctx.selected_deployment else "unknown"
    task = ctx.classification.task_type.value if ctx.classification else "unknown"
    tier = ctx.scorer_result.tier.value if ctx.scorer_result else "unknown"

    REQUESTS_TOTAL.labels(
        deployment=dep_name,
        task_type=task,
        tier=tier,
        routing_source=ctx.routing_source,
        success=str(success).lower(),
    ).inc()

    if ctx.latency_ms:
        REQUEST_LATENCY.labels(deployment=dep_name, task_type=task).observe(ctx.latency_ms)

    if ctx.input_tokens:
        INPUT_TOKENS.labels(deployment=dep_name).inc(ctx.input_tokens)

    if ctx.output_tokens:
        OUTPUT_TOKENS.labels(deployment=dep_name).inc(ctx.output_tokens)

    if ctx.cost_usd:
        COST_USD.labels(deployment=dep_name).inc(ctx.cost_usd)

    if ctx.tokens_before_compression > ctx.tokens_after_compression:
        COMPRESSION_SAVINGS.labels(profile=ctx.compression_profile).inc(
            ctx.tokens_before_compression - ctx.tokens_after_compression
        )

    if ctx.fallback_used:
        FALLBACK_TOTAL.labels(reason="routing_fallback").inc()
