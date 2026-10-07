"""Rolling cost accumulator for 24-hour and 30-day windows.

Sub-Task 3.1 — Admin API Foundation.

Keyed by ``(provider, tier, task_type, tenant_id)``; stores per-bucket USD cost
along with request count and token counts so the admin UI can compute averages.

Windows are approximated by bucketing into hourly slots:
  - 24-hour window: last 24 hourly buckets
  - 30-day window:  last 720 hourly buckets
This avoids per-request timestamp scan while keeping O(1) push cost.

Persistence: each bucket is written through to ``cost_buckets`` in SQLite so
the data survives process restarts.  On first access the accumulator reloads
all rows within the 30-day retention window from SQLite.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from model_plane.logging_setup import get_logger

log = get_logger(__name__)

_HOUR_SECONDS = 3600
_24H_BUCKETS = 24
_30D_BUCKETS = 720  # 30 * 24


@dataclass
class CostBucket:
    """Aggregated cost for one hour-slot."""
    hour_ts: int        # unix timestamp truncated to hour
    cost_usd: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CostSeries:
    """Per-key rolling series capped at 30-day depth."""
    buckets: deque[CostBucket] = field(default_factory=lambda: deque(maxlen=_30D_BUCKETS))

    def push(self, cost_usd: float, input_tokens: int, output_tokens: int) -> CostBucket:
        """Accumulate into the current hour bucket and return it (for persistence)."""
        hour_ts = int(time.time()) // _HOUR_SECONDS * _HOUR_SECONDS
        if self.buckets and self.buckets[-1].hour_ts == hour_ts:
            b = self.buckets[-1]
        else:
            b = CostBucket(hour_ts=hour_ts)
            self.buckets.append(b)
        b.cost_usd += cost_usd
        b.requests += 1
        b.input_tokens += input_tokens
        b.output_tokens += output_tokens
        return b

    def window_sum(self, hours: int) -> dict[str, Any]:
        cutoff = int(time.time()) - hours * _HOUR_SECONDS
        total_cost = 0.0
        total_requests = 0
        total_input = 0
        total_output = 0
        for b in self.buckets:
            if b.hour_ts >= cutoff:
                total_cost += b.cost_usd
                total_requests += b.requests
                total_input += b.input_tokens
                total_output += b.output_tokens
        return {
            "cost_usd": round(total_cost, 6),
            "requests": total_requests,
            "input_tokens": total_input,
            "output_tokens": total_output,
        }


class CostAccumulator:
    """Thread-safe multi-dimensional rolling cost store with SQLite write-through."""

    def __init__(self) -> None:
        self._data: dict[tuple, CostSeries] = defaultdict(CostSeries)
        self._lock = threading.Lock()
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Reload persisted buckets from SQLite (last 30 days)."""
        try:
            from model_plane.db import load_cost_buckets
            cutoff = int(time.time()) // _HOUR_SECONDS * _HOUR_SECONDS - _30D_BUCKETS * _HOUR_SECONDS
            rows = load_cost_buckets(cutoff)
            for row in rows:
                key = (row["provider"], row["tier"], row["task_type"], row["tenant_id"])
                series = self._data[key]
                # Insert in sorted order (SQLite may return them in any order)
                b = CostBucket(
                    hour_ts=row["hour_ts"],
                    cost_usd=row["cost_usd"],
                    requests=row["requests"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                )
                series.buckets.append(b)
            if rows:
                # Re-sort each series by hour_ts after bulk load
                for series in self._data.values():
                    sorted_buckets = sorted(series.buckets, key=lambda b: b.hour_ts)
                    series.buckets = deque(sorted_buckets, maxlen=_30D_BUCKETS)
                log.info("cost_accumulator_reloaded", rows=len(rows))
        except Exception as exc:
            log.warning("cost_accumulator_load_failed", error=str(exc))

    def record(
        self,
        *,
        provider: str,
        tier: str,
        task_type: str,
        tenant_id: str,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        key = (provider, tier, task_type, tenant_id)
        with self._lock:
            bucket = self._data[key].push(cost_usd, input_tokens, output_tokens)

        # Write-through to SQLite (outside the in-memory lock to avoid contention)
        try:
            from model_plane.db import upsert_cost_bucket, prune_cost_buckets
            upsert_cost_bucket(
                provider=provider, tier=tier, task_type=task_type, tenant_id=tenant_id,
                hour_ts=bucket.hour_ts,
                cost_usd=cost_usd, requests=1,
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
            # Prune rows older than 30 days (run occasionally — 1% of calls)
            import random
            if random.random() < 0.01:
                cutoff = int(time.time()) // _HOUR_SECONDS * _HOUR_SECONDS - _30D_BUCKETS * _HOUR_SECONDS
                prune_cost_buckets(cutoff)
        except Exception as exc:
            log.warning("cost_accumulator_persist_failed", error=str(exc))

    def summary(self, hours: int = 24) -> list[dict[str, Any]]:
        """Return cost breakdown for the requested window, one entry per key."""
        result = []
        with self._lock:
            for (provider, tier, task_type, tenant_id), series in self._data.items():
                window = series.window_sum(hours)
                if window["requests"] == 0:
                    continue
                result.append({
                    "provider": provider,
                    "tier": tier,
                    "task_type": task_type,
                    "tenant_id": tenant_id,
                    **window,
                })
        return sorted(result, key=lambda x: -x["cost_usd"])

    def total_cost(self, hours: int = 24) -> float:
        return sum(r["cost_usd"] for r in self.summary(hours))


# ── module-level singleton ────────────────────────────────────────────────────

_accumulator: CostAccumulator | None = None
_acc_lock = threading.Lock()


def get_cost_accumulator() -> CostAccumulator:
    global _accumulator
    if _accumulator is None:
        with _acc_lock:
            if _accumulator is None:
                _accumulator = CostAccumulator()
                log.debug("cost_accumulator_initialised")
    return _accumulator
