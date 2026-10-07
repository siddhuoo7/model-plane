"""Thread-safe in-memory ring buffer for routing decisions.

Sub-Task 3.1 — Admin API Foundation.

Stores the last N ``RoutingRecord`` dicts in a fixed-size circular deque.
N is configurable via ``settings.admin_buffer_size`` (default 10 000).

Records are plain dicts so they can be directly JSON-serialised by the
``GET /admin/api/requests`` endpoint without any ORM or dataclass overhead.

Persistence: each record is appended to ``request_log`` in SQLite so the
buffer survives process restarts. On first access the buffer is warm-loaded
from SQLite up to ``maxlen`` rows.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any

from model_plane.config import settings
from model_plane.logging_setup import get_logger

log = get_logger(__name__)


class RequestRingBuffer:
    """Fixed-capacity thread-safe ring buffer of routing record dicts."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Warm the buffer from the SQLite request_log on startup."""
        try:
            from model_plane.db import load_request_log
            rows = load_request_log(self._maxlen)
            # load_request_log returns newest-first; we want oldest-first in the deque
            for row in reversed(rows):
                try:
                    self._buf.append(json.loads(row["data"]))
                except Exception:
                    pass
            if rows:
                log.info("request_buffer_reloaded", count=len(rows))
        except Exception as exc:
            log.warning("request_buffer_load_failed", error=str(exc))

    def push(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(record)

        # Write-through to SQLite (outside the in-memory lock)
        try:
            from model_plane.db import append_request_log, prune_request_log
            ts = record.get("ts") or time.time()
            append_request_log(ts=float(ts), data=json.dumps(record))
            # Prune to keep the log bounded (run occasionally — 0.5% of calls)
            import random
            if random.random() < 0.005:
                prune_request_log(self._maxlen)
        except Exception as exc:
            log.warning("request_buffer_persist_failed", error=str(exc))

    def snapshot(
        self,
        limit: int = 100,
        provider: str | None = None,
        tier: str | None = None,
        task_type: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to *limit* most-recent records, optionally filtered."""
        with self._lock:
            records = list(self._buf)

        # Apply filters (newest-first)
        filtered = []
        for rec in reversed(records):
            if provider and rec.get("provider") != provider:
                continue
            if tier and rec.get("tier") != tier:
                continue
            if task_type and rec.get("task_type") != task_type:
                continue
            if tenant_id and rec.get("tenant_id") != tenant_id:
                continue
            filtered.append(rec)
            if len(filtered) >= limit:
                break
        return filtered

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


# ── module-level singleton ────────────────────────────────────────────────────

_buffer: RequestRingBuffer | None = None
_buffer_lock = threading.Lock()


def get_request_buffer() -> RequestRingBuffer:
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                _buffer = RequestRingBuffer(maxlen=settings.admin_buffer_size)
                log.debug("request_buffer_initialised", maxlen=settings.admin_buffer_size)
    return _buffer
