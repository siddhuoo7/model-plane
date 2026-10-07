"""Shared pytest fixtures for unit tests.

Fixtures defined here are available to every test module under tests/unit/
without needing an explicit import.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mock_creds_for_catalog(monkeypatch):
    """Make 'openai' and 'local' appear credentialed in catalog filtering.

    The real test environment has no API keys, so catalog.all_healthy() and
    catalog.by_tier() would return empty lists, causing every routing pipeline
    test to fail with "No deployments available".

    Patching the name in catalog.py's namespace (not in provider_creds) is
    required because catalog.py does a module-level import that binds the name
    locally at import time.
    """
    monkeypatch.setattr(
        "model_plane.registry.catalog.provider_has_creds",
        lambda p: p in ("openai", "local"),
    )


@pytest.fixture(autouse=True)
def _mock_db_persistence(monkeypatch):
    """Prevent observability classes from loading or writing real SQLite data.

    Tests that directly instantiate RequestRingBuffer or CostAccumulator must
    not pick up data from a local config/model_plane.db, and must not write
    to it either.  Patch all four DB helpers to no-ops.
    """
    monkeypatch.setattr(
        "model_plane.observability.request_buffer.RequestRingBuffer._load_from_db",
        lambda self: None,
    )
    monkeypatch.setattr(
        "model_plane.observability.cost_accumulator.CostAccumulator._load_from_db",
        lambda self: None,
    )
    # Silence write-through calls so tests run without a live DB
    monkeypatch.setattr(
        "model_plane.observability.request_buffer.RequestRingBuffer.push",
        _push_no_persist,
    )
    monkeypatch.setattr(
        "model_plane.observability.cost_accumulator.CostAccumulator.record",
        _record_no_persist,
    )
    # Mock load_routing_override for classifier namespace so test suites default to regex without DB contamination
    from model_plane.db import load_routing_override as _orig_load_override
    def _mock_load_routing_override(namespace, owner_id):
        if namespace == "classifier":
            return {}
        return _orig_load_override(namespace, owner_id)
    monkeypatch.setattr(
        "model_plane.db.load_routing_override",
        _mock_load_routing_override,
    )


def _push_no_persist(self, record):
    """push() without SQLite write-through, for unit tests."""
    import threading
    with self._lock:
        self._buf.append(record)


def _record_no_persist(self, *, provider, tier, task_type, tenant_id,
                        cost_usd, input_tokens, output_tokens):
    """record() without SQLite write-through, for unit tests."""
    key = (provider, tier, task_type, tenant_id)
    with self._lock:
        self._data[key].push(cost_usd, input_tokens, output_tokens)
