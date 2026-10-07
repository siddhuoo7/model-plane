"""Unit tests for the 15th feature dimension (task_type_tier_signal)."""

from __future__ import annotations

import pytest

from model_plane.classifier.features import RequestFeatures, _get_task_tier_signal
from model_plane.classifier.taxonomy import TaskType


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_features() -> RequestFeatures:
    """Return a minimal RequestFeatures object."""
    return RequestFeatures(
        total_tokens=10,
        last_user_text="hello",
        full_text="hello",
    )


# ── 14-dim default (flag off) ─────────────────────────────────────────────────

def test_as_vector_default_is_14_dim():
    """Without a task_type, as_vector() always returns 14 floats."""
    f = _make_features()
    assert len(f.as_vector()) == 14


def test_as_vector_with_task_type_but_flag_off(monkeypatch):
    """Passing task_type does NOT extend the vector when flag is False."""
    monkeypatch.setattr("model_plane.config.settings", type("S", (), {"features_15_dim_enabled": False})())
    f = _make_features()
    assert len(f.as_vector(task_type=TaskType.CODE_GENERATION)) == 14


# ── 15-dim (flag on) ──────────────────────────────────────────────────────────

def test_as_vector_15_dim_when_flag_enabled(monkeypatch):
    """When flag is True and task_type provided, as_vector() returns 15 floats."""
    monkeypatch.setattr("model_plane.config.settings", type("S", (), {"features_15_dim_enabled": True})())
    f = _make_features()
    vec = f.as_vector(task_type=TaskType.CODE_GENERATION)
    assert len(vec) == 15


def test_15th_dim_value_matches_signal_map(monkeypatch):
    """The 15th element equals the value in TASK_TIER_SIGNAL for that TaskType."""
    monkeypatch.setattr("model_plane.config.settings", type("S", (), {"features_15_dim_enabled": True})())
    signal_map = _get_task_tier_signal()
    f = _make_features()
    for task_type, expected in signal_map.items():
        vec = f.as_vector(task_type=task_type)
        assert len(vec) == 15, f"Expected 15 dims for {task_type}"
        assert vec[14] == pytest.approx(expected), f"Wrong signal for {task_type}"


def test_all_task_types_in_signal_map():
    """Every TaskType has an entry in TASK_TIER_SIGNAL."""
    signal_map = _get_task_tier_signal()
    for task_type in TaskType:
        assert task_type in signal_map, f"{task_type} missing from TASK_TIER_SIGNAL"


def test_signal_values_in_range():
    """All signal values are in [0, 1]."""
    for task_type, value in _get_task_tier_signal().items():
        assert 0.0 <= value <= 1.0, f"{task_type} signal {value} out of range"


def test_no_task_type_no_extension_even_when_flag_on(monkeypatch):
    """If task_type is None, the vector stays 14-dim regardless of flag."""
    monkeypatch.setattr("model_plane.config.settings", type("S", (), {"features_15_dim_enabled": True})())
    f = _make_features()
    assert len(f.as_vector()) == 14
