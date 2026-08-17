"""Unit tests for shadow traffic (A/B fire-and-forget).

get_catalog is imported locally inside maybe_fire_shadow, so we swap the
module-level singleton in model_plane.registry.catalog directly.
"""

from __future__ import annotations

from unittest.mock import patch

import model_plane.registry.catalog as cat_mod
from model_plane.adapters.executor import maybe_fire_shadow
from model_plane.registry.catalog import DeploymentConfig, ModelCatalog
from model_plane.routing.context import RoutingContext


def _make_ctx(dep_name: str = "wx-primary") -> RoutingContext:
    ctx = RoutingContext(request_id="test-shadow", raw_request={})
    dep = DeploymentConfig(
        name=dep_name, litellm_model="watsonx/x", provider="watsonx", tier="medium"
    )
    ctx.selected_deployment = dep
    return ctx


def _make_catalog(*names: str) -> ModelCatalog:
    catalog = ModelCatalog()
    for n in names:
        dep = DeploymentConfig(name=n, litellm_model="x", provider="watsonx", tier="medium")
        catalog.deployments[n] = dep
        catalog.tier_map.setdefault("medium", []).append(n)
    return catalog


# ─────────────────────────────────────────────────────────────────────────────

def test_shadow_not_fired_when_fraction_zero(monkeypatch):
    monkeypatch.setattr("model_plane.adapters.executor.settings.shadow_traffic_fraction", 0.0)
    ctx = _make_ctx()

    with patch("model_plane.adapters.executor.asyncio.ensure_future") as mock_future:
        maybe_fire_shadow(ctx)
        mock_future.assert_not_called()


def test_shadow_not_fired_when_no_other_deployments(monkeypatch):
    monkeypatch.setattr("model_plane.adapters.executor.settings.shadow_traffic_fraction", 1.0)

    catalog = _make_catalog("wx-primary")   # only the primary — nothing else to shadow

    ctx = _make_ctx("wx-primary")

    original = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.adapters.executor.asyncio.ensure_future") as mock_future, \
             patch("model_plane.adapters.executor.random.random", return_value=0.0):
            maybe_fire_shadow(ctx)
            mock_future.assert_not_called()
    finally:
        cat_mod._catalog = original


def test_shadow_fired_when_fraction_one(monkeypatch):
    """With fraction=1.0 and multiple healthy deployments, a shadow call must fire."""
    monkeypatch.setattr("model_plane.adapters.executor.settings.shadow_traffic_fraction", 1.0)

    catalog = _make_catalog("wx-primary", "wx-shadow")
    ctx = _make_ctx("wx-primary")

    original = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.adapters.executor.asyncio.ensure_future") as mock_future, \
             patch("model_plane.adapters.executor.random.random", return_value=0.0):
            maybe_fire_shadow(ctx)
            mock_future.assert_called_once()
    finally:
        cat_mod._catalog = original


def test_shadow_skipped_by_random(monkeypatch):
    """When random.random() > fraction the shadow must NOT fire."""
    monkeypatch.setattr("model_plane.adapters.executor.settings.shadow_traffic_fraction", 0.05)

    catalog = _make_catalog("a", "b")
    ctx = _make_ctx("a")

    original = cat_mod._catalog
    cat_mod._catalog = catalog
    try:
        with patch("model_plane.adapters.executor.asyncio.ensure_future") as mock_future, \
             patch("model_plane.adapters.executor.random.random", return_value=0.99):
            maybe_fire_shadow(ctx)
            mock_future.assert_not_called()
    finally:
        cat_mod._catalog = original
