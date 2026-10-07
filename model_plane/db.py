"""SQLite persistence layer for Model Plane UI configuration.

Stores provider credentials, API keys, and user accounts.
This is the primary persistent store; .env / os.environ is a read-only fallback
loaded at start-up when no SQLite row exists for a given key.

Isolation model
---------------
Provider credentials and API keys are **scoped per user** (owner_id = user.id).
A user can only read, write, or delete their own rows.  .env values are treated
as a global fallback visible to all users but not owned by any.

Database location: config/model_plane.db  (configurable via MODEL_PLANE_DB_PATH)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from model_plane.logging_setup import get_logger

log = get_logger(__name__)

_DB_PATH = Path(os.environ.get("MODEL_PLANE_DB_PATH", "config/model_plane.db"))
_lock = threading.Lock()


# ── Connection helper ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent schema creation — safe to call on every start-up."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kv (
            namespace TEXT    NOT NULL,
            key       TEXT    NOT NULL,
            owner_id  TEXT    NOT NULL DEFAULT '',
            value     TEXT    NOT NULL,
            PRIMARY KEY (namespace, key, owner_id)
        );

        CREATE TABLE IF NOT EXISTS users (
            email           TEXT PRIMARY KEY,
            id              TEXT NOT NULL,
            name            TEXT NOT NULL DEFAULT '',
            hashed_password TEXT NOT NULL,
            salt            TEXT NOT NULL,
            created_at      REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            kid        TEXT PRIMARY KEY,
            label      TEXT NOT NULL DEFAULT '',
            prefix     TEXT NOT NULL,
            raw        TEXT NOT NULL,
            created_at REAL NOT NULL,
            created_by TEXT NOT NULL DEFAULT '',
            owner_id   TEXT NOT NULL DEFAULT ''
        );

        -- Cost accumulator persistence: one row per (provider, tier, task_type, tenant_id, hour_ts)
        CREATE TABLE IF NOT EXISTS cost_buckets (
            provider     TEXT    NOT NULL,
            tier         TEXT    NOT NULL,
            task_type    TEXT    NOT NULL,
            tenant_id    TEXT    NOT NULL,
            hour_ts      INTEGER NOT NULL,
            cost_usd     REAL    NOT NULL DEFAULT 0,
            requests     INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (provider, tier, task_type, tenant_id, hour_ts)
        );

        -- Request ring buffer persistence: last N routing decisions
        CREATE TABLE IF NOT EXISTS request_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        REAL    NOT NULL,
            data      TEXT    NOT NULL
        );

        -- Per-user catalog overrides: enable/disable/delete/add deployments without
        -- touching models.yaml.  models.yaml is the immutable base; this table is
        -- the user-scoped delta layer applied on top at read time.
        --
        -- patch JSON shape:
        --   { "disabled": true }           — hide from this user's routing view
        --   { "deleted": true }            — treat as deleted for this user
        --   { "fields": { ... } }          — field-level overrides (tier, litellm_model, …)
        --   { "added": { <full dep> } }    — deployment added via UI (not in YAML)
        CREATE TABLE IF NOT EXISTS catalog_overrides (
            name      TEXT    NOT NULL,
            owner_id  TEXT    NOT NULL,
            patch     TEXT    NOT NULL,
            updated_at REAL   NOT NULL DEFAULT 0,
            PRIMARY KEY (name, owner_id)
        );

        -- Per-user routing/feature-flag overrides: classifier, smart-routing toggles,
        -- compression config, routing rules — all stored here, routing.yaml is read-only base.
        --
        -- value JSON: arbitrary dict merged onto routing.yaml at read time.
        -- namespace examples:
        --   "smart_routing"  → { security_routing_enabled, context_reuse_enabled, … }
        --   "compression"    → { enabled, profile, token_threshold, … }
        --   "routing_rules"  → { scorer_weights, task_overrides, preferred_providers, … }
        --   "classifier"     → { active, mbert_enabled, laya_enabled }
        --   "provider_trust" → { provider_trust_overrides: { bedrock: "private_cloud", … } }
        CREATE TABLE IF NOT EXISTS routing_overrides (
            namespace  TEXT   NOT NULL,
            owner_id   TEXT   NOT NULL,
            value      TEXT   NOT NULL,
            updated_at REAL   NOT NULL DEFAULT 0,
            PRIMARY KEY (namespace, owner_id)
        );
    """)
    # Migration: add owner_id column to kv and api_keys if they were created
    # without it (upgrading from the previous schema).
    for stmt in (
        "ALTER TABLE kv ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE api_keys ADD COLUMN owner_id TEXT NOT NULL DEFAULT ''",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    # Rebuild the PK-equivalent unique index for kv after migration
    conn.executescript("""
        CREATE UNIQUE INDEX IF NOT EXISTS kv_pk
            ON kv (namespace, key, owner_id);
    """)
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    conn = _connect()
    _migrate(conn)
    return conn


# ── KV namespace helpers (user-scoped) ────────────────────────────────────────
# owner_id='' means "global / system" (ENV-sourced, no user owns it).

def kv_get(namespace: str, key: str, owner_id: str = "") -> str | None:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT value FROM kv WHERE namespace=? AND key=? AND owner_id=?",
            (namespace, key, owner_id),
        ).fetchone()
        conn.close()
    return row["value"] if row else None


def kv_set(namespace: str, key: str, value: str, owner_id: str = "") -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO kv (namespace, key, owner_id, value) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(namespace, key, owner_id) DO UPDATE SET value=excluded.value",
            (namespace, key, owner_id, value),
        )
        conn.commit()
        conn.close()


def kv_delete(namespace: str, key: str, owner_id: str = "") -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "DELETE FROM kv WHERE namespace=? AND key=? AND owner_id=?",
            (namespace, key, owner_id),
        )
        conn.commit()
        conn.close()


def kv_all(namespace: str, owner_id: str = "") -> dict[str, str]:
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT key, value FROM kv WHERE namespace=? AND owner_id=?",
            (namespace, owner_id),
        ).fetchall()
        conn.close()
    return {r["key"]: r["value"] for r in rows}


# ── Provider credentials (user-scoped) ────────────────────────────────────────

_NS_PROVIDER = "provider_creds"


def load_provider_creds(owner_id: str) -> dict[str, dict[str, str]]:
    """Return {provider: {ENV_VAR: value}} for the given user."""
    rows = kv_all(_NS_PROVIDER, owner_id)
    result: dict[str, dict[str, str]] = {}
    for provider, raw in rows.items():
        try:
            result[provider] = json.loads(raw)
        except Exception:
            result[provider] = {}
    return result


def save_provider_creds(provider: str, values: dict[str, str], owner_id: str) -> None:
    """Persist env-var dict for *provider* under *owner_id*. Merges with existing.

    Saving new credentials automatically removes any tombstone so the provider
    is treated as configured again on the next restart.
    """
    existing_raw = kv_get(_NS_PROVIDER, provider, owner_id)
    existing: dict[str, str] = {}
    if existing_raw:
        try:
            existing = json.loads(existing_raw)
        except Exception:
            pass
    # Remove tombstone key so a re-save after clear restores the provider
    existing.pop(_CLEARED_SENTINEL, None)
    merged = {**existing, **{k: v for k, v in values.items() if v}}
    kv_set(_NS_PROVIDER, provider, json.dumps(merged), owner_id)


# Tombstone sentinel — written instead of deleting so bootstrap skips .env
# re-import for providers the user has explicitly cleared.
_CLEARED_SENTINEL = "__cleared__"


def is_provider_creds_cleared(provider: str, owner_id: str) -> bool:
    """Return True when the user has explicitly deleted this provider's credentials."""
    raw = kv_get(_NS_PROVIDER, provider, owner_id)
    if raw:
        try:
            return bool(json.loads(raw).get(_CLEARED_SENTINEL))
        except Exception:
            pass
    return False


def clear_provider_creds(provider: str, owner_id: str) -> None:
    """Record a tombstone so bootstrap won't re-import .env values on restart."""
    kv_set(_NS_PROVIDER, provider, json.dumps({_CLEARED_SENTINEL: True}), owner_id)


def get_provider_env_value(provider: str, env_key: str, owner_id: str) -> str | None:
    """Return the stored value for *env_key* under *provider* for *owner_id*."""
    raw = kv_get(_NS_PROVIDER, provider, owner_id)
    if raw:
        try:
            return json.loads(raw).get(env_key)
        except Exception:
            pass
    return None


# ── API keys (user-scoped) ─────────────────────────────────────────────────────

def load_api_keys(owner_id: str) -> dict[str, dict]:
    """Return all API keys owned by *owner_id*."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM api_keys WHERE owner_id=?", (owner_id,)
        ).fetchall()
        conn.close()
    return {r["kid"]: dict(r) for r in rows}


def save_api_key(kid: str, label: str, prefix: str, raw: str,
                 created_at: float, created_by: str, owner_id: str) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO api_keys (kid, label, prefix, raw, created_at, created_by, owner_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(kid) DO UPDATE SET label=excluded.label, raw=excluded.raw",
            (kid, label, prefix, raw, created_at, created_by, owner_id),
        )
        conn.commit()
        conn.close()


def delete_api_key(kid: str, owner_id: str) -> bool:
    """Delete key *kid* only if it belongs to *owner_id*."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM api_keys WHERE kid=? AND owner_id=?", (kid, owner_id)
        )
        conn.commit()
        conn.close()
    return cur.rowcount > 0


def load_all_api_keys() -> dict[str, dict]:
    """Return ALL API keys across all users — used by bootstrap to build the active key set."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM api_keys").fetchall()
        conn.close()
    return {r["kid"]: dict(r) for r in rows}


# ── Users ──────────────────────────────────────────────────────────────────────

def load_users() -> dict[str, dict]:
    with _lock:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM users").fetchall()
        conn.close()
    return {r["email"]: dict(r) for r in rows}


def save_user(email: str, user_id: str, name: str, hashed_password: str,
              salt: str, created_at: float) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO users (email, id, name, hashed_password, salt, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(email) DO UPDATE SET"
            "  name=excluded.name, hashed_password=excluded.hashed_password, salt=excluded.salt",
            (email, user_id, name, hashed_password, salt, created_at),
        )
        conn.commit()
        conn.close()


# ── Cost accumulator persistence ───────────────────────────────────────────────

_30D_HOURS = 720  # 30 * 24 — maximum retention window


def upsert_cost_bucket(
    provider: str,
    tier: str,
    task_type: str,
    tenant_id: str,
    hour_ts: int,
    cost_usd: float,
    requests: int,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Atomically increment a cost bucket row (UPSERT)."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT INTO cost_buckets
                (provider, tier, task_type, tenant_id, hour_ts,
                 cost_usd, requests, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, tier, task_type, tenant_id, hour_ts)
            DO UPDATE SET
                cost_usd      = cost_buckets.cost_usd      + excluded.cost_usd,
                requests      = cost_buckets.requests      + excluded.requests,
                input_tokens  = cost_buckets.input_tokens  + excluded.input_tokens,
                output_tokens = cost_buckets.output_tokens + excluded.output_tokens
            """,
            (provider, tier, task_type, tenant_id, hour_ts,
             cost_usd, requests, input_tokens, output_tokens),
        )
        conn.commit()
        conn.close()


def load_cost_buckets(cutoff_hour_ts: int) -> list[dict]:
    """Return all cost_buckets rows with hour_ts >= cutoff_hour_ts."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM cost_buckets WHERE hour_ts >= ?",
            (cutoff_hour_ts,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def prune_cost_buckets(cutoff_hour_ts: int) -> None:
    """Delete rows older than the retention window."""
    with _lock:
        conn = _get_conn()
        conn.execute("DELETE FROM cost_buckets WHERE hour_ts < ?", (cutoff_hour_ts,))
        conn.commit()
        conn.close()


# ── Request log persistence ────────────────────────────────────────────────────

def append_request_log(ts: float, data: str) -> None:
    """Insert one request record."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO request_log (ts, data) VALUES (?, ?)",
            (ts, data),
        )
        conn.commit()
        conn.close()


def load_request_log(limit: int) -> list[dict]:
    """Return the most-recent *limit* rows, newest first."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT data FROM request_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def prune_request_log(keep: int) -> None:
    """Keep only the newest *keep* rows."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "DELETE FROM request_log WHERE id NOT IN "
            "(SELECT id FROM request_log ORDER BY id DESC LIMIT ?)",
            (keep,),
        )
        conn.commit()
        conn.close()


# ── Catalog overrides (user-scoped) ────────────────────────────────────────────
# models.yaml is the immutable base.  This table stores per-user deltas so that
# one user's enable/disable/delete/add actions don't affect other users.
#
# patch JSON shape (all keys optional, combined as needed):
#   { "disabled": true }               — disabled for this user (exclude from routing)
#   { "deleted": true }                — treated as deleted for this user
#   { "added": { <DeploymentConfig fields> } } — added via UI (not in YAML)
#   { "fields": { "tier": "...", ... } }      — field-level overrides
#
# Reading: load all overrides for owner_id, merge onto the YAML base.
# Writing: upsert a single row per (name, owner_id).

import time as _time


def load_catalog_overrides(owner_id: str) -> dict[str, dict]:
    """Return {deployment_name: patch_dict} for *owner_id*."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT name, patch FROM catalog_overrides WHERE owner_id=?",
            (owner_id,),
        ).fetchall()
        conn.close()
    result: dict[str, dict] = {}
    for r in rows:
        try:
            result[r["name"]] = json.loads(r["patch"])
        except Exception:
            result[r["name"]] = {}
    return result


def load_all_catalog_overrides() -> dict[str, dict[str, dict]]:
    """Return {owner_id: {deployment_name: patch_dict}} — used by bootstrap."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT owner_id, name, patch FROM catalog_overrides ORDER BY updated_at ASC",
        ).fetchall()
        conn.close()
    result: dict[str, dict[str, dict]] = {}
    for r in rows:
        try:
            result.setdefault(r["owner_id"], {})[r["name"]] = json.loads(r["patch"])
        except Exception:
            pass
    return result


def save_catalog_override(name: str, patch: dict, owner_id: str) -> None:
    """Upsert an override patch for one deployment under *owner_id*.

    Merges with any existing patch so callers can send partial updates.
    """
    with _lock:
        conn = _get_conn()
        existing_row = conn.execute(
            "SELECT patch FROM catalog_overrides WHERE name=? AND owner_id=?",
            (name, owner_id),
        ).fetchone()
        existing: dict = {}
        if existing_row:
            try:
                existing = json.loads(existing_row["patch"])
            except Exception:
                pass
        merged = {**existing, **patch}
        conn.execute(
            "INSERT INTO catalog_overrides (name, owner_id, patch, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(name, owner_id) DO UPDATE SET"
            "  patch=excluded.patch, updated_at=excluded.updated_at",
            (name, owner_id, json.dumps(merged), _time.time()),
        )
        conn.commit()
        conn.close()


def delete_catalog_override(name: str, owner_id: str) -> None:
    """Remove all overrides for *name* under *owner_id* (fully restores YAML base)."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "DELETE FROM catalog_overrides WHERE name=? AND owner_id=?",
            (name, owner_id),
        )
        conn.commit()
        conn.close()

# ── Routing overrides (user-scoped) ────────────────────────────────────────────
# routing.yaml is the immutable base. This table stores per-user overrides for
# smart-routing flags, compression config, routing rules, classifier selection,
# and provider trust — all scoped per user, never touching routing.yaml.


def load_routing_override(namespace: str, owner_id: str) -> dict:
    """Return the stored override dict for (namespace, owner_id), or {}."""
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT value FROM routing_overrides WHERE namespace=? AND owner_id=?",
            (namespace, owner_id),
        ).fetchone()
        conn.close()
    if row:
        try:
            return json.loads(row["value"])
        except Exception:
            pass
    return {}


def save_routing_override(namespace: str, patch: dict, owner_id: str) -> None:
    """Merge *patch* into the stored override for (namespace, owner_id)."""
    with _lock:
        conn = _get_conn()
        existing_row = conn.execute(
            "SELECT value FROM routing_overrides WHERE namespace=? AND owner_id=?",
            (namespace, owner_id),
        ).fetchone()
        existing: dict = {}
        if existing_row:
            try:
                existing = json.loads(existing_row["value"])
            except Exception:
                pass
        merged = {**existing, **patch}
        conn.execute(
            "INSERT INTO routing_overrides (namespace, owner_id, value, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(namespace, owner_id) DO UPDATE SET"
            "  value=excluded.value, updated_at=excluded.updated_at",
            (namespace, owner_id, json.dumps(merged), _time.time()),
        )
        conn.commit()
        conn.close()


def load_all_routing_overrides() -> dict[str, dict[str, dict]]:
    """Return {owner_id: {namespace: value_dict}} — used by bootstrap."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT owner_id, namespace, value FROM routing_overrides ORDER BY updated_at ASC",
        ).fetchall()
        conn.close()
    result: dict[str, dict[str, dict]] = {}
    for r in rows:
        try:
            result.setdefault(r["owner_id"], {})[r["namespace"]] = json.loads(r["value"])
        except Exception:
            pass
    return result
