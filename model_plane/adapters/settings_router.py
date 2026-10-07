"""Admin Settings API — API key management, user authentication, provider credentials.

Endpoints:
  POST   /admin/api/auth/signup         — create first user (email + password)
  POST   /admin/api/auth/login          — get JWT session token
  GET    /admin/api/auth/me             — current user info
  GET    /admin/api/settings/api-keys   — list API keys (masked)
  POST   /admin/api/settings/api-keys   — generate a new API key
  DELETE /admin/api/settings/api-keys/{key_id} — revoke a key
  GET    /admin/api/settings/providers  — credential status per provider
  PUT    /admin/api/settings/providers/{provider} — save provider credentials
  DELETE /admin/api/settings/providers/{provider} — clear UI credentials

Persistence strategy
--------------------
All UI-entered values are stored in SQLite (config/model_plane.db) — the
primary source of truth.  .env / os.environ are treated as **read-only
fallbacks**: values already present there continue to work if no SQLite row
exists, but writing back to .env is no longer done.  This keeps the file-based
config clean and makes the UI the authoritative configuration surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

import base64
import hmac

from model_plane.config import settings
from model_plane.logging_setup import get_logger
import model_plane.db as db

log = get_logger(__name__)

router = APIRouter(prefix="/admin/api", tags=["settings"])

# ── JWT (minimal, no external dep) ───────────────────────────────────────────

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * (padding % 4))


def _jwt_sign(payload: dict) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body   = _b64(json.dumps(payload).encode())
    secret = settings.jwt_secret.encode()
    sig    = _b64(hmac.new(secret, f"{header}.{body}".encode(), hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"


def _jwt_verify(token: str) -> dict:
    try:
        header, body, sig = token.split(".")
        secret = settings.jwt_secret.encode()
        expected = _b64(hmac.new(secret, f"{header}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        payload = json.loads(_b64d(body))
        if payload.get("exp", 0) < time.time():
            raise ValueError("token expired")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc


# ── password hashing ──────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return hashed, salt


def _verify_password(password: str, hashed: str, salt: str) -> bool:
    return hmac.compare_digest(_hash_password(password, salt)[0], hashed)


# ── JWT auth dependency ───────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _require_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    return _jwt_verify(credentials.credentials)


# ── Models ────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/auth/signup", status_code=201)
async def auth_signup(body: SignupRequest) -> dict:
    """Create the first admin user."""
    email = body.email.strip().lower()
    if not email or not body.password:
        raise HTTPException(status_code=422, detail="email and password are required")

    users = db.load_users()
    if users:
        raise HTTPException(
            status_code=409,
            detail="A user already exists. Use /auth/login or contact your admin.",
        )
    hashed, salt = _hash_password(body.password)
    user_id = secrets.token_hex(8)
    name = body.name or email.split("@")[0]
    db.save_user(email, user_id, name, hashed, salt, time.time())
    log.info("user_created", email=email)

    token = _jwt_sign({"sub": email, "id": user_id, "exp": time.time() + 86400 * 30})
    return {"token": token, "email": email, "name": name}


@router.post("/auth/login")
async def auth_login(body: LoginRequest) -> dict:
    email = body.email.strip().lower()
    users = db.load_users()
    user = users.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not _verify_password(body.password, user["hashed_password"], user["salt"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Apply this user's stored provider credentials to the running process so
    # routing works with their keys from the moment they log in.
    owner_id = user["id"]
    try:
        all_creds = db.load_provider_creds(owner_id)
        for provider, values in all_creds.items():
            if values:
                _apply_creds_to_process(provider, values)
    except Exception as exc:
        log.warning("login_creds_apply_failed", error=str(exc))

    token = _jwt_sign({
        "sub": email,
        "id": owner_id,
        "exp": time.time() + 86400 * 30,
    })
    log.info("user_login", email=email)
    return {"token": token, "email": email, "name": user.get("name", "")}


@router.get("/auth/me")
async def auth_me(session: dict = Depends(_require_session)) -> dict:
    email = session.get("sub", "")
    users = db.load_users()
    user = users.get(email, {})
    return {
        "email": email,
        "name": user.get("name", ""),
        "id": user.get("id", ""),
    }


@router.get("/auth/has-users")
async def auth_has_users() -> dict:
    """Check whether any user account exists (used to show signup vs login)."""
    return {"has_users": bool(db.load_users())}


# ─────────────────────────────────────────────────────────────────────────────
# API key management endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/settings/api-keys")
async def api_keys_list(session: dict = Depends(_require_session)) -> dict:
    """Return the calling user's API keys (masked — raw shown only at creation)."""
    owner_id = session.get("id", "")
    keys = db.load_api_keys(owner_id)
    return {
        "keys": [
            {
                "id": kid,
                "label": v.get("label", ""),
                "prefix": v.get("prefix", ""),
                "created_at": v.get("created_at"),
            }
            for kid, v in keys.items()
        ]
    }


@router.post("/settings/api-keys", status_code=201)
async def api_keys_create(
    body: dict = Body(default={}),
    session: dict = Depends(_require_session),
) -> dict:
    """Generate a new random API key owned by the calling user."""
    owner_id = session.get("id", "")
    label = (body.get("label") or "").strip() or "API Key"
    raw   = "mp-" + secrets.token_urlsafe(32)
    kid   = secrets.token_hex(8)
    prefix = raw[:12] + "..."

    db.save_api_key(kid, label, prefix, raw, time.time(), session.get("sub", ""), owner_id)
    _reload_api_keys_in_settings()

    log.info("api_key_created", kid=kid, label=label, owner=session.get("sub", ""))
    return {"id": kid, "label": label, "key": raw, "prefix": prefix}


@router.delete("/settings/api-keys/{key_id}", status_code=204)
async def api_keys_delete(
    key_id: str = Path(...),
    session: dict = Depends(_require_session),
) -> None:
    """Revoke an API key.  Only the owning user can delete their own keys."""
    owner_id = session.get("id", "")
    if not db.delete_api_key(key_id, owner_id):
        raise HTTPException(status_code=404, detail="Key not found")
    _reload_api_keys_in_settings()
    log.info("api_key_deleted", kid=key_id, owner=session.get("sub", ""))


def _reload_api_keys_in_settings() -> None:
    """Rebuild MODEL_PLANE_API_KEYS in the live settings object from SQLite.

    Loads keys from ALL users so every user's keys remain active for
    incoming /v1/* requests regardless of who is currently logged in.
    """
    keys = db.load_all_api_keys()
    raw_keys = [v["raw"] for v in keys.values() if v.get("raw")]
    value = ",".join(raw_keys)
    try:
        object.__setattr__(settings, "api_keys", value)
    except Exception:
        pass
    try:
        from model_plane.adapters.auth import _load_key_hashes
        _load_key_hashes()
    except Exception as exc:
        log.warning("auth_key_reload_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Provider credentials management
# ─────────────────────────────────────────────────────────────────────────────
#
# Persistence strategy:
#   1. Save to SQLite (config/model_plane.db) — primary store.
#   2. Sync live values into os.environ + settings object for the running process.
#   3. .env is NOT written — it remains a read-only fallback.
#
# On every request that checks credentials, SQLite is consulted first.
# If no SQLite row exists, os.environ / settings is the fallback (values loaded
# from .env at process start remain active).
# ─────────────────────────────────────────────────────────────────────────────

# Maps provider name → list of env var names it needs
_PROVIDER_ENV_VARS: dict[str, list[str]] = {
    "openai":    ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "watsonx":   ["WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL"],
    "bedrock":   ["AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION_NAME"],
    "azure":     ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"],
    "vertex_ai": ["VERTEXAI_PROJECT", "VERTEXAI_LOCATION"],
    "cohere":    ["COHERE_API_KEY"],
    "mistral":   ["MISTRAL_API_KEY"],
    "local":     ["LOCAL_VLLM_API_BASE", "LOCAL_VLLM_API_KEY"],
}

# Maps env var name → settings attribute name
_ENV_TO_SETTINGS: dict[str, str] = {
    "OPENAI_API_KEY":           "openai_api_key",
    "ANTHROPIC_API_KEY":        "anthropic_api_key",
    "WATSONX_APIKEY":           "watsonx_api_key",
    "WATSONX_PROJECT_ID":       "watsonx_project_id",
    "WATSONX_URL":              "watsonx_url",
    "AWS_BEARER_TOKEN_BEDROCK": "aws_bearer_token_bedrock",
    "AWS_REGION_NAME":          "aws_region_name",
    "AZURE_API_KEY":            "azure_api_key",
    "AZURE_API_BASE":           "azure_api_base",
    "AZURE_API_VERSION":        "azure_api_version",
    "VERTEXAI_PROJECT":         "vertex_project",
    "VERTEXAI_LOCATION":        "vertex_location",
    "COHERE_API_KEY":           "cohere_api_key",
    "MISTRAL_API_KEY":          "mistral_api_key",
    "LOCAL_VLLM_API_BASE":      "local_vllm_api_base",
    "LOCAL_VLLM_API_KEY":       "local_vllm_api_key",
}


def _sanitize_credential_value(val: str) -> str:
    """Strip non-printable and non-ASCII characters from a credential value.

    API keys and tokens must be pure ASCII.  Values pasted from browsers or
    terminals sometimes carry invisible Unicode characters (zero-width spaces,
    BOM, soft-hyphens, etc.) that cause httpx / LiteLLM to raise
    ``UnicodeEncodeError`` when the value is placed in an HTTP header.
    """
    # Strip leading/trailing whitespace first (covers common copy-paste issues)
    val = val.strip()
    # Keep only printable ASCII characters (0x20-0x7E); drop everything else.
    return "".join(ch for ch in val if 0x20 <= ord(ch) <= 0x7E)


def _apply_creds_to_process(provider: str, values: dict[str, str]) -> None:
    """Sync non-empty values into os.environ and the live settings object.

    Does NOT write to .env — .env is a read-only fallback.
    Values are sanitized to printable ASCII before being applied so that
    non-ASCII characters pasted from browsers never cause UnicodeEncodeError
    in HTTP headers.
    """
    for env_key, val in values.items():
        if not val:
            continue
        clean = _sanitize_credential_value(val)
        if not clean:
            log.warning(
                "credential_value_empty_after_sanitize",
                provider=provider,
                key=env_key,
            )
            continue
        os.environ[env_key] = clean
        attr = _ENV_TO_SETTINGS.get(env_key)
        if attr and hasattr(settings, attr):
            try:
                object.__setattr__(settings, attr, clean)
            except Exception:
                pass

    # When using Bedrock bearer-token mode, disable boto3's EC2 IMDS lookup.
    # Without this boto3 tries http://169.254.169.254/ for instance credentials
    # before every call, adding ~1 s latency and noisy connection-refused logs.
    if "AWS_BEARER_TOKEN_BEDROCK" in values and values.get("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

    log.info("provider_creds_applied_to_process", provider=provider, keys=list(values.keys()))


def _load_provider_creds_for_user(owner_id: str) -> dict[str, dict[str, str | bool]]:
    """Return masked credential status for *owner_id* — user SQLite wins, ENV is fallback.

    If the user has explicitly cleared a provider (tombstone in SQLite), the ENV
    fallback is suppressed so the UI shows 'not set' even when .env has a value.
    """
    sqlite_creds = db.load_provider_creds(owner_id)
    result: dict[str, dict[str, str | bool]] = {}
    for provider, env_vars in _PROVIDER_ENV_VARS.items():
        saved = sqlite_creds.get(provider, {})
        # If user cleared this provider, suppress the .env fallback entirely
        cleared = db.is_provider_creds_cleared(provider, owner_id)
        result[provider] = {
            env_key: bool(saved.get(env_key) or (not cleared and os.environ.get(env_key)))
            for env_key in env_vars
        }
    return result


@router.get("/settings/providers")
async def provider_creds_list(session: dict = Depends(_require_session)) -> dict:
    """Return credential status for the calling user (their SQLite rows + ENV fallback)."""
    owner_id = session.get("id", "")
    return {"providers": _load_provider_creds_for_user(owner_id)}


@router.get("/settings/providers/{provider}/values")
async def provider_creds_values(
    provider: str = Path(...),
    session: dict = Depends(_require_session),
) -> dict:
    """Return the actual stored credential values for *provider* for the calling user.

    Sensitive fields (keys containing 'KEY', 'TOKEN', 'SECRET', 'PASSWORD') are
    masked to a placeholder string so the UI can show the field is set without
    leaking the raw secret.  Non-sensitive fields (URLs, regions, project IDs)
    are returned in full so they pre-fill correctly in the edit form.
    """
    if provider not in _PROVIDER_ENV_VARS:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'")

    owner_id = session.get("id", "")
    sqlite_row = db.load_provider_creds(owner_id).get(provider, {})

    _SENSITIVE = {"KEY", "TOKEN", "SECRET", "PASSWORD"}

    def _mask(env_key: str, val: str) -> str:
        upper = env_key.upper()
        if any(s in upper for s in _SENSITIVE):
            return "••••••••"   # placeholder — tells UI the field is set
        return val              # return URLs, regions, IDs in full

    values: dict[str, str] = {}
    for env_key in _PROVIDER_ENV_VARS[provider]:
        # SQLite row wins; fall back to os.environ for .env-sourced values
        val = sqlite_row.get(env_key) or os.environ.get(env_key) or ""
        values[env_key] = _mask(env_key, val) if val else ""

    return {"provider": provider, "values": values}


@router.put("/settings/providers/{provider}")
async def provider_creds_save(
    provider: str = Path(...),
    body: dict = Body(...),
    session: dict = Depends(_require_session),
) -> dict:
    """Save credentials for a provider, scoped to the calling user.

    Body: dict of env_var_name → value.  Empty string = keep existing value.
    Values are stored in SQLite under the user's own owner_id and immediately
    synced to os.environ/settings so the change takes effect without a restart.
    .env is NOT modified.  Another logged-in user cannot see or overwrite these.
    """
    if provider not in _PROVIDER_ENV_VARS:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'")

    owner_id = session.get("id", "")
    allowed_keys = set(_PROVIDER_ENV_VARS[provider])
    # Sanitize values to printable ASCII before saving — prevents non-ASCII
    # characters pasted from browsers from reaching os.environ / HTTP headers.
    values = {
        k: _sanitize_credential_value(v)
        for k, v in body.items()
        if k in allowed_keys and isinstance(v, str)
    }

    to_save = {k: v for k, v in values.items() if v}
    if to_save:
        # Saving new credentials clears any previous tombstone for this provider
        db.save_provider_creds(provider, to_save, owner_id)
        _apply_creds_to_process(provider, to_save)

    # Reload catalog so credential-filtered views reflect the change immediately
    from model_plane.registry.catalog import reload_catalog
    reload_catalog()

    # Re-validate the catalog against live provider model lists in the background
    # so the newly-saved credentials are used to mark deployments healthy/unhealthy
    # without the user needing to restart or wait for the next startup probe.
    import threading as _threading
    from model_plane.app import _validate_catalog_against_providers
    _threading.Thread(
        target=_validate_catalog_against_providers,
        name=f"catalog-validation-{provider}",
        daemon=True,
    ).start()

    # Return masked status for this user
    updated_raw = db.load_provider_creds(owner_id).get(provider, {})
    updated = {k: bool(updated_raw.get(k) or os.environ.get(k)) for k in allowed_keys}
    log.info("provider_creds_saved", provider=provider, owner=session.get("sub", ""))
    return {"provider": provider, "configured": updated}


@router.delete("/settings/providers/{provider}", status_code=204)
async def provider_creds_clear(
    provider: str = Path(...),
    session: dict = Depends(_require_session),
) -> None:
    """Remove the calling user's SQLite credentials for a provider.

    Also removes the env vars from the live process (os.environ) so the UI
    immediately reflects the change without needing a restart.  .env values
    are NOT touched — if the user restarts the server with keys in .env they
    will reappear via the normal bootstrap path.
    """
    owner_id = session.get("id", "")
    db.clear_provider_creds(provider, owner_id)

    # Clear env vars so _load_provider_creds_for_user no longer falls back to them
    if provider in _PROVIDER_ENV_VARS:
        for env_key in _PROVIDER_ENV_VARS[provider]:
            os.environ.pop(env_key, None)
            attr = _ENV_TO_SETTINGS.get(env_key)
            if attr and hasattr(settings, attr):
                try:
                    object.__setattr__(settings, attr, None)
                except Exception:
                    pass

    from model_plane.registry.catalog import reload_catalog
    reload_catalog()
    log.info("provider_creds_cleared", provider=provider, owner=session.get("sub", ""))


# ── Start-up bootstrap ───────────────────────────────────────────────────────

def bootstrap_credentials_from_db() -> None:
    """Called once at server start.

    1. Loads all API keys from SQLite into the in-memory settings so that
       incoming /v1/* requests can authenticate regardless of which user
       created the key.

    2. Loads ALL provider credentials stored by any user and writes them into
       os.environ so that provider_has_creds() and LiteLLM work immediately
       on startup — without requiring a user login or Test connection click.

       The last value written for each env-var wins (single-tenant assumption:
       all admins share the same provider config).  Values already present in
       os.environ (from a .env file) are NOT overwritten — .env always wins as
       a hard-coded fallback.
    """
    # 1 — API keys
    try:
        _reload_api_keys_in_settings()
        log.info("db_api_keys_bootstrapped")
    except Exception as exc:
        log.warning("db_api_keys_bootstrap_failed", error=str(exc))

    # 2 — Provider credentials: load from SQLite → os.environ
    try:
        all_users = db.load_users()          # {email: {id, ...}}
        applied: list[str] = []
        for user_info in all_users.values():
            owner_id = user_info.get("id", "")
            if not owner_id:
                continue
            user_creds = db.load_provider_creds(owner_id)  # {provider: {ENV_VAR: val}}
            for provider, env_map in user_creds.items():
                # Skip tombstoned providers — user explicitly cleared them, do not
                # re-import .env values on restart for this provider.
                if db.is_provider_creds_cleared(provider, owner_id):
                    # Also strip any env vars so the running process reflects the clear
                    if provider in _PROVIDER_ENV_VARS:
                        for env_key in _PROVIDER_ENV_VARS[provider]:
                            os.environ.pop(env_key, None)
                    continue
                for env_key, env_val in env_map.items():
                    if env_val and not os.environ.get(env_key):
                        # Only set if not already present (.env takes priority)
                        os.environ[env_key] = env_val
                        applied.append(env_key)
        if applied:
            log.info("db_provider_creds_bootstrapped", env_vars=applied)
        else:
            log.info("db_provider_creds_bootstrapped", env_vars=[])
    except Exception as exc:
        log.warning("db_provider_creds_bootstrap_failed", error=str(exc))

    # 3 — Catalog overrides: apply the merged set of all users' overrides onto
    #     the global routing catalog so routing works correctly on restart.
    #     Last-writer-wins across users (consistent with provider creds behaviour).
    try:
        from model_plane.registry.catalog import get_catalog
        base = get_catalog()
        all_catalog_overrides = db.load_all_catalog_overrides()
        # Merge all users' overrides into one combined patch (last-writer-wins)
        merged: dict[str, dict] = {}
        for _uid, user_patches in all_catalog_overrides.items():
            for dep_name, patch in user_patches.items():
                merged[dep_name] = {**merged.get(dep_name, {}), **patch}

        applied_names: list[str] = []
        import copy as _copy
        for dep_name, patch in merged.items():
            if patch.get("deleted"):
                if dep_name in base.deployments:
                    del base.deployments[dep_name]
                    applied_names.append(f"-{dep_name}")
                continue
            if "added" in patch and dep_name not in base.deployments:
                from model_plane.registry.catalog import DeploymentConfig
                added = patch["added"]
                try:
                    dep = DeploymentConfig(
                        name=dep_name,
                        litellm_model=added.get("litellm_model", ""),
                        provider=added.get("provider", "openai"),
                        tier=added.get("tier", "medium"),
                        context_limit=added.get("context_limit", 8192),
                        cost_per_1k_input=added.get("cost_per_1k_input", 0.0),
                        cost_per_1k_output=added.get("cost_per_1k_output", 0.0),
                        capabilities=list(added.get("capabilities", [])),
                        tags=list(added.get("tags", [])),
                        fallback_to=list(added.get("fallback_to", [])),
                        max_parallel_requests=added.get("max_parallel_requests", 50),
                        rpm=added.get("rpm"),
                        api_base=added.get("api_base"),
                        api_key_env=added.get("api_key_env"),
                        healthy=added.get("healthy", True),
                    )
                    base.deployments[dep_name] = dep
                    applied_names.append(f"+{dep_name}")
                except Exception:
                    pass
            if dep_name in base.deployments:
                dep = base.deployments[dep_name]
                if patch.get("disabled"):
                    dep.healthy = False
                for fk, fv in patch.get("fields", {}).items():
                    if hasattr(dep, fk):
                        setattr(dep, fk, fv)
                if applied_names and not applied_names[-1].endswith(dep_name):
                    applied_names.append(f"~{dep_name}")
        # Rebuild tier_map
        base.tier_map = {}
        for d in base.deployments.values():
            base.tier_map.setdefault(d.tier, []).append(d.name)
        log.info("db_catalog_overrides_bootstrapped", changes=applied_names)
    except Exception as exc:
        log.warning("db_catalog_overrides_bootstrap_failed", error=str(exc))
