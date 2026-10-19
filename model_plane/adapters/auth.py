"""FastAPI authentication middleware and helpers."""

from __future__ import annotations

import hashlib
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from model_plane.config import settings
from model_plane.logging_setup import get_logger

log = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# Pre-hash configured keys at startup
_ALLOWED_KEY_HASHES: set[str] = set()

# Map key_hash → owner_id for keys created via the admin UI (stored in DB).
# Keyed by SHA-256 hash (not raw key) so the map cannot leak credentials.
# Built at startup and refreshed after key create / delete.
_KEY_OWNER_MAP: dict[str, str] = {}


def _load_key_hashes() -> None:
    global _ALLOWED_KEY_HASHES, _KEY_OWNER_MAP
    _ALLOWED_KEY_HASHES = {_hash_key(k) for k in settings.api_key_list if k}
    new_owner_map: dict[str, str] = {}
    try:
        from model_plane.db import load_all_api_keys
        for row in load_all_api_keys().values():
            raw = row.get("raw", "")
            owner = row.get("owner_id", "")
            if raw and owner:
                h = _hash_key(raw)
                _ALLOWED_KEY_HASHES.add(h)
                new_owner_map[h] = owner
    except Exception as exc:
        log.warning("api_key_owner_map_load_failed", error=str(exc))
    _KEY_OWNER_MAP = new_owner_map


def resolve_key_owner(token: str) -> str:
    """Return the owner_id for a raw API key, or '' if not found."""
    return _KEY_OWNER_MAP.get(_hash_key(token), "")


def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    """FastAPI dependency that validates Bearer token against configured API keys."""
    # Dev mode: if no env-configured keys, allow all traffic.
    # DB keys (in _KEY_OWNER_MAP) enrich identity but do not enable auth by themselves.
    if not settings.api_key_list:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    if _hash_key(token) not in _ALLOWED_KEY_HASHES:
        log.warning("invalid_api_key_attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
