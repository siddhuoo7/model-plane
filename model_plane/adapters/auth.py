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


def _load_key_hashes() -> None:
    global _ALLOWED_KEY_HASHES
    _ALLOWED_KEY_HASHES = {_hash_key(k) for k in settings.api_key_list if k}


def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    """FastAPI dependency that validates Bearer token against configured API keys."""
    # If no keys configured, allow all (dev mode)
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
