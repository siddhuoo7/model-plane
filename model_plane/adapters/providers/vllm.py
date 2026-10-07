"""Local vLLM provider adapter.

Migrated from executor.py ``_aexecute_local_vllm`` auth/header logic.
``build_kwargs`` only returns the bearer-token header dict; the actual HTTP
call is still performed by ``_aexecute_local_vllm`` in executor.py so that
the streaming + retry path is unchanged.
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig


class VLLMAdapter:
    provider_name = "local"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:
        """Return auth headers for the local vLLM HTTP call.

        The executor's ``_aexecute_local_vllm`` function assembles the full
        httpx request; this adapter just centralises the credential resolution
        so that the provider-specific logic is not scattered in executor.py.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        if dep.api_key_env:
            api_key = settings.local_vllm_api_key or os.environ.get(dep.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        local_vllm_cookie = os.environ.get("LOCAL_VLLM_COOKIE")
        if local_vllm_cookie:
            headers["Cookie"] = local_vllm_cookie

        return {"_vllm_headers": headers}

    def _resolve_base_url(self, dep: DeploymentConfig) -> str | None:
        """Return the base URL for this deployment, trying dep.api_base first
        then the LOCAL_VLLM_API_BASE environment variable."""
        return (
            dep.api_base
            or os.environ.get("LOCAL_VLLM_API_BASE")
            or os.environ.get("MODEL_PLANE_LOCAL_VLLM_API_BASE")
        )

    def list_models(self, dep: DeploymentConfig | None = None) -> list[str]:
        """GET {base}/v1/models and return the list of model IDs.

        Raises RuntimeError with a descriptive message when the endpoint is
        unreachable so the test endpoint can surface it to the UI instead of
        silently returning an empty list.
        """
        import httpx

        # Resolve base URL: dep.api_base → LOCAL_VLLM_API_BASE env var.
        # os.environ is checked *after* _apply_creds_to_process has run so
        # that a URL just saved via the credentials form is picked up here.
        if dep is not None:
            base = self._resolve_base_url(dep)
        else:
            base = (
                os.environ.get("LOCAL_VLLM_API_BASE")
                or os.environ.get("MODEL_PLANE_LOCAL_VLLM_API_BASE")
                or getattr(settings, "local_vllm_api_base", None)
            )
        if not base:
            raise RuntimeError(
                "No API base URL configured for Local / vLLM. "
                "Set the API base URL in the Configure modal and save."
            )
        base = base.rstrip("/")

        headers: dict[str, str] = {}
        api_key = (
            os.environ.get("LOCAL_VLLM_API_KEY")
            or getattr(settings, "local_vllm_api_key", None)
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        verify_ssl = not settings.local_vllm_insecure_skip_verify
        url = f"{base}/v1/models"
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0, verify=verify_ssl)
        except httpx.ConnectError:
            raise RuntimeError(f"Connection refused at {url} — is vLLM running?")
        except httpx.TimeoutException:
            raise RuntimeError(f"Timed out connecting to {url}")
        except Exception as exc:
            raise RuntimeError(f"{type(exc).__name__}: {exc}")

        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")

        data = resp.json()
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]

    def health_probe(self, dep: DeploymentConfig) -> bool:
        """Make a real HTTP GET to the vLLM server to verify it is reachable.

        Tries ``{base}/health`` first (vLLM ≥0.4 standard endpoint),
        then falls back to ``{base}/v1/models``.

        Raises ``RuntimeError`` with a descriptive message when the endpoint
        is unreachable — the caller (``provider_test``) catches this and
        surfaces it as ``healthy=False`` with the error text.
        """
        import httpx

        base = self._resolve_base_url(dep)
        if not base:
            raise RuntimeError(
                "No api_base configured for this deployment and LOCAL_VLLM_API_BASE "
                "is not set. Add api_base to the deployment in config/models.yaml or "
                "set LOCAL_VLLM_API_BASE in your .env file."
            )

        base = base.rstrip("/")
        headers: dict[str, str] = {}
        api_key = settings.local_vllm_api_key or os.environ.get("LOCAL_VLLM_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        verify_ssl = not settings.local_vllm_insecure_skip_verify
        timeout = 5.0  # seconds — fast fail for a health check

        last_error: str = ""
        for path in ("/health", "/v1/models"):
            url = f"{base}{path}"
            try:
                resp = httpx.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
                if resp.status_code < 500:
                    return True
                last_error = f"HTTP {resp.status_code} from {url}"
            except httpx.ConnectError:
                last_error = f"Connection refused at {url} — is vLLM running?"
            except httpx.TimeoutException:
                last_error = f"Timed out connecting to {url} (>{timeout}s)"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        raise RuntimeError(last_error)
