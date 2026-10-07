"""WatsonX provider adapter.

Migrated verbatim from executor.py ``_build_litellm_kwargs`` watsonx block.
No behaviour change.
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig


class WatsonxAdapter:
    provider_name = "watsonx"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:
        kwargs: dict = {}

        # 1. project_id — passed as completion() param AND set in os.environ
        project_id = (
            settings.watsonx_project_id
            or os.environ.get("WATSONX_PROJECT_ID")
            or os.environ.get("WX_PROJECT_ID")
        )
        if project_id:
            kwargs["project_id"] = project_id          # direct completion() param
            os.environ["WATSONX_PROJECT_ID"] = project_id   # LiteLLM env fallback

        # 2. URL — required
        wx_url = (
            settings.watsonx_url
            or os.environ.get("WATSONX_URL")
            or "https://us-south.ml.cloud.ibm.com"
        )
        kwargs["api_base"] = wx_url
        os.environ["WATSONX_URL"] = wx_url

        # 3. API key — LiteLLM reads WATSONX_APIKEY from env automatically;
        #    we ensure it is in os.environ using the correct name.
        api_key = (
            settings.watsonx_api_key                    # read from WATSONX_APIKEY or WATSONX_API_KEY
            or os.environ.get("WATSONX_APIKEY")
            or os.environ.get("WATSONX_API_KEY")
        )
        if api_key:
            os.environ["WATSONX_APIKEY"] = api_key      # canonical name LiteLLM reads

        return kwargs

    def _iam_token(self, api_key: str) -> str:
        """Exchange a watsonx IAM API key for a short-lived IAM access token.

        The watsonx REST API requires a Bearer token, not the raw API key.
        This is a blocking call; use asyncio.to_thread() at the call-site.
        Raises RuntimeError on failure.
        """
        import httpx

        try:
            resp = httpx.post(
                "https://iam.cloud.ibm.com/identity/token",
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": api_key,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            resp.raise_for_status()
            token = resp.json().get("access_token", "")
            if not token:
                raise RuntimeError("IAM token response missing access_token")
            return token
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"IAM token exchange failed: {exc}") from exc

    def list_models(self) -> list[str]:
        """List available watsonx foundation models.

        Exchanges the IAM API key for an access token first, then calls
        GET /ml/v1/foundation_model_specs.
        Raises RuntimeError with a descriptive message on any failure.
        """
        import httpx

        api_key = (
            os.environ.get("WATSONX_APIKEY")
            or os.environ.get("WATSONX_API_KEY")
            or settings.watsonx_api_key
        )
        if not api_key:
            raise RuntimeError(
                "watsonx API key not configured. "
                "Save the API key via the Configure modal first."
            )

        wx_url = (
            os.environ.get("WATSONX_URL")
            or settings.watsonx_url
            or "https://us-south.ml.cloud.ibm.com"
        ).rstrip("/")

        # Exchange API key for IAM access token
        access_token = self._iam_token(api_key)

        url = f"{wx_url}/ml/v1/foundation_model_specs?version=2024-09-16&limit=200"
        try:
            resp = httpx.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15.0,
            )
        except Exception as exc:
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc

        if resp.status_code >= 400:
            raise RuntimeError(
                f"HTTP {resp.status_code} from watsonx: {resp.text[:200]}"
            )

        data = resp.json()
        return [m.get("model_id", "") for m in data.get("resources", []) if m.get("model_id")]

    def health_probe(self, dep: DeploymentConfig) -> bool:  # noqa: ARG002
        """Live probe via model listing."""
        return len(self.list_models()) > 0
