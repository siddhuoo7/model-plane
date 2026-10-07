"""AWS Bedrock provider adapter.

LiteLLM docs: https://docs.litellm.ai/docs/providers/bedrock

Two authentication modes (mutually exclusive — API key wins if both are set):

  API-key mode (recommended when you have a Bedrock API key):
    Set AWS_BEARER_TOKEN_BEDROCK=<your-key>
    LiteLLM will pass it as an Authorization: Bearer header directly.
    No AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY needed.

  IAM / boto3 mode (traditional):
    Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_REGION_NAME
    (or use an instance profile / ECS task role).
"""

from __future__ import annotations

import os

from model_plane.config import settings
from model_plane.registry.catalog import DeploymentConfig


class BedrockAdapter:
    provider_name = "bedrock"

    def build_kwargs(self, dep: DeploymentConfig) -> dict:
        kwargs: dict = {}

        region = (
            dep.extra.get("aws_region_name")
            or settings.aws_region_name
            or os.environ.get("AWS_REGION_NAME")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        if region:
            kwargs["aws_region_name"] = region

        # API-key mode: AWS_BEARER_TOKEN_BEDROCK is read from os.environ by LiteLLM
        # automatically and used as an Authorization: Bearer header.
        # Do NOT pass it as a kwarg — LiteLLM would stuff it into
        # additionalModelRequestFields in the JSON body instead of the header.
        bearer_token = (
            settings.aws_bearer_token_bedrock
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        )
        if bearer_token:
            # Ensure the env var is set so LiteLLM picks it up on this call.
            # (_apply_creds_to_process already does this at login; this is a
            # belt-and-suspenders guard for the bootstrap / test path.)
            os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", bearer_token)
            # Disable boto3 EC2 IMDS lookup — it adds ~1 s latency and noisy
            # "Host is down" logs when running outside AWS without IAM creds.
            os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
            return kwargs

        # IAM / assume-role mode
        role_name = dep.extra.get("aws_role_name") or os.environ.get("AWS_ROLE_NAME")
        if role_name:
            kwargs["aws_role_name"] = role_name

        session_token = os.environ.get("AWS_SESSION_TOKEN")
        if session_token:
            kwargs["aws_session_token"] = session_token

        # Explicit access key (overrides env / instance profile)
        access_key = settings.aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = settings.aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key

        return kwargs

    def _region(self) -> str:
        return (
            settings.aws_region_name
            or os.environ.get("AWS_REGION_NAME")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

    def list_models(self) -> list[str]:
        """Call Bedrock ListFoundationModels to get available model IDs.

        Uses the bearer-token (API key) path first, then falls back to
        boto3 IAM-credential mode if the bearer token is absent.
        Returns bare model IDs (e.g. "anthropic.claude-3-haiku-20240307-v1:0").
        """
        import httpx

        region = self._region()
        bearer = (
            settings.aws_bearer_token_bedrock
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        )

        if bearer:
            # API-key / bearer-token mode — call the Bedrock REST API directly.
            url = f"https://bedrock.{region}.amazonaws.com/foundation-models"
            try:
                resp = httpx.get(
                    url,
                    headers={"Authorization": f"Bearer {bearer}"},
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    m.get("modelId", "")
                    for m in data.get("modelSummaries", [])
                    if m.get("modelId")
                ]
            except Exception:
                return []

        # IAM mode — use boto3 if available.
        try:
            import boto3  # type: ignore[import]

            client = boto3.client(
                "bedrock",
                region_name=region,
                aws_access_key_id=settings.aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=settings.aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            )
            resp = client.list_foundation_models()
            return [m["modelId"] for m in resp.get("modelSummaries", [])]
        except Exception:
            return []

    def health_probe(self, dep: DeploymentConfig) -> bool:  # noqa: ARG002
        """Real probe via model listing API."""
        return len(self.list_models()) > 0
