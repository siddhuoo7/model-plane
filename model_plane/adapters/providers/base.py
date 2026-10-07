"""Provider adapter protocol — all provider-specific kwargs builders implement this."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from model_plane.registry.catalog import DeploymentConfig


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol that every provider adapter must satisfy.

    ``build_kwargs`` receives a fully-resolved ``DeploymentConfig`` and returns
    a dict that is merged into the LiteLLM ``acompletion`` call-site kwargs.
    Return an empty dict when no provider-specific augmentation is needed.
    """

    provider_name: str

    def build_kwargs(self, dep: DeploymentConfig) -> dict:
        """Return provider-specific kwargs to inject into the LiteLLM call."""
        ...

    def health_probe(self, dep: DeploymentConfig) -> bool:
        """Send a minimal 1-token probe to verify the deployment is reachable.

        Returns True if the deployment responds successfully, False otherwise.
        The default implementation returns True (optimistic / skip probe).
        Concrete adapters may override this with a real liveness check.
        """
        ...

    def list_models(self) -> list[str]:
        """Fetch the list of model IDs available from this provider's API.

        Returns a list of model ID strings (bare, without provider prefix).
        Returns an empty list when the provider does not support model listing
        or when credentials are missing / the call fails.
        Concrete adapters should override this to make a real network call.
        """
        ...
