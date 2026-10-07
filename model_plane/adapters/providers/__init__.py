"""Provider adapter registry.

Registry maps provider name strings to adapter instances.
``get_adapter(provider)`` is the single call-site used by executor.py.
Unknown providers fall back to ``DefaultAdapter`` so that future providers
added to models.yaml do not break execution before their adapter is written.
"""

from __future__ import annotations

from model_plane.adapters.providers.anthropic import AnthropicAdapter
from model_plane.adapters.providers.azure_openai import AzureOpenAIAdapter
from model_plane.adapters.providers.base import ProviderAdapter
from model_plane.adapters.providers.bedrock import BedrockAdapter
from model_plane.adapters.providers.cohere import CohereAdapter
from model_plane.adapters.providers.default import DefaultAdapter
from model_plane.adapters.providers.mistral import MistralAdapter
from model_plane.adapters.providers.openai import OpenAIAdapter
from model_plane.adapters.providers.vertex_ai import VertexAIAdapter
from model_plane.adapters.providers.vllm import VLLMAdapter
from model_plane.adapters.providers.watsonx import WatsonxAdapter

_default = DefaultAdapter()

_REGISTRY: dict[str, ProviderAdapter] = {
    "openai":     OpenAIAdapter(),
    "anthropic":  AnthropicAdapter(),
    "watsonx":    WatsonxAdapter(),
    "local":      VLLMAdapter(),
    "bedrock":    BedrockAdapter(),
    "azure":      AzureOpenAIAdapter(),
    "vertex_ai":  VertexAIAdapter(),
    "cohere":     CohereAdapter(),
    "mistral":    MistralAdapter(),
}


def get_adapter(provider: str) -> ProviderAdapter:
    """Return the adapter for *provider*, falling back to the no-op default."""
    return _REGISTRY.get(provider, _default)
