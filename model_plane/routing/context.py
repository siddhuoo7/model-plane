"""Core routing context and result types shared by all routing layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_plane.classifier.classifier import ClassificationResult
from model_plane.classifier.features import RequestFeatures
from model_plane.registry.catalog import DeploymentConfig
from model_plane.scorer.scorer import ScorerResult


@dataclass
class RoutingContext:
    """Everything the routing pipeline knows about a single request."""

    request_id: str
    raw_request: dict[str, Any]
    tenant_id: str | None = None
    session_id: str | None = None
    agent_type: str | None = None

    # filled by pre-routing hooks
    features: RequestFeatures | None = None
    classification: ClassificationResult | None = None
    scorer_result: ScorerResult | None = None

    # routing state
    candidate_deployments: list[DeploymentConfig] = field(default_factory=list)
    selected_deployment: DeploymentConfig | None = None
    routing_source: str = "unset"  # custom_plugin | litellm_auto_router | default
    routing_confidence: float = 0.0

    # compression
    compression_profile: str = "passthrough"
    compressed_request: dict[str, Any] | None = None
    tokens_before_compression: int = 0
    tokens_after_compression: int = 0

    # cache
    cache_warm: bool = False
    prefix_hash: str | None = None

    # response metrics (filled post-call)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    validation_result: str | None = None
    fallback_used: bool = False
    escalated: bool = False
    final_success: bool | None = None

    # ML shadow
    ml_recommended_deployment: str | None = None
    ml_recommendation_confidence: float = 0.0


@dataclass
class RoutingDecision:
    deployment: DeploymentConfig
    source: str  # custom_plugin | litellm_auto_router | default | fallback
    confidence: float
    reasoning: str = ""
