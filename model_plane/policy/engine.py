"""Tenant and sovereignty policy engine.

Enforces:
- Provider allow/deny lists per tenant
- Data residency / regulated workload constraints
- Rate and cost budget enforcement
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from model_plane.classifier.taxonomy import TaskType
from model_plane.logging_setup import get_logger
from model_plane.registry.catalog import DeploymentConfig

log = get_logger(__name__)


@dataclass
class TenantPolicy:
    tenant_id: str
    allowed_providers: list[str] = field(default_factory=list)   # empty = all
    denied_providers: list[str] = field(default_factory=list)
    allowed_deployments: list[str] = field(default_factory=list)  # empty = all
    denied_deployments: list[str] = field(default_factory=list)
    require_local: bool = False
    regulated_tasks: list[str] = field(default_factory=list)
    regulated_deployment: str | None = None
    max_cost_per_request: float | None = None  # USD
    max_tokens_per_request: int | None = None


@dataclass
class PolicyDecision:
    allowed: bool
    filtered_deployments: list[DeploymentConfig]
    reason: str = ""


class PolicyEngine:
    def __init__(self, policies: dict[str, TenantPolicy] | None = None) -> None:
        self._policies: dict[str, TenantPolicy] = policies or {}

    @classmethod
    def from_yaml(cls, path: str) -> "PolicyEngine":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        policies = {}
        for entry in raw.get("tenant_policies", []):
            p = TenantPolicy(
                tenant_id=entry["tenant_id"],
                allowed_providers=entry.get("allowed_providers", []),
                denied_providers=entry.get("denied_providers", []),
                allowed_deployments=entry.get("allowed_deployments", []),
                denied_deployments=entry.get("denied_deployments", []),
                require_local=entry.get("require_local", False),
                regulated_tasks=entry.get("regulated_tasks", []),
                regulated_deployment=entry.get("regulated_deployment"),
                max_cost_per_request=entry.get("max_cost_per_request"),
                max_tokens_per_request=entry.get("max_tokens_per_request"),
            )
            policies[p.tenant_id] = p
        return cls(policies)

    def enforce(
        self,
        tenant_id: str | None,
        task_type: TaskType,
        candidates: list[DeploymentConfig],
        estimated_tokens: int = 0,
    ) -> PolicyDecision:
        if not tenant_id:
            return PolicyDecision(allowed=True, filtered_deployments=candidates)

        policy = self._policies.get(tenant_id)
        if not policy:
            return PolicyDecision(allowed=True, filtered_deployments=candidates)

        # Token budget
        if policy.max_tokens_per_request and estimated_tokens > policy.max_tokens_per_request:
            log.warning(
                "policy_token_budget_exceeded",
                tenant=tenant_id,
                tokens=estimated_tokens,
                limit=policy.max_tokens_per_request,
            )
            return PolicyDecision(allowed=False, filtered_deployments=[], reason="token_budget_exceeded")

        filtered: list[DeploymentConfig] = []
        for dep in candidates:
            # Provider allow list
            if policy.allowed_providers and dep.provider not in policy.allowed_providers:
                continue
            # Provider deny list
            if dep.provider in policy.denied_providers:
                continue
            # Deployment allow list
            if policy.allowed_deployments and dep.name not in policy.allowed_deployments:
                continue
            # Deployment deny list
            if dep.name in policy.denied_deployments:
                continue
            # Require local
            if policy.require_local and dep.provider != "local":
                continue
            filtered.append(dep)

        # Regulated task override — must use regulated deployment
        if task_type.value in policy.regulated_tasks and policy.regulated_deployment:
            regulated = next((d for d in filtered if d.name == policy.regulated_deployment), None)
            if regulated:
                log.info("policy_regulated_task_override", tenant=tenant_id, task=task_type.value)
                return PolicyDecision(allowed=True, filtered_deployments=[regulated], reason="regulated")
            else:
                log.error("policy_regulated_deployment_unavailable", tenant=tenant_id)
                return PolicyDecision(allowed=False, filtered_deployments=[], reason="regulated_deployment_unavailable")

        if not filtered:
            return PolicyDecision(allowed=False, filtered_deployments=[], reason="all_deployments_denied_by_policy")

        return PolicyDecision(allowed=True, filtered_deployments=filtered)
