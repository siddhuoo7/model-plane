"""Security and governance layer for the routing pipeline.

Implements Stage 0.5 (hard eligibility filter) from the cache-aware routing plan.

Design (§13, §18–21 of research doc):
  Security is a HARD filter, not a soft score.  A restricted request must never
  reach an external provider regardless of cache hit rate or cost savings.
  SecurityGuard runs BEFORE any scoring and removes ineligible candidates so
  they cannot be selected by subsequent stages.

  Fail-safe principle (§47): when in doubt, fail closed.  If the detector
  raises an exception the default is CONFIDENTIAL (not PUBLIC), and all
  EXTERNAL_PUBLIC candidates are removed rather than passed through.

PII detection is keyword-heuristic / regex only (no external dependency).
Production deployments can replace _classify_sensitivity_impl() with
Microsoft Presidio — the public interface classify_sensitivity() is stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from model_plane.logging_setup import get_logger

if TYPE_CHECKING:
    from model_plane.registry.catalog import DeploymentConfig

log = get_logger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────────────

class DataSensitivity(str, Enum):
    PUBLIC       = "public"
    INTERNAL     = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED   = "restricted"


class ProviderTrust(str, Enum):
    EXTERNAL_PUBLIC   = "external_public"    # OpenAI, Anthropic, Cohere, Mistral …
    APPROVED_EXTERNAL = "approved_external"  # Azure OpenAI, Bedrock, Vertex AI …
    PRIVATE_CLOUD     = "private_cloud"      # watsonx on IBM Cloud, private GCP …
    ON_PREM           = "on_prem"            # self-hosted vLLM / local models


# ── Default provider → trust mapping ─────────────────────────────────────────
# Overridable via routing.yaml → provider_trust_overrides
# and per-deployment via models.yaml → provider_trust.

_DEFAULT_PROVIDER_TRUST: dict[str, ProviderTrust] = {
    "openai":     ProviderTrust.EXTERNAL_PUBLIC,
    "anthropic":  ProviderTrust.EXTERNAL_PUBLIC,
    "cohere":     ProviderTrust.EXTERNAL_PUBLIC,
    "mistral":    ProviderTrust.EXTERNAL_PUBLIC,
    "azure":      ProviderTrust.APPROVED_EXTERNAL,
    "bedrock":    ProviderTrust.APPROVED_EXTERNAL,
    "vertex_ai":  ProviderTrust.APPROVED_EXTERNAL,
    "watsonx":    ProviderTrust.PRIVATE_CLOUD,
    "vllm":       ProviderTrust.ON_PREM,
    "local":      ProviderTrust.ON_PREM,
}

# ── Policy matrix ─────────────────────────────────────────────────────────────
# (DataSensitivity, ProviderTrust) → allowed?
# Matches §21 of the research document.

_POLICY_MATRIX: dict[tuple[DataSensitivity, ProviderTrust], bool] = {
    (DataSensitivity.PUBLIC,       ProviderTrust.EXTERNAL_PUBLIC):   True,
    (DataSensitivity.PUBLIC,       ProviderTrust.APPROVED_EXTERNAL): True,
    (DataSensitivity.PUBLIC,       ProviderTrust.PRIVATE_CLOUD):     True,
    (DataSensitivity.PUBLIC,       ProviderTrust.ON_PREM):           True,
    (DataSensitivity.INTERNAL,     ProviderTrust.EXTERNAL_PUBLIC):   False,
    (DataSensitivity.INTERNAL,     ProviderTrust.APPROVED_EXTERNAL): True,
    (DataSensitivity.INTERNAL,     ProviderTrust.PRIVATE_CLOUD):     True,
    (DataSensitivity.INTERNAL,     ProviderTrust.ON_PREM):           True,
    (DataSensitivity.CONFIDENTIAL, ProviderTrust.EXTERNAL_PUBLIC):   False,
    (DataSensitivity.CONFIDENTIAL, ProviderTrust.APPROVED_EXTERNAL): False,
    (DataSensitivity.CONFIDENTIAL, ProviderTrust.PRIVATE_CLOUD):     True,
    (DataSensitivity.CONFIDENTIAL, ProviderTrust.ON_PREM):           True,
    (DataSensitivity.RESTRICTED,   ProviderTrust.EXTERNAL_PUBLIC):   False,
    (DataSensitivity.RESTRICTED,   ProviderTrust.APPROVED_EXTERNAL): False,
    (DataSensitivity.RESTRICTED,   ProviderTrust.PRIVATE_CLOUD):     False,
    (DataSensitivity.RESTRICTED,   ProviderTrust.ON_PREM):           True,
}


# ── PII / secret detection heuristics ─────────────────────────────────────────
# Lightweight regex patterns.  Replace the body of _classify_sensitivity_impl()
# with a Presidio Analyzer call in production (Phase 5 upgrade).

# PII that is serious enough to escalate to RESTRICTED on its own.
_PII_PATTERNS_RESTRICTED: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL",       re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("PHONE",       re.compile(r"\b(?:\+?\d[\s\-.]?){7,14}\d\b")),
    ("SSN",         re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ \-]?){13,16}\b")),
]

# PII that is INTERNAL-level on its own — it only escalates to RESTRICTED
# when combined with another RESTRICTED-level PII type or keyword.
# An IP address commonly appears in log lines, URLs, and code snippets; treating
# it as RESTRICTED alone causes too many false positives for developer tools.
_PII_PATTERNS_INTERNAL: list[tuple[str, re.Pattern[str]]] = [
    ("IP_ADDRESS",  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

# Combined list used by callers that just need the full set of PII types detected.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = (
    _PII_PATTERNS_RESTRICTED + _PII_PATTERNS_INTERNAL
)

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS_KEY",     re.compile(r"AKIA[0-9A-Z]{16}")),
    ("API_KEY",     re.compile(
        r"(?i)(?:api[_\-]?key|secret[_\-]?key|bearer)[\s=:]+['\"]?[A-Za-z0-9\-_]{16,}"
    )),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
]

_RESTRICTED_TERMS: re.Pattern[str] = re.compile(
    r"\b(?:customer[- ]data|patient[- ]data|medical[- ]record|phi|hipaa|gdpr|"
    r"account[- ]number|ssn|social[- ]security|passport[- ]number|"
    r"credit[- ]card|classified|top[- ]secret)\b",
    re.IGNORECASE,
)

_CONFIDENTIAL_TERMS: re.Pattern[str] = re.compile(
    r"\b(?:confidential|internal[- ]only|proprietary|not[- ]for[- ]distribution|"
    r"trade[- ]secret)\b",
    re.IGNORECASE,
)


def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate user/system text from a messages array."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


def _classify_sensitivity_impl(
    text: str,
    disabled_pii_rules: set[str] | None = None,
) -> tuple[DataSensitivity, list[str], list[str]]:
    """Core detection logic — replaceable with Presidio in Phase 5 upgrade.

    Sensitivity ladder:
      RESTRICTED   — secrets, high-severity PII (email/phone/SSN/CC), or
                     restricted keywords; OR internal-PII combined with another
                     restricted signal.
      CONFIDENTIAL — confidential-term keywords alone.
      INTERNAL     — internal-level PII only (e.g. bare IP address in a log
                     line / URL), no other sensitive signals.
      PUBLIC       — nothing detected.

    disabled_pii_rules: set of PII pattern IDs to skip (e.g. {"IP_ADDRESS"}).
      Disabled patterns are still included in pii_types for the audit trail
      but are excluded from the sensitivity calculation.
    """
    disabled = disabled_pii_rules or set()

    # Collect RESTRICTED-level PII hits (excluding disabled rules)
    restricted_pii: list[str] = []
    for name, pattern in _PII_PATTERNS_RESTRICTED:
        if name not in disabled and pattern.search(text):
            restricted_pii.append(name)

    # Collect INTERNAL-level PII hits (excluding disabled rules)
    internal_pii: list[str] = []
    for name, pattern in _PII_PATTERNS_INTERNAL:
        if name not in disabled and pattern.search(text):
            internal_pii.append(name)

    # Audit: also collect disabled-but-matched PII for the log trail
    disabled_matched: list[str] = []
    for name, pattern in _PII_PATTERNS:
        if name in disabled and pattern.search(text):
            disabled_matched.append(name)

    # pii_types in the audit includes disabled matches (informational only)
    pii_types = restricted_pii + internal_pii + disabled_matched

    secret_types: list[str] = []
    for name, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            secret_types.append(name)

    # RESTRICTED: any secret, any active restricted-level PII, any restricted keyword.
    has_restricted_keyword = bool(_RESTRICTED_TERMS.search(text))
    if secret_types or restricted_pii or has_restricted_keyword:
        return DataSensitivity.RESTRICTED, pii_types, secret_types

    # CONFIDENTIAL: confidential keyword, no higher signal already matched.
    if _CONFIDENTIAL_TERMS.search(text):
        return DataSensitivity.CONFIDENTIAL, pii_types, secret_types

    # INTERNAL: active internal-level PII present (e.g. IP address in a log/URL).
    if internal_pii:
        return DataSensitivity.INTERNAL, pii_types, secret_types

    return DataSensitivity.PUBLIC, pii_types, secret_types


def classify_sensitivity(
    messages: list[dict[str, Any]],
) -> tuple[DataSensitivity, list[str], list[str]]:
    """Classify the data sensitivity of a request.

    Public interface — stable across Presidio upgrade.

    Returns:
        (DataSensitivity, pii_types, secret_types)

    Fail-safe: if detection raises, returns CONFIDENTIAL to prevent accidental
    external routing of potentially sensitive content.
    """
    try:
        text = _extract_text(messages)
        return _classify_sensitivity_impl(text)
    except Exception as exc:
        log.warning("sensitivity_classification_failed", error=str(exc))
        return DataSensitivity.CONFIDENTIAL, [], []


# ── SecurityContext ───────────────────────────────────────────────────────────

@dataclass
class SecurityContext:
    """Per-request security classification and policy decisions.

    Attached to RoutingContext.security_ctx so the full audit trail is
    available in logs and observability hooks.
    """

    data_sensitivity: DataSensitivity = DataSensitivity.PUBLIC
    pii_types: list[str] = field(default_factory=list)
    secret_types: list[str] = field(default_factory=list)
    # Deployments removed by the hard filter: {deployment_name: reason_string}
    blocked_deployments: dict[str, str] = field(default_factory=dict)
    # Free-text audit summary logged per request
    audit_summary: str = ""


# ── SecurityGuard ─────────────────────────────────────────────────────────────

class SecurityGuard:
    """Stage 0.5: hard eligibility filter.

    Takes a list of candidate deployments and returns the subset that is
    allowed to receive this request under the current data-sensitivity policy.

    Removed candidates are recorded in SecurityContext.blocked_deployments
    to provide a full audit trail.

    Args:
        provider_trust_map: Optional overrides loaded from
            routing.yaml → provider_trust_overrides.  Maps provider string
            (e.g. "my_custom_provider") to a ProviderTrust string value.
    """

    def __init__(
        self,
        provider_trust_map: dict[str, str] | None = None,
        disabled_pii_rules: set[str] | None = None,
    ) -> None:
        self._trust_map: dict[str, ProviderTrust] = dict(_DEFAULT_PROVIDER_TRUST)
        if provider_trust_map:
            for prov, trust_str in provider_trust_map.items():
                try:
                    self._trust_map[prov] = ProviderTrust(trust_str)
                except ValueError:
                    log.warning(
                        "unknown_provider_trust_value",
                        provider=prov,
                        value=trust_str,
                    )
        # PII rule IDs disabled via the admin UI / API (e.g. {"IP_ADDRESS"}).
        # Patterns in this set are skipped during classify_sensitivity().
        self._disabled_pii_rules: set[str] = disabled_pii_rules or set()

    def provider_trust(self, provider: str) -> ProviderTrust:
        """Return the trust level for *provider*, falling back to EXTERNAL_PUBLIC."""
        return self._trust_map.get(provider, ProviderTrust.EXTERNAL_PUBLIC)

    def classify(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[DataSensitivity, list[str], list[str]]:
        """Classify sensitivity respecting this guard's disabled_pii_rules.

        Wraps the module-level classify_sensitivity() but filters out any PII
        pattern whose ID is in self._disabled_pii_rules before scoring.
        If all patterns that fired are disabled, the sensitivity is downgraded
        to the level it would have reached without those patterns.
        """
        try:
            text = _extract_text(messages)
            return _classify_sensitivity_impl(text, self._disabled_pii_rules)
        except Exception as exc:
            log.warning("sensitivity_classification_failed", error=str(exc))
            return DataSensitivity.CONFIDENTIAL, [], []

    def filter(
        self,
        candidates: list["DeploymentConfig"],
        security_ctx: SecurityContext,
    ) -> list["DeploymentConfig"]:
        """Return the subset of *candidates* allowed for this request.

        Modifies *security_ctx* in-place to record blocked deployments and
        write the audit_summary.

        Fail-safe: if an unexpected exception occurs, all EXTERNAL_PUBLIC
        candidates are removed rather than passed through.
        """
        try:
            return self._filter_impl(candidates, security_ctx)
        except Exception as exc:
            log.error("security_guard_filter_failed", error=str(exc))
            # Fail closed: remove any external-public candidates
            safe = [
                d for d in candidates
                if self.provider_trust(d.provider) != ProviderTrust.EXTERNAL_PUBLIC
            ]
            for dep in candidates:
                if dep not in safe:
                    security_ctx.blocked_deployments[dep.name] = (
                        "security_guard_error: fail_closed"
                    )
            security_ctx.audit_summary = f"fail_closed error={exc}"
            return safe

    def _filter_impl(
        self,
        candidates: list["DeploymentConfig"],
        security_ctx: SecurityContext,
    ) -> list["DeploymentConfig"]:
        allowed: list["DeploymentConfig"] = []
        sensitivity = security_ctx.data_sensitivity

        for dep in candidates:
            # Per-deployment trust override takes precedence over default map
            dep_trust_str: str = getattr(dep, "provider_trust", "") or ""
            if dep_trust_str:
                try:
                    trust = ProviderTrust(dep_trust_str)
                except ValueError:
                    trust = self.provider_trust(dep.provider)
            else:
                trust = self.provider_trust(dep.provider)

            permitted = _POLICY_MATRIX.get((sensitivity, trust), False)
            if permitted:
                allowed.append(dep)
            else:
                reason = (
                    f"policy_denied: data={sensitivity.value} "
                    f"provider_trust={trust.value}"
                )
                security_ctx.blocked_deployments[dep.name] = reason
                log.debug(
                    "security_guard_blocked",
                    deployment=dep.name,
                    provider=dep.provider,
                    trust=trust.value,
                    sensitivity=sensitivity.value,
                )

        security_ctx.audit_summary = (
            f"sensitivity={sensitivity.value} "
            f"pii={security_ctx.pii_types or 'none'} "
            f"secrets={security_ctx.secret_types or 'none'} "
            f"blocked={list(security_ctx.blocked_deployments.keys()) or 'none'} "
            f"allowed={[d.name for d in allowed]}"
        )
        return allowed


_GLOBAL_OWNER = "__global__"


def get_security_guard(
    provider_trust_map: dict[str, str] | None = None,
    owner_id: str | None = None,
) -> SecurityGuard:
    """Return a SecurityGuard reflecting the current provider trust and PII rule overrides.

    Loads provider trust overrides and PII rule disables from SQLite (user/tenant + global fallback).
    """
    disabled_pii_rules: set[str] = set()
    combined_trust_map: dict[str, str] = dict(provider_trust_map or {})
    try:
        import model_plane.db as _db
        # Always load the global config; overlay with tenant/user-specific if present.
        for key in (_GLOBAL_OWNER, owner_id):
            if not key:
                continue
            pii_cfg = _db.load_routing_override("pii_rules", key)
            disabled_list = pii_cfg.get("disabled", [])
            if isinstance(disabled_list, list):
                disabled_pii_rules.update(str(r) for r in disabled_list)

            # Also check user-specific provider trust overrides stored in routing_rules or provider_trust
            routing_override = _db.load_routing_override("routing_rules", key)
            if "provider_trust_overrides" in routing_override:
                combined_trust_map.update(routing_override["provider_trust_overrides"])
            prov_trust_override = _db.load_routing_override("provider_trust", key)
            if "provider_trust_overrides" in prov_trust_override:
                combined_trust_map.update(prov_trust_override["provider_trust_overrides"])
    except Exception:
        pass  # fail-open: missing DB entry → no rules disabled
    return SecurityGuard(combined_trust_map or None, disabled_pii_rules or None)
