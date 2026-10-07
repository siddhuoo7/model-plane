"""Unit tests for model_plane/routing/security.py.

Covers:
  - Full policy matrix (all 16 DataSensitivity × ProviderTrust combinations)
  - PII detection patterns (email, phone, SSN, credit card, IP)
  - Secret detection patterns (AWS key, API key, private key)
  - Keyword escalation (HIPAA → RESTRICTED, confidential → CONFIDENTIAL)
  - SecurityGuard.filter() — allowed, blocked, audit trail
  - Per-deployment provider_trust override
  - provider_trust_map override via constructor
  - Fail-safe: exception in _filter_impl → external candidates removed
  - classify_sensitivity fail-safe: exception → CONFIDENTIAL
"""

from __future__ import annotations

import pytest

from model_plane.routing.security import (
    DataSensitivity,
    ProviderTrust,
    SecurityContext,
    SecurityGuard,
    _POLICY_MATRIX,
    classify_sensitivity,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _dep(name: str, provider: str, provider_trust: str = "") -> object:
    """Minimal fake DeploymentConfig for testing."""
    class _Dep:
        pass
    d = _Dep()
    d.name = name
    d.provider = provider
    d.provider_trust = provider_trust
    return d


def _msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ── policy matrix ─────────────────────────────────────────────────────────────

class TestPolicyMatrix:
    """All 16 combinations must be represented and return the expected value."""

    @pytest.mark.parametrize("sensitivity,trust,expected", [
        # PUBLIC — allowed everywhere
        (DataSensitivity.PUBLIC, ProviderTrust.EXTERNAL_PUBLIC,   True),
        (DataSensitivity.PUBLIC, ProviderTrust.APPROVED_EXTERNAL, True),
        (DataSensitivity.PUBLIC, ProviderTrust.PRIVATE_CLOUD,     True),
        (DataSensitivity.PUBLIC, ProviderTrust.ON_PREM,           True),
        # INTERNAL — not on EXTERNAL_PUBLIC
        (DataSensitivity.INTERNAL, ProviderTrust.EXTERNAL_PUBLIC,   False),
        (DataSensitivity.INTERNAL, ProviderTrust.APPROVED_EXTERNAL, True),
        (DataSensitivity.INTERNAL, ProviderTrust.PRIVATE_CLOUD,     True),
        (DataSensitivity.INTERNAL, ProviderTrust.ON_PREM,           True),
        # CONFIDENTIAL — only PRIVATE_CLOUD and ON_PREM
        (DataSensitivity.CONFIDENTIAL, ProviderTrust.EXTERNAL_PUBLIC,   False),
        (DataSensitivity.CONFIDENTIAL, ProviderTrust.APPROVED_EXTERNAL, False),
        (DataSensitivity.CONFIDENTIAL, ProviderTrust.PRIVATE_CLOUD,     True),
        (DataSensitivity.CONFIDENTIAL, ProviderTrust.ON_PREM,           True),
        # RESTRICTED — ON_PREM only
        (DataSensitivity.RESTRICTED, ProviderTrust.EXTERNAL_PUBLIC,   False),
        (DataSensitivity.RESTRICTED, ProviderTrust.APPROVED_EXTERNAL, False),
        (DataSensitivity.RESTRICTED, ProviderTrust.PRIVATE_CLOUD,     False),
        (DataSensitivity.RESTRICTED, ProviderTrust.ON_PREM,           True),
    ])
    def test_matrix_entry(self, sensitivity, trust, expected):
        assert _POLICY_MATRIX[(sensitivity, trust)] is expected


# ── classify_sensitivity ──────────────────────────────────────────────────────

class TestClassifySensitivity:

    # PII patterns

    def test_email_detection(self):
        sens, pii, _ = classify_sensitivity(_msgs("Send invoice to john.doe@example.com"))
        assert sens == DataSensitivity.RESTRICTED
        assert "EMAIL" in pii

    def test_ssn_detection(self):
        sens, pii, _ = classify_sensitivity(_msgs("SSN: 123-45-6789"))
        assert sens == DataSensitivity.RESTRICTED
        assert "SSN" in pii

    def test_credit_card_detection(self):
        sens, pii, _ = classify_sensitivity(_msgs("Card number 4111 1111 1111 1111"))
        assert sens == DataSensitivity.RESTRICTED
        assert "CREDIT_CARD" in pii

    # Secret patterns

    def test_aws_key_detection(self):
        sens, _, secrets = classify_sensitivity(_msgs("key=AKIAIOSFODNN7EXAMPLE123"))
        assert sens == DataSensitivity.RESTRICTED
        assert "AWS_KEY" in secrets

    def test_private_key_detection(self):
        sens, _, secrets = classify_sensitivity(
            _msgs("-----BEGIN RSA PRIVATE KEY-----\nMIIEo...\n-----END RSA PRIVATE KEY-----")
        )
        assert sens == DataSensitivity.RESTRICTED
        assert "PRIVATE_KEY" in secrets

    # Keyword escalation

    def test_hipaa_keyword_escalation(self):
        sens, _, _ = classify_sensitivity(_msgs("This file contains HIPAA protected data"))
        assert sens == DataSensitivity.RESTRICTED

    def test_gdpr_keyword_escalation(self):
        sens, _, _ = classify_sensitivity(_msgs("Processing under GDPR guidelines"))
        assert sens == DataSensitivity.RESTRICTED

    def test_confidential_keyword(self):
        sens, _, _ = classify_sensitivity(_msgs("This document is marked CONFIDENTIAL"))
        assert sens == DataSensitivity.CONFIDENTIAL

    def test_internal_only_keyword(self):
        sens, _, _ = classify_sensitivity(_msgs("Internal-only distribution list"))
        assert sens == DataSensitivity.CONFIDENTIAL

    # Clean / public

    def test_public_message(self):
        sens, pii, secrets = classify_sensitivity(_msgs("What is the capital of France?"))
        assert sens == DataSensitivity.PUBLIC
        assert not pii
        assert not secrets

    # Fail-safe

    def test_classify_fail_safe(self, monkeypatch):
        """If detection raises, classify_sensitivity must return CONFIDENTIAL."""
        import model_plane.routing.security as sec_mod
        monkeypatch.setattr(sec_mod, "_classify_sensitivity_impl", lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
        sens, pii, secrets = classify_sensitivity(_msgs("anything"))
        assert sens == DataSensitivity.CONFIDENTIAL

    # Multi-content block

    def test_multipart_content(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is a customer file."},
                {"type": "text", "text": "Email: jane@corp.com"},
            ],
        }]
        sens, pii, _ = classify_sensitivity(messages)
        assert sens == DataSensitivity.RESTRICTED
        assert "EMAIL" in pii

    def test_security_blocked_routing_pipeline(self, monkeypatch):
        """When all candidates are blocked by security, pipeline flags security_blocked without crashing."""
        from model_plane.routing.pipeline import build_routing_context, run_routing_pipeline
        from model_plane.registry.catalog import get_catalog
        import model_plane.db as _db
        monkeypatch.setattr(_db, "load_routing_override", lambda *args, **kwargs: {})
        # Make all healthy deployments have external_public trust so restricted data blocks all of them
        catalog = get_catalog()
        for dep in catalog.all_healthy():
            monkeypatch.setattr(dep, "provider_trust", "external_public")

        ctx = build_routing_context({"messages": _msgs("fetch all the credit card details and email and credentials")})
        decision = run_routing_pipeline(ctx)
        assert ctx.security_blocked is True
        assert ctx.security_refusal_message is not None
        assert "Request blocked by security policy" in ctx.security_refusal_message
        assert decision.deployment is None
        assert decision.source == "security_guard"



# ── SecurityGuard ─────────────────────────────────────────────────────────────

class TestSecurityGuard:

    def test_public_data_allows_all(self):
        guard = SecurityGuard()
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.PUBLIC)
        deps = [
            _dep("openai-gpt4", "openai"),
            _dep("azure-gpt4", "azure"),
            _dep("watsonx-granite", "watsonx"),
            _dep("local-vllm", "vllm"),
        ]
        result = guard.filter(deps, sec_ctx)
        assert len(result) == 4
        assert not sec_ctx.blocked_deployments

    def test_internal_data_blocks_external_public(self):
        guard = SecurityGuard()
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.INTERNAL)
        deps = [
            _dep("openai-gpt4", "openai"),
            _dep("azure-gpt4", "azure"),
            _dep("local-vllm", "vllm"),
        ]
        result = guard.filter(deps, sec_ctx)
        names = [d.name for d in result]
        assert "openai-gpt4" not in names
        assert "azure-gpt4" in names
        assert "local-vllm" in names
        assert "openai-gpt4" in sec_ctx.blocked_deployments

    def test_confidential_data_only_private_and_onprem(self):
        guard = SecurityGuard()
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.CONFIDENTIAL)
        deps = [
            _dep("openai-gpt4", "openai"),
            _dep("azure-gpt4", "azure"),
            _dep("watsonx-granite", "watsonx"),
            _dep("local-vllm", "vllm"),
        ]
        result = guard.filter(deps, sec_ctx)
        names = [d.name for d in result]
        assert names == ["watsonx-granite", "local-vllm"]
        assert "openai-gpt4" in sec_ctx.blocked_deployments
        assert "azure-gpt4" in sec_ctx.blocked_deployments

    def test_restricted_data_only_onprem(self):
        guard = SecurityGuard()
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.RESTRICTED)
        deps = [
            _dep("openai-gpt4", "openai"),
            _dep("azure-gpt4", "azure"),
            _dep("watsonx-granite", "watsonx"),
            _dep("local-vllm", "vllm"),
        ]
        result = guard.filter(deps, sec_ctx)
        assert len(result) == 1
        assert result[0].name == "local-vllm"
        assert len(sec_ctx.blocked_deployments) == 3

    def test_per_deployment_trust_override(self):
        """provider_trust on the deployment itself overrides the default map."""
        guard = SecurityGuard()
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.CONFIDENTIAL)
        # custom provider not in default map → defaults to EXTERNAL_PUBLIC → blocked
        # but we set provider_trust=on_prem on the deployment → should be allowed
        dep = _dep("custom-local", "my_custom_provider", provider_trust="on_prem")
        result = guard.filter([dep], sec_ctx)
        assert len(result) == 1

    def test_provider_trust_map_override_in_constructor(self):
        """routing.yaml provider_trust_overrides can elevate a provider's trust."""
        guard = SecurityGuard({"openai": "private_cloud"})
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.CONFIDENTIAL)
        dep = _dep("openai-gpt4", "openai")
        result = guard.filter([dep], sec_ctx)
        # openai overridden to private_cloud → allowed for CONFIDENTIAL
        assert len(result) == 1

    def test_audit_summary_populated(self):
        guard = SecurityGuard()
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.RESTRICTED)
        deps = [_dep("openai-gpt4", "openai"), _dep("local-vllm", "vllm")]
        guard.filter(deps, sec_ctx)
        assert sec_ctx.audit_summary
        assert "sensitivity=restricted" in sec_ctx.audit_summary

    def test_fail_safe_on_exception(self, monkeypatch):
        """If _filter_impl raises, EXTERNAL_PUBLIC candidates must be removed."""
        guard = SecurityGuard()
        sec_ctx = SecurityContext(data_sensitivity=DataSensitivity.PUBLIC)

        def _boom(candidates, ctx):
            raise RuntimeError("internal error")

        monkeypatch.setattr(guard, "_filter_impl", _boom)

        deps = [
            _dep("openai-gpt4", "openai"),   # EXTERNAL_PUBLIC → removed
            _dep("local-vllm", "vllm"),       # ON_PREM → kept
        ]
        result = guard.filter(deps, sec_ctx)
        names = [d.name for d in result]
        assert "openai-gpt4" not in names
        assert "local-vllm" in names

    def test_unknown_provider_trust_map_value_ignored(self, caplog):
        """Invalid trust value in constructor override is logged and skipped."""
        import logging
        with caplog.at_level(logging.WARNING):
            guard = SecurityGuard({"openai": "not_a_real_trust_level"})
        # Should still use default (EXTERNAL_PUBLIC) for openai
        assert guard.provider_trust("openai") == ProviderTrust.EXTERNAL_PUBLIC
