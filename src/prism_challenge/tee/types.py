"""Stable TEE classification and reason-code vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TeeClassification(StrEnum):
    """TEE validation outcomes (contract-wide vocabulary)."""

    LOCAL_FIXTURE_PASS = "LOCAL-FIXTURE PASS"
    REAL_PROVIDER_PASS = "REAL-PROVIDER PASS"
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"


class TeeProviderKind(StrEnum):
    LOCAL_FIXTURE = "local_fixture"
    LIUM = "lium"
    TARGON = "targon"
    UNKNOWN = "unknown"


class TeeReasonCode(StrEnum):
    """Machine-readable rejection/acceptance reasons (non-secret)."""

    ACCEPTED_LOCAL_FIXTURE = "accepted_local_fixture"
    VERIFIER_DISABLED = "verifier_disabled"
    VERIFIER_MISCONFIGURED = "verifier_misconfigured"
    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_MALFORMED = "evidence_malformed"
    EVIDENCE_WRONG_TYPE = "evidence_wrong_type"
    EVIDENCE_UNKNOWN_VERSION = "evidence_unknown_version"
    EVIDENCE_UNKNOWN_PROVIDER = "evidence_unknown_provider"
    EVIDENCE_UNKNOWN_TYPE = "evidence_unknown_type"
    EVIDENCE_UNKNOWN_FIELD = "evidence_unknown_field"
    EVIDENCE_OVERSIZE = "evidence_oversize"
    ENCODING_INVALID = "encoding_invalid"
    COMPONENT_MISSING = "component_missing"
    TDX_SIGNATURE_INVALID = "tdx_signature_invalid"
    TDX_CHAIN_UNTRUSTED = "tdx_chain_untrusted"
    TDX_CERT_REJECTED = "tdx_cert_rejected"
    GPU_SIGNATURE_INVALID = "gpu_signature_invalid"
    GPU_ALG_CONFUSION = "gpu_alg_confusion"
    GPU_UNTRUSTED_KEY = "gpu_untrusted_key"
    TRUST_LOCATOR_FORBIDDEN = "trust_locator_forbidden"
    PROVIDER_MISMATCH = "provider_mismatch"
    ISSUER_MISMATCH = "issuer_mismatch"
    AUDIENCE_MISMATCH = "audience_mismatch"
    FRESHNESS_INVALID = "freshness_invalid"
    NONCE_MISSING = "nonce_missing"
    NONCE_MISMATCH = "nonce_mismatch"
    NONCE_REPLAY = "nonce_replay"
    WORKLOAD_MISMATCH = "workload_mismatch"
    IMAGE_DIGEST_MISMATCH = "image_digest_mismatch"
    MEASUREMENT_MISMATCH = "measurement_mismatch"
    MEASUREMENT_POLICY_EMPTY = "measurement_policy_empty"
    TCB_POLICY_REJECTED = "tcb_policy_rejected"
    GPU_IDENTITY_MISMATCH = "gpu_identity_mismatch"
    CROSS_BINDING_MISMATCH = "cross_binding_mismatch"
    PROOF_BINDING_MISMATCH = "proof_binding_mismatch"
    PROVIDER_BLOCKED = "provider_blocked"
    PROVIDER_FUTURE_BLOCKED = "provider_future_blocked"
    ADAPTER_NOT_READY = "adapter_not_ready"
    ADAPTER_ENDPOINT_FAILURE = "adapter_endpoint_failure"
    METADATA_NOT_ATTESTATION = "metadata_not_attestation"
    CLAIMED_TIER_IGNORED = "claimed_tier_ignored"


@dataclass(frozen=True)
class TeeDecision:
    """Verifier decision surface (never secret-bearing)."""

    accepted: bool
    classification: TeeClassification
    reason: TeeReasonCode
    provider: TeeProviderKind
    effective_tier: int
    evidence_digest: str | None = None
    trust_root_fingerprint: str | None = None
    gpu_key_fingerprint: str | None = None
    image_digest: str | None = None
    nonce_digest: str | None = None
    work_unit_id: str | None = None
    validated_claims: tuple[str, ...] = ()
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_audit_record(self) -> dict[str, Any]:
        """Non-secret discrimination fields safe to persist.

        Includes an honest ``validation_source`` and ``summary`` line so status
        APIs/CLI/lab strings cannot omit the LOCAL-FIXTURE label (VAL-TEEREQ-003/010).
        """

        # Lazy import keeps types free of classification helper cycles at import time.
        from .classification import (
            decision_public_surface,
            decision_validation_source,
            human_summary_line,
        )

        surface = decision_public_surface(self)
        return {
            "accepted": self.accepted,
            "classification": surface["classification"],
            "reason": self.reason.value,
            "provider": self.provider.value,
            "effective_tier": self.effective_tier,
            "evidence_digest": self.evidence_digest,
            "trust_root_fingerprint": self.trust_root_fingerprint,
            "gpu_key_fingerprint": self.gpu_key_fingerprint,
            "image_digest": self.image_digest,
            "nonce_digest": self.nonce_digest,
            "work_unit_id": self.work_unit_id,
            "validated_claims": list(self.validated_claims),
            "detail": self.detail,
            "validation_source": decision_validation_source(self),
            "summary": human_summary_line(self),
            "real_provider_pass": surface["real_provider_pass"],
            "local_fixture_pass": surface["local_fixture_pass"],
        }


def fail_decision(
    *,
    reason: TeeReasonCode,
    provider: TeeProviderKind = TeeProviderKind.UNKNOWN,
    classification: TeeClassification = TeeClassification.FAIL,
    detail: str = "",
    **extra: Any,
) -> TeeDecision:
    return TeeDecision(
        accepted=False,
        classification=classification,
        reason=reason,
        provider=provider,
        effective_tier=0,
        detail=detail,
        **extra,
    )
