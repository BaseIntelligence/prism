"""Provider-scoped, fail-closed TEE verifier orchestration."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from base.challenge_sdk.proof import ExecutionProof

from .adapters import blocked_for_provider, select_adapter
from .config import TeeVerifierConfig
from .evidence import EvidenceParseError, parse_attestation_mapping
from .nonce_store import NonceStore, nonce_digest
from .types import (
    TeeClassification,
    TeeDecision,
    TeeProviderKind,
    TeeReasonCode,
    fail_decision,
)
from .verify_crypto import CryptoVerifyError, claim_bindings_match, verify_gpu_eat, verify_tdx_quote

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class TeeVerifier:
    """Verify opaque ExecutionProof.attestation with exact bindings.

    Resource-bounded, secret-safe, and never launches evaluators. Claimed proof
    tiers never control the outcome; only verified evidence can elevate to tier 2
    under LOCAL-FIXTURE mode.
    """

    def __init__(
        self,
        config: TeeVerifierConfig,
        *,
        nonce_store: NonceStore | None = None,
        now_fn: Any | None = None,
    ) -> None:
        self.config = config
        self.nonce_store = nonce_store
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    async def verify_proof(
        self,
        proof: ExecutionProof,
        *,
        work_unit_id: str,
        submission_id: str | None = None,
        expected_nonce: str | None = None,
        consume_nonce: bool = True,
    ) -> TeeDecision:
        """Verify TEE evidence on ``proof`` for ``work_unit_id``.

        When evidence is absent, returns FAIL/missing with effective tier 0.
        Provider readiness for Lium/Targon is always BLOCKED in this feature.
        """

        submission_id = submission_id or work_unit_id
        claimed = int(getattr(proof, "tier", 0) or 0)

        if not self.config.enabled:
            return fail_decision(
                reason=TeeReasonCode.VERIFIER_DISABLED,
                classification=TeeClassification.BLOCKED,
                detail="tee verification disabled",
                work_unit_id=work_unit_id,
            )

        attestation = proof.attestation
        # Provider metadata alone is never attestation (VAL-TEE-026).
        if attestation is None:
            return fail_decision(
                reason=TeeReasonCode.EVIDENCE_MISSING,
                detail="no attestation",
                work_unit_id=work_unit_id,
            )

        try:
            parsed = parse_attestation_mapping(
                attestation,
                max_quote_b64_chars=self.config.max_quote_b64_chars,
                max_jwt_chars=self.config.max_jwt_chars,
                max_evidence_json_bytes=self.config.max_evidence_json_bytes,
            )
        except EvidenceParseError as exc:
            return fail_decision(
                reason=exc.reason,
                detail=exc.detail,
                work_unit_id=work_unit_id,
            )

        provider = parsed.provider
        adapter = select_adapter(provider, self.config)
        if adapter is None:
            return fail_decision(
                reason=TeeReasonCode.EVIDENCE_UNKNOWN_PROVIDER,
                provider=provider,
                detail=f"unknown provider {provider}",
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
            )

        # Provider-scoped readiness gate (Lium/Targon never grant REAL-PROVIDER PASS).
        # Local fixture may proceed only when its readiness report is ready.
        readiness = adapter.readiness_report(self.config)
        if provider is TeeProviderKind.LIUM or provider is TeeProviderKind.TARGON:
            return TeeDecision(
                accepted=False,
                classification=TeeClassification.BLOCKED,
                reason=readiness.reason,
                provider=provider,
                effective_tier=0,
                evidence_digest=parsed.evidence_digest(),
                work_unit_id=work_unit_id,
                detail=readiness.detail,
                metadata={
                    "readiness": readiness.as_dict(),
                    "missing_items": list(readiness.missing_items),
                    "would_grant_real_provider_pass": False,
                },
            )

        if provider is not TeeProviderKind.LOCAL_FIXTURE:
            return fail_decision(
                reason=TeeReasonCode.EVIDENCE_UNKNOWN_PROVIDER,
                provider=provider,
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
            )

        blocked = blocked_for_provider(provider, self.config)
        if blocked is not None:
            return TeeDecision(
                accepted=False,
                classification=blocked.classification,
                reason=blocked.reason,
                provider=provider,
                effective_tier=0,
                evidence_digest=parsed.evidence_digest(),
                work_unit_id=work_unit_id,
                detail=blocked.detail,
            )

        if self.config.require_nonce_store and self.nonce_store is None:
            return fail_decision(
                reason=TeeReasonCode.VERIFIER_MISCONFIGURED,
                provider=provider,
                classification=TeeClassification.BLOCKED,
                detail="nonce store required",
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
            )
        if not expected_nonce:
            return fail_decision(
                reason=TeeReasonCode.NONCE_MISSING,
                provider=provider,
                detail="expected nonce is mandatory",
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
            )

        if not self.config.expected_image_digest or not _DIGEST_RE.fullmatch(
            self.config.expected_image_digest
        ):
            return fail_decision(
                reason=TeeReasonCode.VERIFIER_MISCONFIGURED,
                provider=provider,
                detail="expected_image_digest missing/invalid",
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
            )
        if proof.image_digest != self.config.expected_image_digest:
            return fail_decision(
                reason=TeeReasonCode.IMAGE_DIGEST_MISMATCH,
                provider=provider,
                detail="proof image_digest does not match pin",
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
                image_digest=proof.image_digest,
            )

        now = self._now_fn()
        try:
            tdx_body, root_fp = verify_tdx_quote(parsed.tdx_quote_b64, self.config, now=now)
            gpu_claims, gpu_fp = verify_gpu_eat(parsed.gpu_eat_jwt, self.config)
        except CryptoVerifyError as exc:
            return fail_decision(
                reason=exc.reason,
                provider=provider,
                detail=exc.detail,
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
            )

        expected = {
            "provider": "local_fixture",
            "nonce": expected_nonce,
            "work_unit_id": work_unit_id,
            "submission_id": submission_id,
            "image_digest": self.config.expected_image_digest,
            "workload_id": self.config.workload_id or self.config.challenge_slug,
            "workload_version": self.config.workload_version or "1",
            "challenge_slug": self.config.challenge_slug,
            "manifest_sha256": proof.manifest_sha256,
            "worker_pubkey": proof.worker_signature.worker_pubkey,
            "session_id": tdx_body.get("session_id"),
        }
        # Session must be present in both components (set after tdx parse).
        if not expected["session_id"]:
            return fail_decision(
                reason=TeeReasonCode.CROSS_BINDING_MISMATCH,
                provider=provider,
                detail="missing session_id",
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
            )

        bind_err = claim_bindings_match(
            tdx_body,
            gpu_claims,
            expected=expected,
            config=self.config,
            now=now,
        )
        if bind_err is not None:
            return fail_decision(
                reason=bind_err,
                provider=provider,
                detail=bind_err.value,
                work_unit_id=work_unit_id,
                evidence_digest=parsed.evidence_digest(),
                trust_root_fingerprint=root_fp,
                gpu_key_fingerprint=gpu_fp,
                image_digest=self.config.expected_image_digest,
                nonce_digest=nonce_digest(expected_nonce),
            )

        # Atomic nonce consumption only at final acceptance (VAL-TEE-018).
        if consume_nonce:
            assert self.nonce_store is not None or not self.config.require_nonce_store
            if self.nonce_store is not None:
                ok = await self.nonce_store.try_consume(
                    nonce=expected_nonce,
                    provider=provider.value,
                    work_unit_id=work_unit_id,
                    evidence_digest=parsed.evidence_digest(),
                )
                if not ok:
                    return fail_decision(
                        reason=TeeReasonCode.NONCE_REPLAY,
                        provider=provider,
                        detail="nonce already consumed",
                        work_unit_id=work_unit_id,
                        evidence_digest=parsed.evidence_digest(),
                        trust_root_fingerprint=root_fp,
                        gpu_key_fingerprint=gpu_fp,
                        nonce_digest=nonce_digest(expected_nonce),
                    )

        validated = (
            "issuer",
            "audience",
            "nonce",
            "freshness",
            "workload",
            "image_digest",
            "measurements",
            "gpu_identity",
            "tdx_signature",
            "gpu_signature",
            "cross_binding",
            "proof_binding",
            "trust_root",
        )
        _ = claimed  # claimed tier never elevates
        return TeeDecision(
            accepted=True,
            classification=TeeClassification.LOCAL_FIXTURE_PASS,
            reason=TeeReasonCode.ACCEPTED_LOCAL_FIXTURE,
            provider=provider,
            effective_tier=2,
            evidence_digest=parsed.evidence_digest(),
            trust_root_fingerprint=root_fp,
            gpu_key_fingerprint=gpu_fp,
            image_digest=self.config.expected_image_digest,
            nonce_digest=nonce_digest(expected_nonce),
            work_unit_id=work_unit_id,
            validated_claims=validated,
            detail="local fixture verified",
            metadata={"validation_source": "local_fixture"},
        )


async def verify_proof_tee(
    proof: ExecutionProof,
    *,
    config: TeeVerifierConfig,
    work_unit_id: str,
    submission_id: str | None = None,
    expected_nonce: str | None = None,
    nonce_store: NonceStore | None = None,
    now: datetime | None = None,
    consume_nonce: bool = True,
) -> TeeDecision:
    verifier = TeeVerifier(
        config,
        nonce_store=nonce_store,
        now_fn=(lambda: now) if now is not None else None,
    )
    return await verifier.verify_proof(
        proof,
        work_unit_id=work_unit_id,
        submission_id=submission_id,
        expected_nonce=expected_nonce,
        consume_nonce=consume_nonce,
    )


def compute_effective_tier_with_tee(
    proof: ExecutionProof,
    *,
    pinned_image_digest: str | None,
    tee_decision: TeeDecision | None,
) -> int:
    """Derive effective tier: claimed never controls; TEE elevates only on accept.

    Rules (conservative):
    * accepted LOCAL-FIXTURE (or future REAL-PROVIDER) with effective_tier 2 -> 2
    * else elevating only tier-1 when image digest + provider pod binding match
    * claimed tier 2 with failed/blocked TEE -> 0 (never silently stay at 2,
      never fall through to 1 unless independent tier-1 policy is recorded)
    * claimed tier 0/unknown -> 0
    """

    claimed = int(getattr(proof, "tier", 0) or 0)
    if tee_decision is not None and tee_decision.accepted and tee_decision.effective_tier >= 2:
        return 2

    # Any claim of tier 2 without verified TEE downgrades to 0.
    if claimed >= 2:
        return 0

    if claimed == 1:
        provider = proof.provider
        matches_digest = bool(pinned_image_digest) and proof.image_digest == pinned_image_digest
        has_pod = provider is not None and bool(provider.pod_id)
        return 1 if matches_digest and has_pod else 0

    return 0


def evidence_present_nonempty(attestation: Any) -> bool:
    """True when opaque evidence keys are populated (NOT a verification success)."""

    if not isinstance(attestation, Mapping):
        return False
    return any(bool(attestation.get(k)) for k in ("tdx_quote_b64", "gpu_eat_jwt"))


def bound_evidence_digest(attestation: Any) -> str | None:
    if not isinstance(attestation, Mapping):
        return None
    try:
        import json

        raw = json.dumps(dict(attestation), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()
    except Exception:  # noqa: BLE001
        return None
