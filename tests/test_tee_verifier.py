"""Prism-only fail-closed TEE verifier: local fixtures + rejection matrix.

Provider credentials remain unset; no network calls; output classification is never
REAL-PROVIDER PASS for these fixtures (LOCAL-FIXTURE PASS only).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from prism_challenge.audit import effective_tier
from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature, compute_tier
from prism_challenge.tee import (
    InMemoryNonceStore,
    TeeClassification,
    TeeReasonCode,
    TeeVerifier,
    TeeVerifierConfig,
)
from prism_challenge.tee.adapters import LiumAdapter, TargonAdapter, select_adapter
from prism_challenge.tee.crypto_local import DEFAULT_MEASUREMENTS, LocalFixtureAuthority
from prism_challenge.tee.evidence import EvidenceParseError, parse_attestation_mapping
from prism_challenge.tee.verifier import compute_effective_tier_with_tee

IMAGE = "sha256:" + "ab" * 32
MANIFEST = "cd" * 32
UNIT = "unit-tee-1"
WORKER_PUB = "5WorkerLocalFixturePubkeyExample0001"
NONCE = "nonce-abcdef0123456789"


def _authority() -> LocalFixtureAuthority:
    return LocalFixtureAuthority.generate(now=datetime(2026, 7, 12, 12, 0, tzinfo=UTC))


def _cfg(auth: LocalFixtureAuthority, **overrides: Any) -> TeeVerifierConfig:
    base = dict(
        enabled=True,
        mode="local_fixture",
        expected_provider="local_fixture",
        expected_issuer=auth.issuer,
        expected_audience="prism.tee.verify",
        tdx_trust_roots_pem=(auth.ca_pem(),),
        gpu_trusted_keys_pem={auth.gpu_kid: auth.gpu_public_pem()},
        expected_image_digest=IMAGE,
        allowed_measurements=dict(DEFAULT_MEASUREMENTS),
        challenge_slug="prism",
        workload_id="prism",
        workload_version="1",
        require_nonce_store=True,
        max_age_seconds=3_600,
        clock_skew_seconds=30,
    )
    base.update(overrides)
    return TeeVerifierConfig(**base)


def _proof(attestation: dict[str, Any] | None, *, tier: int = 2) -> ExecutionProof:
    return ExecutionProof(
        version=1,
        tier=tier,  # type: ignore[arg-type]
        manifest_sha256=MANIFEST,
        image_digest=IMAGE if tier >= 1 else None,
        provider=ProviderInfo(name="local_fixture", pod_id="pod-1", executor_id="ex-1"),
        worker_signature=WorkerSignature(worker_pubkey=WORKER_PUB, sig="0xab"),
        attestation=attestation,
    )


def _fixture(auth: LocalFixtureAuthority, **kwargs: Any) -> dict[str, Any]:
    defaults = dict(
        nonce=NONCE,
        work_unit_id=UNIT,
        submission_id=UNIT,
        image_digest=IMAGE,
        workload_id="prism",
        workload_version="1",
        challenge_slug="prism",
        manifest_sha256=MANIFEST,
        worker_pubkey=WORKER_PUB,
        now=datetime(2026, 7, 12, 12, 0, tzinfo=UTC),
    )
    defaults.update(kwargs)
    return auth.build_attestation(**defaults)


@pytest.fixture(autouse=True)
def _unset_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIUM_API_KEY", raising=False)
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    # Guard against accidental network use in this suite.
    assert os.environ.get("LIUM_API_KEY") is None
    assert os.environ.get("TARGON_API_KEY") is None


def test_populated_opaque_fields_do_not_claim_or_elevate_tier() -> None:
    opaque = {"tdx_quote_b64": "QUOTE", "gpu_eat_jwt": "JWT"}
    asserted = compute_tier(
        image_digest=IMAGE,
        provider=ProviderInfo(name="lium", pod_id="p"),
        attestation=opaque,
    )
    assert asserted != 2
    proof = _proof(opaque, tier=1)
    # Claimed dishonest 2 in a hand-built envelope stays verifiable-as-0 without TEE.
    dishonest = ExecutionProof(
        version=1,
        tier=2,
        manifest_sha256=MANIFEST,
        image_digest=IMAGE,
        provider=ProviderInfo(name="lium", pod_id="p"),
        worker_signature=WorkerSignature(worker_pubkey=WORKER_PUB, sig="0xab"),
        attestation=opaque,
    )
    assert effective_tier(dishonest, pinned_image_digest=IMAGE) == 0
    assert effective_tier(proof, pinned_image_digest=IMAGE) == 1


def test_valid_local_fixture_is_local_fixture_pass_only() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    att = _fixture(auth)
    proof = _proof(att)

    decision = asyncio.run(verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert decision.accepted is True
    assert decision.classification is TeeClassification.LOCAL_FIXTURE_PASS
    assert decision.classification.value == "LOCAL-FIXTURE PASS"
    assert decision.effective_tier == 2
    assert decision.reason is TeeReasonCode.ACCEPTED_LOCAL_FIXTURE
    assert decision.metadata.get("validation_source") == "local_fixture"
    assert decision.trust_root_fingerprint
    assert decision.gpu_key_fingerprint
    assert decision.evidence_digest
    assert effective_tier(proof, pinned_image_digest=IMAGE, tee_decision=decision) == 2
    # Must never look like real-provider pass
    assert decision.classification is not TeeClassification.REAL_PROVIDER_PASS


def test_missing_disabled_and_misconfigured_fail_closed() -> None:
    auth = _authority()
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    att = _fixture(auth)
    proof = _proof(att)

    disabled = TeeVerifier(_cfg(auth, enabled=False), nonce_store=store, now_fn=lambda: now)
    d1 = asyncio.run(disabled.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert d1.accepted is False
    assert d1.reason is TeeReasonCode.VERIFIER_DISABLED

    empty = TeeVerifier(
        TeeVerifierConfig(enabled=True, tdx_trust_roots_pem=(), gpu_trusted_keys_pem={}),
        nonce_store=store,
        now_fn=lambda: now,
    )
    d2 = asyncio.run(empty.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert d2.accepted is False
    assert d2.effective_tier == 0
    assert d2.reason in {
        TeeReasonCode.VERIFIER_MISCONFIGURED,
        TeeReasonCode.NONCE_MISSING,
        TeeReasonCode.IMAGE_DIGEST_MISMATCH,
    } or d2.classification in {TeeClassification.BLOCKED, TeeClassification.FAIL}

    no_store = TeeVerifier(_cfg(auth), nonce_store=None, now_fn=lambda: now)
    d3 = asyncio.run(no_store.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert d3.accepted is False
    assert d3.reason is TeeReasonCode.VERIFIER_MISCONFIGURED


@pytest.mark.parametrize(
    "value",
    [None, {}, [], "", "  ", 12, True, {"tdx_quote_b64": "x"}, {"gpu_eat_jwt": "y"}],
)
def test_missing_null_empty_wrong_type_fail(value: object) -> None:
    with pytest.raises(EvidenceParseError):
        parse_attestation_mapping(value)


def test_unknown_version_provider_type_field_fail() -> None:
    auth = _authority()
    att = _fixture(auth)
    bad_version = {**att, "version": 99}
    with pytest.raises(EvidenceParseError) as e1:
        parse_attestation_mapping(bad_version)
    assert e1.value.reason is TeeReasonCode.EVIDENCE_UNKNOWN_VERSION

    bad_provider = {**att, "provider": "othercloud"}
    with pytest.raises(EvidenceParseError) as e2:
        parse_attestation_mapping(bad_provider)
    assert e2.value.reason is TeeReasonCode.EVIDENCE_UNKNOWN_PROVIDER

    bad_type = {**att, "evidence_type": "not-a-type"}
    with pytest.raises(EvidenceParseError) as e3:
        parse_attestation_mapping(bad_type)
    assert e3.value.reason is TeeReasonCode.EVIDENCE_UNKNOWN_TYPE

    extra = {**att, "unexpected_critical": "x"}
    with pytest.raises(EvidenceParseError) as e4:
        parse_attestation_mapping(extra)
    assert e4.value.reason is TeeReasonCode.EVIDENCE_UNKNOWN_FIELD

    locator = {**att, "jku": "https://evil.example/jwks"}
    with pytest.raises(EvidenceParseError) as e5:
        parse_attestation_mapping(locator)
    assert e5.value.reason is TeeReasonCode.TRUST_LOCATOR_FORBIDDEN


@pytest.mark.parametrize(
    "version",
    [
        True,  # bool is int subclass; must not pass via True == 1
        1.0,  # float equal to 1 must not coerce
        "1",  # string form rejected at closed schema boundary
        "1.0",
        False,
        0,
        1.5,
        None,
    ],
)
def test_version_coerced_ambiguous_forms_rejected(version: object) -> None:
    """VAL-TEE-006: reject coerced/ambiguous version forms at parse boundary."""
    auth = _authority()
    att = {**_fixture(auth), "version": version}
    with pytest.raises(EvidenceParseError) as exc:
        parse_attestation_mapping(att)
    assert exc.value.reason is TeeReasonCode.EVIDENCE_UNKNOWN_VERSION


def test_version_strict_int_one_accepted() -> None:
    """VAL-TEE-006: canonical integer version=1 parses closed schema."""
    auth = _authority()
    att = {**_fixture(auth), "version": 1}
    parsed = parse_attestation_mapping(att)
    assert parsed.version == "1"
    assert parsed.provider.value == "local_fixture"


def test_oversize_and_encoding_reject() -> None:
    with pytest.raises(EvidenceParseError) as e1:
        parse_attestation_mapping(
            {
                "version": 1,
                "provider": "local_fixture",
                "evidence_type": "prism.tee.v1",
                "tdx_quote_b64": "!!!!",
                "gpu_eat_jwt": "a.b.c",
            }
        )
    assert e1.value.reason is TeeReasonCode.ENCODING_INVALID

    huge = "A" * 70_000
    with pytest.raises(EvidenceParseError) as e2:
        parse_attestation_mapping(
            {
                "version": 1,
                "provider": "local_fixture",
                "evidence_type": "prism.tee.v1",
                "tdx_quote_b64": base64.b64encode(b"x").decode() + huge,
                "gpu_eat_jwt": "aaa.bbb.ccc",
            }
        )
    assert e2.value.reason in {
        TeeReasonCode.EVIDENCE_OVERSIZE,
        TeeReasonCode.ENCODING_INVALID,
    }


def test_tdx_bitflip_and_forged_gpu_reject() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    att = _fixture(auth)
    raw = bytearray(base64.b64decode(att["tdx_quote_b64"]))
    raw[20] ^= 0x01
    flipped = {
        **att,
        "tdx_quote_b64": base64.b64encode(bytes(raw)).decode("ascii"),
    }
    d = asyncio.run(verifier.verify_proof(_proof(flipped), work_unit_id=UNIT, expected_nonce=NONCE))
    assert d.accepted is False
    assert d.reason in {
        TeeReasonCode.ENCODING_INVALID,
        TeeReasonCode.TDX_SIGNATURE_INVALID,
        TeeReasonCode.EVIDENCE_MALFORMED,
        TeeReasonCode.TDX_CHAIN_UNTRUSTED,
    }

    # alg=none
    hdr = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    pay = base64.urlsafe_b64encode(json.dumps({"iss": auth.issuer}).encode()).rstrip(b"=")
    none_jwt = f"{hdr.decode()}.{pay.decode()}."
    d2 = asyncio.run(
        verifier.verify_proof(
            _proof({**att, "gpu_eat_jwt": none_jwt}),
            work_unit_id=UNIT,
            expected_nonce=NONCE + "-2",
        )
    )
    assert d2.accepted is False
    assert d2.reason in {
        TeeReasonCode.GPU_ALG_CONFUSION,
        TeeReasonCode.GPU_SIGNATURE_INVALID,
        TeeReasonCode.GPU_UNTRUSTED_KEY,
        TeeReasonCode.ENCODING_INVALID,
    }


def test_nonce_replay_and_mismatch() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    att = _fixture(auth)
    proof = _proof(att)

    ok = asyncio.run(verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert ok.accepted is True
    replay = asyncio.run(verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert replay.accepted is False
    assert replay.reason is TeeReasonCode.NONCE_REPLAY

    # previously valid body vs new nonce
    mismatched = asyncio.run(
        verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce="different-nonce")
    )
    assert mismatched.accepted is False
    assert mismatched.reason is TeeReasonCode.NONCE_MISMATCH


def test_nonce_not_consumed_on_failure() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    # wrong image in fixture
    att = _fixture(auth, image_digest="sha256:" + "00" * 32)
    bad = asyncio.run(verifier.verify_proof(_proof(att), work_unit_id=UNIT, expected_nonce=NONCE))
    assert bad.accepted is False
    assert asyncio.run(store.is_consumed(NONCE)) is False

    good = _fixture(auth)
    ok = asyncio.run(verifier.verify_proof(_proof(good), work_unit_id=UNIT, expected_nonce=NONCE))
    assert ok.accepted is True
    assert asyncio.run(store.is_consumed(NONCE)) is True


def test_workload_image_measurement_gpu_bindings() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)

    wrong_unit = _fixture(auth, work_unit_id="other-unit", submission_id="other-unit")
    d1 = asyncio.run(
        verifier.verify_proof(_proof(wrong_unit), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    assert d1.accepted is False
    assert d1.reason is TeeReasonCode.WORKLOAD_MISMATCH

    wrong_ms = dict(DEFAULT_MEASUREMENTS)
    wrong_ms["mrtd"] = "ff" * 32
    d2 = asyncio.run(
        verifier.verify_proof(
            _proof(_fixture(auth, measurements=wrong_ms, nonce=NONCE + "m")),
            work_unit_id=UNIT,
            expected_nonce=NONCE + "m",
        )
    )
    assert d2.accepted is False
    assert d2.reason is TeeReasonCode.MEASUREMENT_MISMATCH

    d3 = asyncio.run(
        verifier.verify_proof(
            _proof(_fixture(auth, gpu_model="GTX1080", nonce=NONCE + "g")),
            work_unit_id=UNIT,
            expected_nonce=NONCE + "g",
        )
    )
    assert d3.accepted is False
    assert d3.reason is TeeReasonCode.GPU_IDENTITY_MISMATCH

    d4 = asyncio.run(
        verifier.verify_proof(
            _proof(_fixture(auth, debug=True, nonce=NONCE + "d")),
            work_unit_id=UNIT,
            expected_nonce=NONCE + "d",
        )
    )
    assert d4.accepted is False
    assert d4.reason is TeeReasonCode.TCB_POLICY_REJECTED


def test_cross_splice_tdx_gpu_rejected() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    a = _fixture(auth, session_id="session-a", nonce=NONCE + "a")
    b = _fixture(auth, session_id="session-b", nonce=NONCE + "b")
    spliced = {**a, "gpu_eat_jwt": b["gpu_eat_jwt"]}
    d = asyncio.run(
        verifier.verify_proof(_proof(spliced), work_unit_id=UNIT, expected_nonce=NONCE + "a")
    )
    assert d.accepted is False
    assert d.reason in {
        TeeReasonCode.CROSS_BINDING_MISMATCH,
        TeeReasonCode.NONCE_MISMATCH,
        TeeReasonCode.WORKLOAD_MISMATCH,
    }


def test_claimed_tier_never_controls_effective() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)

    # Claimed 0 with full provenance metadata is still effective 0.
    t0 = ExecutionProof(
        version=1,
        tier=0,
        manifest_sha256=MANIFEST,
        image_digest=None,
        provider=ProviderInfo(name="local_fixture", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey=WORKER_PUB, sig="0xab"),
        attestation=None,
    )
    assert compute_effective_tier_with_tee(t0, pinned_image_digest=IMAGE, tee_decision=None) == 0

    t1 = _proof(None, tier=1)
    assert compute_effective_tier_with_tee(t1, pinned_image_digest=IMAGE, tee_decision=None) == 1

    # Claimed 2 without a TeeDecision is effective 0 even with structured-looking fields.
    fake_t2 = _proof(
        {
            "version": 1,
            "provider": "local_fixture",
            "evidence_type": "prism.tee.v1",
            "tdx_quote_b64": base64.b64encode(b"not-real").decode(),
            "gpu_eat_jwt": "aaa.bbb.ccc",
        },
        tier=2,
    )
    assert (
        compute_effective_tier_with_tee(fake_t2, pinned_image_digest=IMAGE, tee_decision=None) == 0
    )

    # Only accepted decision elevates 2.
    n = NONCE + "claim"
    att = _fixture(auth, nonce=n)
    proof = _proof(att, tier=2)
    decision = asyncio.run(verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce=n))
    assert decision.accepted is True
    assert (
        compute_effective_tier_with_tee(proof, pinned_image_digest=IMAGE, tee_decision=decision)
        == 2
    )
    # Dishonest claim 1 with accepted TEE decision still uses verifier tier.
    low_claim = _proof(att, tier=1)
    assert (
        compute_effective_tier_with_tee(low_claim, pinned_image_digest=IMAGE, tee_decision=decision)
        == 2
    )


def test_lium_and_targon_blocked() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)

    lium_att = _fixture(auth, provider="lium")
    # evidence parse allows lium provider name; verifier blocks elevation
    d1 = asyncio.run(
        verifier.verify_proof(_proof(lium_att), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    assert d1.accepted is False
    assert d1.classification is TeeClassification.BLOCKED
    assert d1.reason is TeeReasonCode.PROVIDER_BLOCKED

    targon_att = _fixture(auth, provider="targon", nonce=NONCE + "t")
    d2 = asyncio.run(
        verifier.verify_proof(_proof(targon_att), work_unit_id=UNIT, expected_nonce=NONCE + "t")
    )
    assert d2.accepted is False
    assert d2.classification is TeeClassification.BLOCKED
    assert d2.reason is TeeReasonCode.PROVIDER_FUTURE_BLOCKED

    assert select_adapter("lium", cfg) is not None
    ready, reason, _ = LiumAdapter().readiness(cfg)
    assert ready is False
    assert reason is TeeReasonCode.PROVIDER_BLOCKED
    ready_t, reason_t, _ = TargonAdapter().readiness(cfg)
    assert ready_t is False
    assert reason_t is TeeReasonCode.PROVIDER_FUTURE_BLOCKED


def test_metadata_only_never_tier2() -> None:
    proof = ExecutionProof(
        version=1,
        tier=1,
        manifest_sha256=MANIFEST,
        image_digest=IMAGE,
        provider=ProviderInfo(name="lium", pod_id="pod-1", executor_id="ex-1", miner_hotkey="hk"),
        worker_signature=WorkerSignature(worker_pubkey=WORKER_PUB, sig="0xab"),
        attestation=None,
    )
    assert effective_tier(proof, pinned_image_digest=IMAGE) == 1
    assert compute_effective_tier_with_tee(proof, pinned_image_digest=IMAGE, tee_decision=None) != 2


def test_freshness_window() -> None:
    auth = _authority()
    cfg = _cfg(auth, max_age_seconds=60)
    store = InMemoryNonceStore()
    issued = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    att = _fixture(auth, now=issued)
    late = issued + timedelta(hours=2)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: late)
    d = asyncio.run(verifier.verify_proof(_proof(att), work_unit_id=UNIT, expected_nonce=NONCE))
    assert d.accepted is False
    assert d.reason is TeeReasonCode.FRESHNESS_INVALID


def test_concurrent_nonce_single_winner() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    att = _fixture(auth)
    proof = _proof(att)

    async def run_both() -> list[bool]:
        a = asyncio.create_task(
            verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE)
        )
        b = asyncio.create_task(
            verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE)
        )
        results = await asyncio.gather(a, b)
        return [r.accepted for r in results]

    outcomes = asyncio.run(run_both())
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1


def test_no_secrets_in_decision_record() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    decision = asyncio.run(
        verifier.verify_proof(_proof(_fixture(auth)), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    blob = json.dumps(decision.to_audit_record())
    assert "BEGIN PRIVATE" not in blob
    assert auth.gpu_key.private_numbers().private_value.to_bytes(32, "big").hex() not in blob
    assert "LIUM_API_KEY" not in blob
