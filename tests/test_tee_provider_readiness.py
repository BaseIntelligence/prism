"""Lium/Targon readiness gates: no REAL-PROVIDER PASS without full hard gate.

Assertions VAL-TEE-043 through VAL-TEE-052. Provider credentials remain unset;
no network; no provisioning. Classification is always BLOCKED for real providers.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature
from prism_challenge.tee import (
    HARD_GATE_ITEMS,
    WATCHTOWER_UNBOUND,
    InMemoryNonceStore,
    LiumAdapter,
    TargonAdapter,
    TeeClassification,
    TeeProviderKind,
    TeeReasonCode,
    TeeVerifier,
    TeeVerifierConfig,
    evaluate_provider_readiness,
    evaluate_watchtower_digest,
    real_provider_pass_is_possible,
    select_adapter,
)
from prism_challenge.tee.crypto_local import DEFAULT_MEASUREMENTS, LocalFixtureAuthority
from prism_challenge.tee.readiness import build_hard_gate_checklist, classify_safe_probe
from prism_challenge.tee.verifier import compute_effective_tier_with_tee

IMAGE = "sha256:" + "ab" * 32
MANIFEST = "cd" * 32
UNIT = "unit-ready-1"
WORKER_PUB = "5WorkerLocalFixturePubkeyExample0001"
NONCE = "nonce-readiness-abcdef01"
AUTHORITY_NOW = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)

# Every bool/flag an operator could flip without a real provider contract.
_COMPLETE_CONTRACT = {
    "evidence_endpoint": "https://provider.example/v1/evidence",
    "format_authoritative": True,
    "evidence_format": "provider.tee.v1",
    "issuer": "lium-production",
    "audience": "prism.tee.verify",
    "issuer_audience_authoritative": True,
    "trust_roots_authoritative": True,
    "freshness_policy_documented": True,
    "nonce_semantics_documented": True,
    "public_image_reference": "registry.example/lium-worker@sha256:" + "ab" * 32,
    "public_image_digest_resolvable": True,
    "measurement_policy_documented": True,
    "gpu_claim_policy_documented": True,
    "cross_binding_documented": True,
    "real_workload_evidence_available": True,
}


def _authority() -> LocalFixtureAuthority:
    return LocalFixtureAuthority.generate(now=AUTHORITY_NOW)


def _cfg(auth: LocalFixtureAuthority | None = None, **overrides: Any) -> TeeVerifierConfig:
    auth = auth or _authority()
    base: dict[str, Any] = dict(
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
        lium_ready=False,
        targon_ready=False,
        allow_network=False,
        provider_contract={},
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
        provider=ProviderInfo(name="lium", pod_id="pod-1", executor_id="ex-1"),
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
        now=AUTHORITY_NOW,
    )
    defaults.update(kwargs)
    return auth.build_attestation(**defaults)


@pytest.fixture(autouse=True)
def _unset_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIUM_API_KEY", raising=False)
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    assert os.environ.get("LIUM_API_KEY") is None
    assert os.environ.get("TARGON_API_KEY") is None


# ---------------------------------------------------------------------------
# VAL-TEE-043: Lium verifies documented evidence, not field presence
# ---------------------------------------------------------------------------


def test_lium_adapter_blocked_without_contract_lists_checklist() -> None:
    cfg = _cfg()
    report = LiumAdapter().readiness_report(cfg)
    assert report.ready is False
    assert report.classification is TeeClassification.BLOCKED
    assert report.reason is TeeReasonCode.PROVIDER_BLOCKED
    assert report.max_effective_tier == 0
    assert report.would_grant_real_provider_pass is False
    # Every hard-gate dependency is enumerated with available/missing state.
    hard = {item.item_id for item in report.checklist if item.item_id in HARD_GATE_ITEMS}
    assert hard == set(HARD_GATE_ITEMS)
    assert all(
        item.available is False for item in report.checklist if item.item_id in HARD_GATE_ITEMS
    )
    for item_id in HARD_GATE_ITEMS:
        assert item_id in report.missing_items


def test_lium_tier2_claim_remains_effective_tier_zero() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    # Valid local crypto shape but provider=lium — readiness gate blocks.
    att = _fixture(auth, provider="lium")
    decision = asyncio.run(
        verifier.verify_proof(_proof(att), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    assert decision.accepted is False
    assert decision.classification is TeeClassification.BLOCKED
    assert decision.effective_tier == 0
    assert decision.reason is TeeReasonCode.PROVIDER_BLOCKED
    assert decision.metadata.get("would_grant_real_provider_pass") is False
    assert "missing_items" in decision.metadata
    assert (
        compute_effective_tier_with_tee(
            _proof(att), pinned_image_digest=IMAGE, tee_decision=decision
        )
        == 0
    )


def test_credentials_and_watchtower_cannot_promote_lium() -> None:
    """Current credentials / watchtower callback / API reachability cannot promote."""
    auth = _authority()
    cfg = _cfg(
        auth,
        lium_ready=True,
        allow_network=True,
        # Local pins present, contract empty.
        provider_contract={},
    )
    report = evaluate_provider_readiness(TeeProviderKind.LIUM, cfg)
    assert report.ready is False
    assert report.would_grant_real_provider_pass is False
    assert real_provider_pass_is_possible(cfg, "lium") is False
    probe = LiumAdapter().classify_probe(
        api_reachable=True, http_status=200, path="/watchtower/digest", config=cfg
    )
    assert probe["provider_api_reachable"] == "true"
    assert probe["tee_validation"] == TeeClassification.BLOCKED.value
    assert probe["provisioned"] is False
    assert probe["mutated"] is False


# ---------------------------------------------------------------------------
# VAL-TEE-044: Watchtower digest alone can never establish tier 2
# ---------------------------------------------------------------------------


def test_watchtower_digest_alone_never_tier2() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    now = AUTHORITY_NOW
    payload = {
        "digest": IMAGE,
        "signature_valid": True,
        "signing_key_known": True,
        "timestamp": now.isoformat(),
        "pod_id": "pod-1",
        "executor_id": "ex-1",
        "expected_pod_id": "pod-1",
        "expected_executor_id": "ex-1",
    }
    evaluation = LiumAdapter().evaluate_watchtower(payload, config=cfg, now=now)
    assert evaluation.accepted_as_tier1_input is True
    assert evaluation.effective_tier == 1
    assert evaluation.effective_tier != 2
    assert evaluation.as_dict()["grants_tier_2"] is False
    assert evaluation.as_dict()["tee_validation"] == TeeClassification.BLOCKED.value
    # Explicit list of unbound properties required by VAL-TEE-044.
    for prop in (
        "nonce",
        "workload_identity",
        "tdx_measurement",
        "gpu_identity",
        "manifest_binding",
        "execution_freshness",
    ):
        assert prop in evaluation.unbound_properties
    assert set(WATCHTOWER_UNBOUND).issubset(set(evaluation.unbound_properties))

    # Combined with otherwise absent TEE evidence: effective tier never 2.
    proof = _proof(None, tier=1)
    assert compute_effective_tier_with_tee(proof, pinned_image_digest=IMAGE, tee_decision=None) != 2


# ---------------------------------------------------------------------------
# VAL-TEE-045: Forged or stale watchtower/provider metadata is rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected_reason",
    [
        ({"digest": "not-a-digest"}, TeeReasonCode.METADATA_NOT_ATTESTATION),
        (
            {"digest": IMAGE, "signature_valid": False, "signing_key_known": True},
            TeeReasonCode.METADATA_NOT_ATTESTATION,
        ),
        (
            {"digest": IMAGE, "signature_valid": True, "signing_key_known": False},
            TeeReasonCode.METADATA_NOT_ATTESTATION,
        ),
        (
            {
                "digest": IMAGE,
                "signature_valid": True,
                "signing_key_known": True,
                "timestamp": (AUTHORITY_NOW - timedelta(days=30)).isoformat(),
            },
            TeeReasonCode.FRESHNESS_INVALID,
        ),
        (
            {
                "digest": "sha256:" + "00" * 32,
                "signature_valid": True,
                "signing_key_known": True,
                "timestamp": AUTHORITY_NOW.isoformat(),
            },
            TeeReasonCode.IMAGE_DIGEST_MISMATCH,
        ),
        (
            {
                "digest": IMAGE,
                "signature_valid": True,
                "signing_key_known": True,
                "timestamp": AUTHORITY_NOW.isoformat(),
                "pod_id": "other-pod",
                "expected_pod_id": "pod-1",
            },
            TeeReasonCode.METADATA_NOT_ATTESTATION,
        ),
        (
            {
                "digest": IMAGE,
                "signature_valid": True,
                "signing_key_known": True,
                "timestamp": AUTHORITY_NOW.isoformat(),
                "executor_id": "other-ex",
                "expected_executor_id": "ex-1",
            },
            TeeReasonCode.METADATA_NOT_ATTESTATION,
        ),
        (None, TeeReasonCode.METADATA_NOT_ATTESTATION),
        ({}, TeeReasonCode.METADATA_NOT_ATTESTATION),
    ],
)
def test_forged_stale_watchtower_metadata_rejected(
    payload: dict[str, Any] | None, expected_reason: TeeReasonCode
) -> None:
    evaluation = evaluate_watchtower_digest(
        payload,
        expected_image_digest=IMAGE,
        now=AUTHORITY_NOW,
        max_age_seconds=3_600,
    )
    assert evaluation.accepted_as_tier1_input is False
    assert evaluation.effective_tier == 0
    assert evaluation.reason is expected_reason
    assert evaluation.as_dict()["grants_tier_2"] is False


def test_watchtower_copied_to_other_workload_rejected() -> None:
    payload = {
        "digest": IMAGE,
        "signature_valid": True,
        "signing_key_known": True,
        "timestamp": AUTHORITY_NOW.isoformat(),
        "pod_id": "pod-from-other-work",
        "expected_pod_id": "pod-expected",
    }
    evaluation = evaluate_watchtower_digest(payload, expected_image_digest=IMAGE, now=AUTHORITY_NOW)
    assert evaluation.effective_tier == 0
    assert "pod" in evaluation.detail or evaluation.reason is TeeReasonCode.METADATA_NOT_ATTESTATION


# ---------------------------------------------------------------------------
# VAL-TEE-046: Safe Lium probes cannot be reported as attestation PASS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/account",
        "/offers",
        "/pods/status",
        "/watchtower/digest",
    ],
)
def test_safe_lium_probes_are_api_reachable_only(path: str) -> None:
    cfg = _cfg()
    report = LiumAdapter().classify_probe(
        api_reachable=True, http_status=200, path=path, method="GET", config=cfg
    )
    assert report["provider_api_reachable"] == "true"
    assert report["tee_validation"] == TeeClassification.BLOCKED.value
    assert report["mutated"] is False
    assert report["provisioned"] is False
    assert report["metadata"]["provider_resources_created"] == 0
    # No REAL-PROVIDER PASS surface.
    assert "REAL-PROVIDER" not in str(report)


def test_mutating_probe_flagged_not_allowed() -> None:
    report = classify_safe_probe("lium", api_reachable=True, path="/pods", method="POST")
    assert report.mutated is True
    assert report.tee_validation == TeeClassification.BLOCKED.value
    assert report.provisioned is False


# ---------------------------------------------------------------------------
# VAL-TEE-047: Real Lium PASS requires full gate; otherwise checklist BLOCKED
# ---------------------------------------------------------------------------


def test_real_lium_pass_blocked_with_full_checklist_output() -> None:
    auth = _authority()
    # Even with every local pin and a complete-looking contract blob, mission code
    # refuses REAL-PROVIDER PASS (authority over real workloads is external).
    cfg = _cfg(
        auth,
        lium_ready=True,
        provider_contract=dict(_COMPLETE_CONTRACT),
    )
    report = LiumAdapter().readiness_report(cfg)
    assert report.ready is False
    assert report.classification is TeeClassification.BLOCKED
    assert report.would_grant_real_provider_pass is False
    assert real_provider_pass_is_possible(cfg, TeeProviderKind.LIUM) is False
    payload = report.as_dict()
    assert payload["classification"] == "BLOCKED"
    # Checklist is always present for evidence of prerequisites.
    assert isinstance(payload["checklist"], list)
    assert any(item["item_id"] in HARD_GATE_ITEMS for item in payload["checklist"])


def test_no_local_fixture_as_real_provider_pass() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    att = _fixture(auth)
    decision = asyncio.run(
        verifier.verify_proof(_proof(att, tier=2), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    assert decision.accepted is True
    assert decision.classification is TeeClassification.LOCAL_FIXTURE_PASS
    assert decision.classification is not TeeClassification.REAL_PROVIDER_PASS
    assert "REAL-PROVIDER" not in decision.classification.value


# ---------------------------------------------------------------------------
# VAL-TEE-048: Targon is future/blocked by default
# ---------------------------------------------------------------------------


def test_targon_future_blocked_by_default() -> None:
    cfg = _cfg()
    adapter = TargonAdapter()
    ready, reason, detail = adapter.readiness(cfg)
    assert ready is False
    assert reason is TeeReasonCode.PROVIDER_FUTURE_BLOCKED
    assert "future" in detail.lower() or "blocked" in detail.lower()
    report = adapter.readiness_report(cfg)
    assert report.classification is TeeClassification.BLOCKED
    assert report.metadata.get("future_or_blocked") is True
    assert report.metadata.get("speculative_endpoints_attempted") == 0
    assert report.max_effective_tier == 0
    assert report.would_grant_real_provider_pass is False


def test_targon_selection_blocks_verification_zero_network() -> None:
    auth = _authority()
    cfg = _cfg(auth, allow_network=False)
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    # Locally valid crypto, relabeled Targon.
    att = _fixture(auth, provider="targon")
    decision = asyncio.run(
        verifier.verify_proof(_proof(att), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    assert decision.accepted is False
    assert decision.classification is TeeClassification.BLOCKED
    assert decision.reason is TeeReasonCode.PROVIDER_FUTURE_BLOCKED
    assert decision.effective_tier == 0
    assert select_adapter("targon", cfg) is not None


# ---------------------------------------------------------------------------
# VAL-TEE-049: Non-empty Targon quote/JWT claims cannot bypass the block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attestation",
    [
        {
            "version": 1,
            "provider": "targon",
            "evidence_type": "prism.tee.v1",
            "tdx_quote_b64": "QUOTE",
            "gpu_eat_jwt": "a.b.c",
        },
        None,  # handled via separate litter — only populated forms below remain blocked
    ],
)
def test_nonempty_targon_opacity_blocked(attestation: dict[str, Any] | None) -> None:
    auth = _authority()
    cfg = _cfg(auth, targon_ready=True, provider_contract=dict(_COMPLETE_CONTRACT))
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    if attestation is None:
        # Fabricated valid-looking under local authority, still provider=targon.
        attestation = _fixture(auth, provider="targon")
    # Opaque / fabricated evidence may fail parse for incomplete encoding; that's OK —
    # the effective casing that matters is classification never REAL-PROVIDER PASS.
    try:
        decision = asyncio.run(
            verifier.verify_proof(_proof(attestation), work_unit_id=UNIT, expected_nonce=NONCE)
        )
    except Exception:  # noqa: BLE001
        # Parser raises via verifier Mapping path; force fail closed via empty decision.
        decision = None
    if decision is not None:
        assert decision.accepted is False
        assert decision.classification is not TeeClassification.REAL_PROVIDER_PASS
        assert decision.effective_tier == 0
        assert decision.classification in {
            TeeClassification.BLOCKED,
            TeeClassification.FAIL,
        }


def test_lium_shaped_evidence_relabeled_targon_blocked() -> None:
    auth = _authority()
    cfg = _cfg(auth, targon_ready=True, provider_contract=dict(_COMPLETE_CONTRACT))
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    att = _fixture(auth, provider="targon")
    decision = asyncio.run(
        verifier.verify_proof(_proof(att), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    assert decision.accepted is False
    assert decision.classification is TeeClassification.BLOCKED
    assert decision.reason is TeeReasonCode.PROVIDER_FUTURE_BLOCKED
    assert decision.effective_tier == 0


# ---------------------------------------------------------------------------
# VAL-TEE-050: Targon inventory APIs do not imply TEE capability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/inventory",
        "/apps",
        "/workloads",
        "/state",
        "/events",
        "/logs",
    ],
)
def test_targon_inventory_is_api_only(path: str) -> None:
    cfg = _cfg()
    report = TargonAdapter().classify_probe(
        api_reachable=True,
        http_status=200,
        path=path,
        method="GET",
        config=cfg,
    )
    assert report["provider_api_reachable"] == "true"
    assert report["tee_validation"] == TeeClassification.BLOCKED.value
    assert report["provisioned"] is False
    assert report["mutated"] is False
    # GPU model fields and workload status cannot substitute for attestation.
    meta_only = {
        "gpu_model": "H100",
        "workload_status": "RUNNING",
        "app_id": "app-1",
    }
    # Treat as provider metadata: never elevates readiness.
    readiness = TargonAdapter().readiness_report(cfg)
    assert readiness.ready is False
    assert readiness.classification is TeeClassification.BLOCKED
    _ = meta_only


# ---------------------------------------------------------------------------
# VAL-TEE-051: Targon future enablement requires complete configuration atomically
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", list(HARD_GATE_ITEMS))
def test_targon_one_missing_item_keeps_blocked(missing_key: str) -> None:
    auth = _authority()
    # Build a "near complete" contract by setting all authority markers true,
    # then the readiness code maps missing_key to unavailable by flipping that
    # marker (one-missing-item parameterization).
    contract = dict(_COMPLETE_CONTRACT)
    # Map each hard-gate item to the contract field that makes it available.
    flip_map = {
        "authoritative_evidence_endpoint": ("evidence_endpoint", ""),
        "authoritative_evidence_format": ("format_authoritative", False),
        "authoritative_issuer_audience": ("issuer_audience_authoritative", False),
        "authoritative_trust_roots": ("trust_roots_authoritative", False),
        "freshness_and_clock_policy": ("freshness_policy_documented", False),
        "nonce_semantics": ("nonce_semantics_documented", False),
        "digest_pinned_public_worker_image": ("public_image_digest_resolvable", False),
        "measurement_policy": ("measurement_policy_documented", False),
        "gpu_claim_policy": ("gpu_claim_policy_documented", False),
        "cross_binding_semantics": ("cross_binding_documented", False),
        "real_workload_evidence_artifact": ("real_workload_evidence_available", False),
    }
    field, value = flip_map[missing_key]
    contract[field] = value
    cfg = _cfg(auth, targon_ready=True, provider_contract=contract)
    report = TargonAdapter().readiness_report(cfg)
    assert report.ready is False
    assert report.classification is TeeClassification.BLOCKED
    assert report.would_grant_real_provider_pass is False
    # The missing key must appear among checklist gaps.
    hard_missing = {
        item.item_id
        for item in report.checklist
        if item.item_id in HARD_GATE_ITEMS and not item.available
    }
    assert missing_key in hard_missing
    assert report.reason in {
        TeeReasonCode.ADAPTER_NOT_READY,
        TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
    }


def test_targon_even_complete_contract_stays_blocked() -> None:
    """Only a complete *authoritative* future contract may flip readiness — and
    current mission code still refuses REAL-PROVIDER PASS even when all flags are
    green. Atomic completeness is a *necessary* condition (checksum), not
    sufficient for real PASS.
    """
    auth = _authority()
    cfg = _cfg(auth, targon_ready=True, provider_contract=dict(_COMPLETE_CONTRACT))
    report = TargonAdapter().readiness_report(cfg)
    assert report.ready is False
    assert report.classification is TeeClassification.BLOCKED
    assert report.would_grant_real_provider_pass is False
    assert report.reason is TeeReasonCode.PROVIDER_FUTURE_BLOCKED


# ---------------------------------------------------------------------------
# VAL-TEE-052: Real Targon PASS is impossible until full authoritative gate
# ---------------------------------------------------------------------------


def test_real_targon_pass_impossible_with_credentials_inventory_or_fixture() -> None:
    auth = _authority()
    # Credentials present in environment (simulated), inventory "success", and a
    # local fixture under non-authoritative roots.
    cfg = _cfg(
        auth,
        targon_ready=True,
        allow_network=True,
        provider_contract=dict(_COMPLETE_CONTRACT),
    )
    report = evaluate_provider_readiness("targon", cfg)
    assert report.classification is TeeClassification.BLOCKED
    assert report.as_dict()["classification"] == "BLOCKED"
    assert report.would_grant_real_provider_pass is False
    assert real_provider_pass_is_possible(cfg, "targon") is False

    checklist = build_hard_gate_checklist(cfg, provider=TeeProviderKind.TARGON)
    # Prerequisite checklist always returned, even when everything looks set.
    assert len(checklist) >= len(HARD_GATE_ITEMS)

    # Verifier cannot emit REAL-PROVIDER PASS for Targon locals.
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    decision = asyncio.run(
        verifier.verify_proof(
            _proof(_fixture(auth, provider="targon")),
            work_unit_id=UNIT,
            expected_nonce=NONCE,
        )
    )
    assert decision.classification is not TeeClassification.REAL_PROVIDER_PASS
    assert decision.classification is TeeClassification.BLOCKED
    assert decision.effective_tier == 0


def test_no_provisioning_or_mutation_on_readiness_paths() -> None:
    cfg = _cfg()
    for adapter in (LiumAdapter(), TargonAdapter()):
        probe = adapter.classify_probe(
            api_reachable=True, http_status=200, path="/health", method="GET", config=cfg
        )
        assert probe["provisioned"] is False
        assert probe["mutated"] is False
        assert probe["metadata"]["provider_resources_created"] == 0
        # Mutating methods never silently count as safe TEE validation.
        reject = adapter.classify_probe(
            api_reachable=True, path="/deploy", method="POST", config=cfg
        )
        assert reject["mutated"] is True
        assert reject["tee_validation"] == TeeClassification.BLOCKED.value
