"""Classification honesty + provider block matrix (VAL-TEEREQ-003/004/005/010/012).

LOCAL-FIXTURE can accept only as LOCAL-FIXTURE PASS. Lium adapters stay BLOCKED for
REAL-PROVIDER PASS even with credentials / complete-looking contracts / pod metadata.
Targon remains future-blocked. Smuggling LOCAL-FIXTURE as REAL-PROVIDER in API/CLI/lab
surfaces is forbidden. Residual removed LLM env still fails closed at boot. Readiness
HARD_GATE_ITEMS remain in force for any future real-provider path.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

import pytest

from prism_challenge.config import REMOVED_LLM_ENV_NAMES, REMOVED_LLM_SETTING_NAMES, PrismSettings
from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature
from prism_challenge.tee import (
    HARD_GATE_ITEMS,
    LOCAL_FIXTURE_PASS_LABEL,
    REAL_PROVIDER_PASS_LABEL,
    ClassificationHonestyError,
    InMemoryNonceStore,
    LiumAdapter,
    TargonAdapter,
    TeeClassification,
    TeeProviderKind,
    TeeReasonCode,
    TeeVerifier,
    TeeVerifierConfig,
    assert_honest_classification_surface,
    assert_not_real_provider_pass,
    coerce_accepted_fixture_classification,
    decision_public_surface,
    evaluate_provider_readiness,
    human_summary_line,
    real_provider_pass_is_possible,
    smoke_deploy_labels,
)
from prism_challenge.tee.crypto_local import DEFAULT_MEASUREMENTS, LocalFixtureAuthority
from prism_challenge.tee.types import TeeDecision

IMAGE = "sha256:" + "ab" * 32
MANIFEST = "cd" * 32
UNIT = "unit-class-1"
WORKER_PUB = "5WorkerLocalFixturePubkeyExample0001"
NONCE = "nonce-classification-abcdef01"
AUTHORITY_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

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
    "public_image_reference": "registry.example/lium-worker@" + IMAGE,
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


def _proof(
    attestation: dict[str, Any] | None, *, tier: int = 2, provider: str = "local_fixture"
) -> ExecutionProof:
    return ExecutionProof(
        version=1,
        tier=tier,  # type: ignore[arg-type]
        manifest_sha256=MANIFEST,
        image_digest=IMAGE if tier >= 1 else None,
        provider=ProviderInfo(name=provider, pod_id="pod-1", executor_id="ex-1"),
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
    for name in list(REMOVED_LLM_ENV_NAMES):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# VAL-TEEREQ-003: Local fixture accepts only as LOCAL-FIXTURE PASS
# ---------------------------------------------------------------------------


def test_local_fixture_accepts_as_local_fixture_only() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    att = _fixture(auth)
    decision = asyncio.run(
        verifier.verify_proof(_proof(att), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    assert decision.accepted is True
    assert decision.classification is TeeClassification.LOCAL_FIXTURE_PASS
    assert decision.classification is not TeeClassification.REAL_PROVIDER_PASS
    assert decision.classification.value == LOCAL_FIXTURE_PASS_LABEL
    assert decision.classification.value != REAL_PROVIDER_PASS_LABEL
    assert decision.metadata.get("validation_source") == "local_fixture"
    assert decision.effective_tier == 2

    audit = decision.to_audit_record()
    assert audit["classification"] == LOCAL_FIXTURE_PASS_LABEL
    assert audit["validation_source"] == "local_fixture"
    assert audit["local_fixture_pass"] is True
    assert audit["real_provider_pass"] is False
    assert "REAL-PROVIDER" not in audit["classification"]
    assert "REAL-PROVIDER" not in audit["summary"]
    assert "local_fixture" in audit["summary"]


def test_local_fixture_public_surface_and_summary_honest() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    decision = asyncio.run(
        verifier.verify_proof(
            _proof(_fixture(auth)),
            work_unit_id=UNIT,
            expected_nonce=NONCE,
        )
    )
    surface = decision_public_surface(decision)
    summary = human_summary_line(decision)
    assert surface["classification"] == LOCAL_FIXTURE_PASS_LABEL
    assert surface["validation_source"] == "local_fixture"
    assert surface["real_provider_pass"] is False
    assert surface["production_mine_badge"] is False
    assert surface["live_emission_authority"] is False
    assert "LOCAL-FIXTURE" in summary
    assert "local_fixture" in summary
    assert REAL_PROVIDER_PASS_LABEL not in summary
    assert_honest_classification_surface(surface)
    assert_honest_classification_surface(
        {
            "classification": surface["classification"],
            "validation_source": surface["validation_source"],
            "provider": surface["provider"],
            "summary": summary,
            "real_provider_pass": False,
            "production_mine_badge": False,
            "live_emission_authority": False,
        }
    )


# ---------------------------------------------------------------------------
# VAL-TEEREQ-004: Lium adapter cannot produce REAL-PROVIDER PASS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # bare separated external config
        {"lium_ready": True},
        {"lium_ready": True, "allow_network": True},
        {"lium_ready": True, "provider_contract": dict(_COMPLETE_CONTRACT)},
        {
            "lium_ready": True,
            "allow_network": True,
            "provider_contract": dict(_COMPLETE_CONTRACT),
        },
    ],
)
def test_lium_readiness_would_grant_real_provider_pass_false(kwargs: dict[str, Any]) -> None:
    auth = _authority()
    cfg = _cfg(auth, **kwargs)
    report = LiumAdapter().readiness_report(cfg)
    assert report.ready is False
    assert report.classification is TeeClassification.BLOCKED
    assert report.would_grant_real_provider_pass is False
    assert report.max_effective_tier == 0
    assert real_provider_pass_is_possible(cfg, TeeProviderKind.LIUM) is False
    assert real_provider_pass_is_possible(cfg, "lium") is False
    payload = report.as_dict()
    assert payload["would_grant_real_provider_pass"] is False
    assert payload["classification"] == TeeClassification.BLOCKED.value
    # Hard gates remain enumerated even when operator flips every available knob.
    hard = {item.item_id for item in report.checklist if item.item_id in HARD_GATE_ITEMS}
    assert hard == set(HARD_GATE_ITEMS)


def test_lium_with_pod_metadata_and_nonempty_quotes_stays_blocked() -> None:
    """Creds + inventory + pod metadata + opaque quote/JWT still never REAL PASS."""
    auth = _authority()
    cfg = _cfg(
        auth,
        lium_ready=True,
        allow_network=True,
        provider_contract=dict(_COMPLETE_CONTRACT),
    )
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    # Opaque non-empty claim keys alone (field presence is never attestation).
    # Extra unknown keys fail closed at parse; curated opaque shape still cannot PASS.
    opaque = {
        "version": 1,
        "provider": "lium",
        "evidence_type": "prism.tee.v1",
        "tdx_quote_b64": "QUOTE" + "A" * 64,
        "gpu_eat_jwt": "a.b.c",
    }
    # Locally-minted crypto labeled lium also stays blocked (provider scope wins).
    crypto_as_lium = _fixture(auth, provider="lium")
    for attestation, expect_blocked in ((opaque, False), (crypto_as_lium, True)):
        decision = asyncio.run(
            verifier.verify_proof(
                _proof(attestation, provider="lium"),
                work_unit_id=UNIT,
                expected_nonce=NONCE,
            )
        )
        assert decision.accepted is False
        # Parse failures are FAIL; readiness-blocked provider paths are BLOCKED.
        # Neither is REAL-PROVIDER PASS and neither elevates.
        assert decision.classification in {
            TeeClassification.BLOCKED,
            TeeClassification.FAIL,
        }
        if expect_blocked:
            assert decision.classification is TeeClassification.BLOCKED
            assert decision.metadata.get("would_grant_real_provider_pass") is False
        assert decision.classification is not TeeClassification.REAL_PROVIDER_PASS
        assert decision.effective_tier == 0
        surface = decision_public_surface(decision)
        assert surface["real_provider_pass"] is False
        assert surface["classification"] != REAL_PROVIDER_PASS_LABEL


def test_lium_deploy_smoke_labels_never_claim_real_pass() -> None:
    labels_ok = smoke_deploy_labels(deploy_ok=True)
    labels_fail = smoke_deploy_labels(deploy_ok=False)
    assert labels_ok["deploy_smoke"] == "DEPLOY SMOKE PASS"
    assert labels_fail["deploy_smoke"] == "DEPLOY SMOKE FAIL"
    for labels in (labels_ok, labels_fail):
        assert labels["real_provider_tee"] == "BLOCKED"
        assert labels["real_provider_pass"] == "BLOCKED"
        assert REAL_PROVIDER_PASS_LABEL not in labels["deploy_smoke"]
        # Dual classification language must remain independent.
        assert "DEPLOY SMOKE" in labels["deploy_smoke"]


# ---------------------------------------------------------------------------
# VAL-TEEREQ-005: Targon path remains future-blocked for REAL-PROVIDER PASS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"targon_ready": True},
        {"targon_ready": True, "provider_contract": dict(_COMPLETE_CONTRACT)},
        {
            "targon_ready": True,
            "allow_network": True,
            "provider_contract": dict(_COMPLETE_CONTRACT),
        },
    ],
)
def test_targon_future_blocked_for_real_provider_pass(kwargs: dict[str, Any]) -> None:
    auth = _authority()
    cfg = _cfg(auth, **kwargs)
    report = TargonAdapter().readiness_report(cfg)
    assert report.ready is False
    assert report.classification is TeeClassification.BLOCKED
    assert report.would_grant_real_provider_pass is False
    assert report.max_effective_tier == 0
    assert real_provider_pass_is_possible(cfg, "targon") is False
    assert report.reason in {
        TeeReasonCode.PROVIDER_FUTURE_BLOCKED,
        TeeReasonCode.ADAPTER_NOT_READY,
    }
    assert report.metadata.get("future_or_blocked") is True
    assert report.metadata.get("speculative_endpoints_attempted", 0) == 0


def test_targon_evidence_does_not_unlock_production_score_gate() -> None:
    from prism_challenge.tee.score_gate import decision_authorizes_score

    auth = _authority()
    cfg = _cfg(
        auth, targon_ready=True, provider_contract=dict(_COMPLETE_CONTRACT), mode="production"
    )
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    decision = asyncio.run(
        verifier.verify_proof(
            _proof(_fixture(auth, provider="targon"), provider="targon"),
            work_unit_id=UNIT,
            expected_nonce=NONCE,
        )
    )
    assert decision.accepted is False
    assert decision.classification is TeeClassification.BLOCKED
    # Production TEE-required mode must not authorize scoring from Targon block.
    authz = decision_authorizes_score(decision, require_for_score=True, mode="production")
    assert authz.authorized is False


# ---------------------------------------------------------------------------
# VAL-TEEREQ-010: cannot smuggle LOCAL-FIXTURE as REAL-PROVIDER
# ---------------------------------------------------------------------------


def test_smuggle_local_as_real_provider_rejected() -> None:
    with pytest.raises(ClassificationHonestyError):
        assert_honest_classification_surface(
            {
                "classification": REAL_PROVIDER_PASS_LABEL,
                "validation_source": "local_fixture",
                "provider": "local_fixture",
                "summary": "TEE REAL-PROVIDER PASS · source=local_fixture",
                "real_provider_pass": True,
            }
        )


def test_smuggle_local_fixture_pass_with_mine_badge_rejected() -> None:
    with pytest.raises(ClassificationHonestyError):
        assert_honest_classification_surface(
            {
                "classification": LOCAL_FIXTURE_PASS_LABEL,
                "validation_source": "local_fixture",
                "provider": "local_fixture",
                "summary": f"TEE {LOCAL_FIXTURE_PASS_LABEL} · source=local_fixture",
                "real_provider_pass": False,
                "production_mine_badge": True,
            }
        )


def test_smuggle_local_with_live_emission_authority_rejected() -> None:
    with pytest.raises(ClassificationHonestyError):
        assert_honest_classification_surface(
            {
                "classification": LOCAL_FIXTURE_PASS_LABEL,
                "validation_source": "local_fixture",
                "provider": "local_fixture",
                "summary": f"TEE {LOCAL_FIXTURE_PASS_LABEL} · source=local_fixture",
                "live_emission_authority": True,
            }
        )


def test_coerce_real_provider_label_on_local_fixture() -> None:
    coerced = coerce_accepted_fixture_classification(
        TeeClassification.REAL_PROVIDER_PASS,
        provider=TeeProviderKind.LOCAL_FIXTURE,
        reason=TeeReasonCode.ACCEPTED_LOCAL_FIXTURE,
    )
    assert coerced is TeeClassification.LOCAL_FIXTURE_PASS
    with pytest.raises(ClassificationHonestyError):
        assert_not_real_provider_pass(
            classification=TeeClassification.REAL_PROVIDER_PASS,
            provider=TeeProviderKind.LOCAL_FIXTURE,
        )


def test_public_surface_coerces_mislabeled_local_decision() -> None:
    # Simulate a corrupted decision object that mis-tags classification.
    bad = TeeDecision(
        accepted=True,
        classification=TeeClassification.REAL_PROVIDER_PASS,
        reason=TeeReasonCode.ACCEPTED_LOCAL_FIXTURE,
        provider=TeeProviderKind.LOCAL_FIXTURE,
        effective_tier=2,
        detail="should never happen",
        metadata={"validation_source": "local_fixture"},
    )
    surface = decision_public_surface(bad)
    assert surface["classification"] == LOCAL_FIXTURE_PASS_LABEL
    assert surface["real_provider_pass"] is False
    assert surface["local_fixture_pass"] is True
    assert REAL_PROVIDER_PASS_LABEL not in surface["summary"]
    assert "local_fixture" in surface["summary"]


def test_lab_cli_string_constraints_on_honest_summary() -> None:
    auth = _authority()
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    decision = asyncio.run(
        verifier.verify_proof(
            _proof(_fixture(auth)),
            work_unit_id=UNIT,
            expected_nonce=NONCE,
        )
    )
    summary = human_summary_line(decision)
    # Substring constraints from VAL-TEEREQ-010.
    assert "LOCAL-FIXTURE" in summary or "local_fixture" in summary
    assert REAL_PROVIDER_PASS_LABEL not in summary
    assert "production mine" not in summary.lower()
    assert "live-emission" not in summary.lower()


# ---------------------------------------------------------------------------
# VAL-TEEREQ-012: residual removed LLM env still fail closed at boot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_name",
    sorted(
        {
            "PRISM_LLM_GATEWAY_URL",
            "PRISM_GATEWAY_TOKEN",
            "PRISM_LLM_REVIEW_ENABLED",
            "PRISM_COMPONENT_AGENT_ENABLED",
            "PRISM_COMPONENT_HOLD_LOW_CONFIDENCE",
            "BASE_LLM_GATEWAY_URL",
            "BASE_GATEWAY_TOKEN",
        }
        & set(REMOVED_LLM_ENV_NAMES)
        | {
            "PRISM_LLM_GATEWAY_URL",
            "PRISM_GATEWAY_TOKEN",
            "PRISM_LLM_REVIEW_ENABLED",
            "PRISM_COMPONENT_AGENT_ENABLED",
            "PRISM_COMPONENT_HOLD_LOW_CONFIDENCE",
            "BASE_LLM_GATEWAY_URL",
            "BASE_GATEWAY_TOKEN",
        }
    ),
)
def test_residual_removed_llm_env_boot_reject(
    monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    # Clear the whole residual set first so only the targeted key triggers rejection.
    for name in REMOVED_LLM_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BASE_LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("BASE_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("BASE_GATEWAY_TOKEN_FILE", raising=False)
    monkeypatch.setenv(env_name, "true" if "URL" not in env_name else "http://gateway/llm/v1")
    with pytest.raises(ValueError, match="removed Prism LLM"):
        PrismSettings(shared_token="test-token")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"llm_review_enabled": True},
        {"llm_gateway_url": "http://gateway/llm/v1"},
        {"llm_gateway_token": "tok"},
        {"component_agent_enabled": True},
        {"component_agent_model": "gpt-4o"},
    ],
)
def test_residual_removed_llm_kwargs_boot_reject(kwargs: dict[str, Any]) -> None:
    assert set(kwargs).issubset(REMOVED_LLM_SETTING_NAMES)
    with pytest.raises(ValueError, match="removed Prism LLM"):
        PrismSettings(shared_token="test-token", **kwargs)


def test_clean_boot_without_residual_llm_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(REMOVED_LLM_ENV_NAMES) + [
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "BASE_GATEWAY_TOKEN_FILE",
    ]:
        monkeypatch.delenv(name, raising=False)
    settings = PrismSettings(shared_token="test-token", database_path="/tmp/prism-class.sqlite3")
    assert settings.tee is not None
    assert settings.shared_token == "test-token"
    # No gateway fields remain on the settings model.
    assert "llm_gateway_url" not in PrismSettings.model_fields
    assert "llm_review_enabled" not in PrismSettings.model_fields


# ---------------------------------------------------------------------------
# Hard-gate matrix still in force (document + check)
# ---------------------------------------------------------------------------


def test_hard_gate_items_still_in_force_and_complete() -> None:
    # Exactly the authoritatively documented dependency checklist (11 items).
    assert len(HARD_GATE_ITEMS) == 11
    required = {
        "authoritative_evidence_endpoint",
        "authoritative_evidence_format",
        "authoritative_issuer_audience",
        "authoritative_trust_roots",
        "freshness_and_clock_policy",
        "nonce_semantics",
        "digest_pinned_public_worker_image",
        "measurement_policy",
        "gpu_claim_policy",
        "cross_binding_semantics",
        "real_workload_evidence_artifact",
    }
    assert set(HARD_GATE_ITEMS) == required
    # Provider readiness evaluation continues to refuse REAL-PROVIDER PASS.
    cfg = _cfg(lium_ready=True, targon_ready=True, provider_contract=dict(_COMPLETE_CONTRACT))
    for provider in (TeeProviderKind.LIUM, TeeProviderKind.TARGON):
        report = evaluate_provider_readiness(provider, cfg)
        assert report.would_grant_real_provider_pass is False
        assert report.ready is False
        assert real_provider_pass_is_possible(cfg, provider) is False


def test_provider_block_matrix_combined() -> None:
    """End-to-end classification + provider block matrix for lab honesty."""
    auth = _authority()
    matrix: list[dict[str, Any]] = []

    # Local fixture accept case
    cfg = _cfg(auth)
    store = InMemoryNonceStore()
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: AUTHORITY_NOW)
    local = asyncio.run(
        verifier.verify_proof(_proof(_fixture(auth)), work_unit_id=UNIT, expected_nonce=NONCE)
    )
    matrix.append(
        {
            "path": "local_fixture",
            "classification": local.classification.value,
            "accepted": local.accepted,
            "real_provider_pass": False,
        }
    )

    # Lium blocked cases (creds / ready / contract / opaque meta)
    for label, overrides in (
        ("lium_bare", {}),
        (
            "lium_ready_and_contract",
            {"lium_ready": True, "provider_contract": dict(_COMPLETE_CONTRACT)},
        ),
    ):
        report = LiumAdapter().readiness_report(_cfg(auth, **overrides))
        matrix.append(
            {
                "path": label,
                "classification": report.classification.value,
                "accepted": report.ready,
                "would_grant_real_provider_pass": report.would_grant_real_provider_pass,
            }
        )

    # Targon future blocked
    t_report = TargonAdapter().readiness_report(
        _cfg(auth, targon_ready=True, provider_contract=dict(_COMPLETE_CONTRACT))
    )
    matrix.append(
        {
            "path": "targon_future",
            "classification": t_report.classification.value,
            "accepted": t_report.ready,
            "would_grant_real_provider_pass": t_report.would_grant_real_provider_pass,
        }
    )

    # Assertions across the matrix
    by_path = {row["path"]: row for row in matrix}
    assert by_path["local_fixture"]["classification"] == LOCAL_FIXTURE_PASS_LABEL
    assert by_path["local_fixture"]["accepted"] is True
    assert by_path["local_fixture"]["real_provider_pass"] is False
    for label in ("lium_bare", "lium_ready_and_contract", "targon_future"):
        assert by_path[label]["classification"] == TeeClassification.BLOCKED.value
        assert by_path[label]["accepted"] is False
        assert by_path[label]["would_grant_real_provider_pass"] is False
    # No path may claim REAL-PROVIDER PASS.
    assert all(row["classification"] != REAL_PROVIDER_PASS_LABEL for row in matrix)


def test_assert_not_real_provider_pass_blocks_lium_true_flag() -> None:
    with pytest.raises(ClassificationHonestyError):
        assert_not_real_provider_pass(
            classification=TeeClassification.BLOCKED,
            provider=TeeProviderKind.LIUM,
            would_grant_real_provider_pass=True,
        )


def test_credentials_presence_in_env_never_unlocks_real_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate credentials present by name only (values redacted / fake).
    monkeypatch.setenv("LIUM_API_KEY", "redacted-not-a-real-key")
    monkeypatch.setenv("TARGON_API_KEY", "redacted-not-a-real-key")
    assert os.environ.get("LIUM_API_KEY") is not None
    cfg = _cfg(
        lium_ready=True,
        targon_ready=True,
        allow_network=True,
        provider_contract=dict(_COMPLETE_CONTRACT),
    )
    assert real_provider_pass_is_possible(cfg, "lium") is False
    assert real_provider_pass_is_possible(cfg, "targon") is False
    assert LiumAdapter().readiness_report(cfg).would_grant_real_provider_pass is False
    assert TargonAdapter().readiness_report(cfg).would_grant_real_provider_pass is False
