"""TEE-required production scoring gates (VAL-TEEREQ-001/002/006/007/008/009/011).

When ``tee.require_for_score`` is enabled, missing / forged / incomplete / watchtower-only
/ tier0/tier1 / legacy broker evidence cannot finalize a production score, architecture
family row, or emission-ready weight map. Deterministic admission is unchanged. No live
Swarm, set_weights, or REAL-PROVIDER invented contracts.
"""

from __future__ import annotations

import base64
import io
import math
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, TeeConfig, WorkerPlaneConfig
from prism_challenge.ingestion import ResultIngestionError, ingest_work_unit_result
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    ProviderInfo,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.tee import (
    InMemoryNonceStore,
    TeeClassification,
    TeeReasonCode,
    TeeVerifier,
    TeeVerifierConfig,
    decision_authorizes_score,
    evaluate_watchtower_digest,
)
from prism_challenge.tee.crypto_local import DEFAULT_MEASUREMENTS, LocalFixtureAuthority
from prism_challenge.tee.score_gate import (
    SUBREASON_CONFIG,
    SUBREASON_LOW_TIER,
    SUBREASON_MISSING_DECISION,
    SUBREASON_WATCHTOWER,
    TEE_REQUIRED_REASON,
)
from prism_challenge.tee.types import TeeDecision, TeeProviderKind, fail_decision
from prism_challenge.weights import get_weights

WORKER_KEY = "//WorkerTeeRequired"
IMAGE = "sha256:" + "ab" * 32
MANIFEST_HEX = "cd" * 32
UNIT = "unit-tee-req-1"
NONCE = "nonce-tee-required-001"

TINY_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

TINY_TRAIN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""


def _bundle() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _manifest(marker: str = "v2") -> dict[str, Any]:
    covered_bytes = 4096
    online_loss = [10.0, 6.0, 3.0, 2.0]
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": online_loss,
            "sum_neg_log_likelihood_nats": 900.0,
            "covered_bytes": covered_bytes,
            "predicted_tokens": 96,
            "step0_loss": online_loss[0],
            "consumed_batches": len(online_loss),
            "random_init_baseline_nats": math.log(50257),
            "prequential_bpb": 1.23,
            "marker": marker,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
    }


def _settings(
    tmp_path: Path,
    *,
    require_for_score: bool = True,
    mode: str = "local_fixture",
    worker_plane_enabled: bool = True,
    tee_kwargs: dict[str, Any] | None = None,
) -> PrismSettings:
    tee = TeeConfig(
        enabled=True,
        mode=mode,  # type: ignore[arg-type]
        require_for_score=require_for_score,
        expected_provider="local_fixture",
        expected_issuer="prism-local-fixture",
        expected_audience="prism.tee.verify",
        **(tee_kwargs or {}),
    )
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'tee-req.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
        worker_plane=WorkerPlaneConfig(
            enabled=worker_plane_enabled,
            signing_key=WORKER_KEY,
            pinned_image_digest=IMAGE,
        ),
        tee=tee,
    )


def _authority() -> LocalFixtureAuthority:
    return LocalFixtureAuthority.generate(now=datetime(2026, 7, 13, 12, 0, tzinfo=UTC))


def _cfg(auth: LocalFixtureAuthority, **overrides: Any) -> TeeVerifierConfig:
    base: dict[str, Any] = dict(
        enabled=True,
        mode="local_fixture",
        require_for_score=True,
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


def _score(db_path: Path, submission_id: str) -> float | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else float(row[0])


def _arch_family_count(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM architecture_families").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _submission_status(db_path: Path, submission_id: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT status FROM submissions WHERE id=?", (submission_id,)).fetchone()
    finally:
        conn.close()
    return None if row is None else str(row[0])


async def _seed(app, hotkey: str = "hk-tee-req") -> str:
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_bundle(), filename="project.zip")
    )
    return sub.id


# ---------------------------------------------------------------------------
# Pure gate unit coverage
# ---------------------------------------------------------------------------


def test_decision_authorizes_score_no_op_when_flag_off() -> None:
    auth = decision_authorizes_score(None, require_for_score=False)
    assert auth.authorized is True


def test_decision_authorizes_score_missing_decision() -> None:
    auth = decision_authorizes_score(None, require_for_score=True)
    assert auth.authorized is False
    assert auth.reason == TEE_REQUIRED_REASON
    assert auth.subreason == SUBREASON_MISSING_DECISION


def test_decision_authorizes_score_forged_fail() -> None:
    decision = fail_decision(
        reason=TeeReasonCode.TDX_SIGNATURE_INVALID,
        provider=TeeProviderKind.LOCAL_FIXTURE,
    )
    auth = decision_authorizes_score(decision, require_for_score=True)
    assert auth.authorized is False
    assert auth.reason == TEE_REQUIRED_REASON


def test_decision_authorizes_score_tier0_or_tier1_alone() -> None:
    for tier in (0, 1):
        decision = TeeDecision(
            accepted=True,
            classification=TeeClassification.LOCAL_FIXTURE_PASS,
            reason=TeeReasonCode.ACCEPTED_LOCAL_FIXTURE,
            provider=TeeProviderKind.LOCAL_FIXTURE,
            effective_tier=tier,
        )
        auth = decision_authorizes_score(decision, require_for_score=True, mode="local_fixture")
        assert auth.authorized is False
        assert auth.subreason == SUBREASON_LOW_TIER


def test_decision_authorizes_score_local_fixture_ok_in_fixture_mode() -> None:
    decision = TeeDecision(
        accepted=True,
        classification=TeeClassification.LOCAL_FIXTURE_PASS,
        reason=TeeReasonCode.ACCEPTED_LOCAL_FIXTURE,
        provider=TeeProviderKind.LOCAL_FIXTURE,
        effective_tier=2,
    )
    auth = decision_authorizes_score(decision, require_for_score=True, mode="local_fixture")
    assert auth.authorized is True


def test_decision_authorizes_score_local_fixture_blocked_in_production_mode() -> None:
    decision = TeeDecision(
        accepted=True,
        classification=TeeClassification.LOCAL_FIXTURE_PASS,
        reason=TeeReasonCode.ACCEPTED_LOCAL_FIXTURE,
        provider=TeeProviderKind.LOCAL_FIXTURE,
        effective_tier=2,
    )
    auth = decision_authorizes_score(decision, require_for_score=True, mode="production")
    assert auth.authorized is False


def test_watchtower_match_never_authorizes_score() -> None:
    """VAL-TEEREQ-009: matching watchtower digest is METADATA_NOT_ATTESTATION only."""
    evaluation = evaluate_watchtower_digest(
        {
            "digest": IMAGE,
            "signature_valid": True,
            "signing_key_known": True,
            "timestamp": datetime(2026, 7, 13, 12, 0, tzinfo=UTC).isoformat(),
            "pod_id": "pod-1",
            "executor_id": "ex-1",
        },
        expected_image_digest=IMAGE,
        max_age_seconds=3_600,
        now=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    # Perfect digest match remains metadata-only: never LOCAL/REAL PASS, never tier 2.
    assert evaluation.reason is TeeReasonCode.METADATA_NOT_ATTESTATION
    assert evaluation.effective_tier <= 1
    # Even a forged "accepted" decision carrying METADATA_NOT_ATTESTATION fails the gate.
    decision = TeeDecision(
        accepted=True,
        classification=TeeClassification.LOCAL_FIXTURE_PASS,
        reason=TeeReasonCode.METADATA_NOT_ATTESTATION,
        provider=TeeProviderKind.LIUM,
        effective_tier=2,
        detail=evaluation.detail,
    )
    auth = decision_authorizes_score(decision, require_for_score=True, mode="local_fixture")
    assert auth.authorized is False
    assert auth.subreason == SUBREASON_WATCHTOWER


def test_incomplete_config_decision_fails_gate() -> None:
    decision = fail_decision(
        reason=TeeReasonCode.VERIFIER_MISCONFIGURED,
        classification=TeeClassification.BLOCKED,
        detail="expected_image_digest missing/invalid",
    )
    auth = decision_authorizes_score(decision, require_for_score=True)
    assert auth.authorized is False
    assert auth.subreason == SUBREASON_CONFIG


# ---------------------------------------------------------------------------
# Ingestion integration under TEE-required
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _unset_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIUM_API_KEY", raising=False)
    monkeypatch.delenv("TARGON_API_KEY", raising=False)


async def test_ingestion_missing_attestation_rejects_score(tmp_path: Path) -> None:
    """VAL-TEEREQ-001: missing/null attestation does not finalize score."""
    settings = _settings(tmp_path, require_for_score=True)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id=submission_id,
        tier=0,
    )
    result = {
        "executed": 1,
        PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
        MANIFEST_PAYLOAD_KEY: manifest,
        "tee_nonce": NONCE,
    }
    auth = LocalFixtureAuthority.generate()
    # Intentionally incomplete so verify_proof yields fail for missing att.
    verifier = TeeVerifier(_cfg(auth), nonce_store=InMemoryNonceStore())
    db_path = tmp_path / "tee-req.sqlite3"
    before = _arch_family_count(db_path)

    with pytest.raises(ResultIngestionError) as exc:
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref=submission_id,
            result=result,
            pinned_image_digest=IMAGE,
            tee_verifier=verifier,
            expected_tee_nonce=NONCE,
        )
    assert exc.value.reason == TEE_REQUIRED_REASON
    assert _score(db_path, submission_id) is None
    assert _arch_family_count(db_path) == before
    weights = await get_weights(app.state.repository, settings.epoch_seconds)
    assert weights == {}


async def test_ingestion_null_attestation_rejects_score(tmp_path: Path) -> None:
    """VAL-TEEREQ-001: explicit null attestation object yields no score."""
    settings = _settings(tmp_path, require_for_score=True)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    digest = compute_manifest_sha256(manifest)
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=digest,
        unit_id=submission_id,
        tier=1,
        image_digest=IMAGE,
        provider=ProviderInfo(name="lium", pod_id="pod-1", executor_id="ex-1"),
    )
    payload = proof.model_dump(mode="json")
    payload["attestation"] = None
    result = {
        PROOF_PAYLOAD_KEY: payload,
        MANIFEST_PAYLOAD_KEY: manifest,
        "tee_nonce": NONCE,
    }
    auth = _authority()
    verifier = TeeVerifier(_cfg(auth), nonce_store=InMemoryNonceStore())
    db_path = tmp_path / "tee-req.sqlite3"

    with pytest.raises(ResultIngestionError) as exc:
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref=submission_id,
            result=result,
            pinned_image_digest=IMAGE,
            tee_verifier=verifier,
            expected_tee_nonce=NONCE,
        )
    assert exc.value.reason == TEE_REQUIRED_REASON
    assert _score(db_path, submission_id) is None


async def test_ingestion_forged_evidence_yields_no_score(tmp_path: Path) -> None:
    """VAL-TEEREQ-002: forged TDX/GPU material never scores."""
    settings = _settings(tmp_path, require_for_score=True)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    signer = worker_signer_from_key(WORKER_KEY)
    auth = _authority()
    manifest = _manifest()
    digest = compute_manifest_sha256(manifest)
    forged = auth.build_attestation(
        nonce=NONCE,
        work_unit_id=submission_id,
        submission_id=submission_id,
        image_digest=IMAGE,
        workload_id="prism",
        workload_version="1",
        challenge_slug="prism",
        manifest_sha256=digest,
        worker_pubkey=signer.worker_pubkey,
        now=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    # Corrupt signature bytes to force crypto fail.
    forged["tdx_quote_b64"] = base64.b64encode(b"not-a-real-quote").decode("ascii")
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=digest,
        unit_id=submission_id,
        tier=2,
        attestation=forged,
        image_digest=IMAGE,
        provider=ProviderInfo(name="local_fixture", pod_id="pod-1", executor_id="ex-1"),
    )
    result = {
        PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
        MANIFEST_PAYLOAD_KEY: manifest,
        "tee_nonce": NONCE,
    }
    verifier = TeeVerifier(_cfg(auth), nonce_store=InMemoryNonceStore())
    db_path = tmp_path / "tee-req.sqlite3"

    with pytest.raises(ResultIngestionError) as exc:
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref=submission_id,
            result=result,
            pinned_image_digest=IMAGE,
            tee_verifier=verifier,
            expected_tee_nonce=NONCE,
        )
    assert exc.value.reason == TEE_REQUIRED_REASON
    assert _score(db_path, submission_id) is None


async def test_incomplete_config_ablation_rejects_score(tmp_path: Path) -> None:
    """VAL-TEEREQ-006: missing trust roots / measurements / image pin fail closed."""
    settings = _settings(tmp_path, require_for_score=True)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    signer = worker_signer_from_key(WORKER_KEY)
    auth = _authority()
    related_digest = compute_manifest_sha256(_manifest())
    good = auth.build_attestation(
        nonce=NONCE,
        work_unit_id=submission_id,
        submission_id=submission_id,
        image_digest=IMAGE,
        workload_id="prism",
        workload_version="1",
        challenge_slug="prism",
        manifest_sha256=related_digest,
        worker_pubkey=signer.worker_pubkey,
        now=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    ablations = {
        "no_roots": dict(tdx_trust_roots_pem=()),
        "no_gpu_keys": dict(gpu_trusted_keys_pem={}),
        "no_image": dict(expected_image_digest=None),
        "no_measurements": dict(allowed_measurements={}),
        "disabled": dict(enabled=False),
    }
    db_path = tmp_path / "tee-req.sqlite3"
    for label, overrides in ablations.items():
        sub_id = await _seed(app, hotkey=f"hk-{label}")
        manifest = _manifest(marker=label)
        digest = compute_manifest_sha256(manifest)
        # Re-bind attestation digests to this unit.
        att = auth.build_attestation(
            nonce=f"{NONCE}-{label}",
            work_unit_id=sub_id,
            submission_id=sub_id,
            image_digest=IMAGE,
            workload_id="prism",
            workload_version="1",
            challenge_slug="prism",
            manifest_sha256=digest,
            worker_pubkey=signer.worker_pubkey,
            now=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
        )
        # Use the good template only to ensure evidence is well-formed; config kills acceptance.
        _ = good
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=digest,
            unit_id=sub_id,
            tier=2,
            attestation=att,
            image_digest=IMAGE,
            provider=ProviderInfo(name="local_fixture", pod_id="pod-1", executor_id="ex-1"),
        )
        result = {
            PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
            MANIFEST_PAYLOAD_KEY: manifest,
            "tee_nonce": f"{NONCE}-{label}",
        }
        verifier = TeeVerifier(
            _cfg(auth, **overrides),
            nonce_store=InMemoryNonceStore(),
        )
        with pytest.raises(ResultIngestionError) as exc:
            await ingest_work_unit_result(
                worker=app.state.worker,
                work_unit_id=sub_id,
                submission_ref=sub_id,
                result=result,
                pinned_image_digest=IMAGE,
                tee_verifier=verifier,
                expected_tee_nonce=f"{NONCE}-{label}",
            )
        assert exc.value.reason == TEE_REQUIRED_REASON, label
        assert _score(db_path, sub_id) is None, label


async def test_tier0_and_tier1_proofs_insufficient(tmp_path: Path) -> None:
    """VAL-TEEREQ-008: tier 0/1 alone cannot authorize scored finalization."""
    settings = _settings(tmp_path, require_for_score=True)
    app = create_app(settings)
    await app.state.database.init()
    auth = _authority()
    signer = worker_signer_from_key(WORKER_KEY)
    db_path = tmp_path / "tee-req.sqlite3"

    for tier in (0, 1):
        sub_id = await _seed(app, hotkey=f"hk-t{tier}")
        manifest = _manifest(marker=f"t{tier}")
        digest = compute_manifest_sha256(manifest)
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=digest,
            unit_id=sub_id,
            tier=tier,  # type: ignore[arg-type]
            image_digest=IMAGE if tier >= 1 else None,
            provider=(
                ProviderInfo(name="lium", pod_id="pod-1", executor_id="ex-1") if tier >= 1 else None
            ),
        )
        result = {
            PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
            MANIFEST_PAYLOAD_KEY: manifest,
            "tee_nonce": f"{NONCE}-t{tier}",
        }
        verifier = TeeVerifier(_cfg(auth), nonce_store=InMemoryNonceStore())
        with pytest.raises(ResultIngestionError) as exc:
            await ingest_work_unit_result(
                worker=app.state.worker,
                work_unit_id=sub_id,
                submission_ref=sub_id,
                result=result,
                pinned_image_digest=IMAGE,
                tee_verifier=verifier,
                expected_tee_nonce=f"{NONCE}-t{tier}",
            )
        assert exc.value.reason == TEE_REQUIRED_REASON
        assert _score(db_path, sub_id) is None


async def test_legacy_finalize_without_tee_fails_closed(tmp_path: Path) -> None:
    """VAL-TEEREQ-007: broker/base_gpu finalize McFarland without accepted TEE."""
    settings = _settings(tmp_path, require_for_score=True, worker_plane_enabled=False)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    db_path = tmp_path / "tee-req.sqlite3"
    before = _arch_family_count(db_path)
    # Direct path as if legacy re-exec completed with a scoreable manifest.
    anti = type("Anti", (), {"multiplier": 1.0})()
    await app.state.worker._finalize_container_score(  # noqa: SLF001 - intentional gate test
        submission_id=submission_id,
        arch_hash="family-" + "a" * 58,
        anti=anti,
        manifest=_manifest(marker="legacy"),
        hotkey="hk-legacy",
        skip_heldout=True,
        tee_score_authorized=False,
    )
    assert _score(db_path, submission_id) is None
    assert _arch_family_count(db_path) == before
    status = _submission_status(db_path, submission_id)
    assert status == "failed"
    weights = await get_weights(app.state.repository, settings.epoch_seconds)
    assert weights == {}


async def test_rejected_tee_does_not_insert_leaderboard_row(tmp_path: Path) -> None:
    """VAL-TEEREQ-011: rejected TEE unit does not create architecture family scoring row."""
    settings = _settings(tmp_path, require_for_score=True)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    auth = _authority()
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    digest = compute_manifest_sha256(manifest)
    # Incomplete attestation (missing quote/JWT components) fails verification → no score row.
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=digest,
        unit_id=submission_id,
        tier=2,
        image_digest=IMAGE,
        provider=ProviderInfo(name="local_fixture", pod_id="pod-1", executor_id="ex-1"),
        attestation={"type": "prism.tee.v1", "version": 1, "provider": "local_fixture"},
    )
    result = {
        PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
        MANIFEST_PAYLOAD_KEY: manifest,
        "tee_nonce": NONCE,
    }
    verifier = TeeVerifier(_cfg(auth), nonce_store=InMemoryNonceStore())
    db_path = tmp_path / "tee-req.sqlite3"
    before_families = _arch_family_count(db_path)

    with pytest.raises(ResultIngestionError):
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref=submission_id,
            result=result,
            pinned_image_digest=IMAGE,
            tee_verifier=verifier,
            expected_tee_nonce=NONCE,
        )
    assert _score(db_path, submission_id) is None
    assert _arch_family_count(db_path) == before_families
    # No positive emission-ready weights from miner self-report.
    assert await get_weights(app.state.repository, settings.epoch_seconds) == {}


async def test_accepted_local_fixture_can_score_when_required(tmp_path: Path) -> None:
    """Positive control: LOCAL-FIXTURE PASS in fixture mode may finalize under the gate."""
    settings = _settings(tmp_path, require_for_score=True, mode="local_fixture")
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    auth = _authority()
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest(marker="ok")
    digest = compute_manifest_sha256(manifest)
    att = auth.build_attestation(
        nonce=NONCE,
        work_unit_id=submission_id,
        submission_id=submission_id,
        image_digest=IMAGE,
        workload_id="prism",
        workload_version="1",
        challenge_slug="prism",
        manifest_sha256=digest,
        worker_pubkey=signer.worker_pubkey,
        now=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=digest,
        unit_id=submission_id,
        tier=2,
        attestation=att,
        image_digest=IMAGE,
        provider=ProviderInfo(name="local_fixture", pod_id="pod-1", executor_id="ex-1"),
    )
    result = {
        PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
        MANIFEST_PAYLOAD_KEY: manifest,
        "tee_nonce": NONCE,
    }
    verifier = TeeVerifier(
        _cfg(auth),
        nonce_store=InMemoryNonceStore(),
        now_fn=lambda: datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    db_path = tmp_path / "tee-req.sqlite3"
    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref=submission_id,
        result=result,
        pinned_image_digest=IMAGE,
        tee_verifier=verifier,
        expected_tee_nonce=NONCE,
    )
    assert outcome.status == "accepted"
    assert outcome.finalized is True
    assert _score(db_path, submission_id) is not None
    assert _score(db_path, submission_id) > 0.0
    # Sanity: local-fixture classification only (never REAL-PROVIDER).
    assert outcome.effective_tier == 2


async def test_gate_flag_off_allows_legacy_scored_finalize(tmp_path: Path) -> None:
    """When require_for_score is false, legacy finalize still writes scores (compat)."""
    settings = _settings(tmp_path, require_for_score=False, worker_plane_enabled=False)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    db_path = tmp_path / "tee-req.sqlite3"
    anti = type("Anti", (), {"multiplier": 1.0})()
    await app.state.worker._finalize_container_score(  # noqa: SLF001
        submission_id=submission_id,
        arch_hash="family-" + "b" * 58,
        anti=anti,
        manifest=_manifest(marker="flag-off"),
        hotkey="hk-flag-off",
        skip_heldout=True,
        tee_score_authorized=False,
    )
    assert _score(db_path, submission_id) is not None


async def test_verifier_called_when_attestation_absent_under_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with no attestation, the wired verifier runs so TEE-required sees a decision."""
    settings = _settings(tmp_path, require_for_score=True)
    app = create_app(settings)
    await app.state.database.init()
    submission_id = await _seed(app)
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id=submission_id,
        tier=0,
    )
    result = {
        PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
        MANIFEST_PAYLOAD_KEY: manifest,
    }
    called = {"n": 0}

    async def _verify(*_a: Any, **_k: Any) -> TeeDecision:
        called["n"] += 1
        return fail_decision(reason=TeeReasonCode.EVIDENCE_MISSING)

    mock_verifier = AsyncMock()
    mock_verifier.verify_proof = _verify
    mock_verifier.config = TeeVerifierConfig(require_for_score=True, mode="local_fixture")
    mock_verifier.nonce_store = None

    with pytest.raises(ResultIngestionError) as exc:
        await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref=submission_id,
            result=result,
            tee_verifier=mock_verifier,  # type: ignore[arg-type]
        )
    assert called["n"] == 1
    assert exc.value.reason == TEE_REQUIRED_REASON
