"""Durable TEE nonce ledger survives store re-open (restart-safe)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature
from prism_challenge.tee.crypto_local import DEFAULT_MEASUREMENTS, LocalFixtureAuthority
from prism_challenge.tee.nonce_store import DurableNonceStore
from prism_challenge.tee.verifier import TeeVerifier, TeeVerifierConfig

IMAGE = "sha256:" + "ef" * 32
MANIFEST = "ab" * 32
UNIT = "durable-unit-1"
WORKER_PUB = "5DurableWorker"
NONCE = "durable-nonce-1"


def test_durable_nonce_survives_reopen(tmp_path: Path) -> None:
    auth = LocalFixtureAuthority.generate(now=datetime(2026, 7, 12, 12, 0, tzinfo=UTC))
    db = tmp_path / "prism.sqlite3"
    store = DurableNonceStore(db)
    asyncio.run(store.ensure_schema())

    cfg = TeeVerifierConfig(
        enabled=True,
        mode="local_fixture",
        expected_issuer=auth.issuer,
        expected_audience="prism.tee.verify",
        tdx_trust_roots_pem=(auth.ca_pem(),),
        gpu_trusted_keys_pem={auth.gpu_kid: auth.gpu_public_pem()},
        expected_image_digest=IMAGE,
        allowed_measurements=dict(DEFAULT_MEASUREMENTS),
        challenge_slug="prism",
        workload_id="prism",
        workload_version="1",
    )
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    verifier = TeeVerifier(cfg, nonce_store=store, now_fn=lambda: now)
    att = auth.build_attestation(
        nonce=NONCE,
        work_unit_id=UNIT,
        submission_id=UNIT,
        image_digest=IMAGE,
        workload_id="prism",
        workload_version="1",
        challenge_slug="prism",
        manifest_sha256=MANIFEST,
        worker_pubkey=WORKER_PUB,
        now=now,
    )
    proof = ExecutionProof(
        version=1,
        tier=2,
        manifest_sha256=MANIFEST,
        image_digest=IMAGE,
        provider=ProviderInfo(name="local_fixture", pod_id="p"),
        worker_signature=WorkerSignature(worker_pubkey=WORKER_PUB, sig="0xab"),
        attestation=att,
    )
    first = asyncio.run(verifier.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert first.accepted is True

    # Simulate restart: new store instance on same path.
    store2 = DurableNonceStore(db)
    verifier2 = TeeVerifier(cfg, nonce_store=store2, now_fn=lambda: now)
    second = asyncio.run(verifier2.verify_proof(proof, work_unit_id=UNIT, expected_nonce=NONCE))
    assert second.accepted is False
    assert second.reason.value == "nonce_replay"
