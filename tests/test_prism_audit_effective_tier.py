"""Tier verification + effective-tier audit sampling (VAL-PRISM-019).

A worker's CLAIMED proof tier is only honoured when its backing metadata is verifiable; otherwise
the EFFECTIVE tier used for audit sampling is downgraded. These tests pin the downgrade rules and
show that seeded sampling statistics follow the EFFECTIVE tier, not the dishonest claim. Offline,
pure functions (no GPU, no DB).
"""

from __future__ import annotations

from prism_challenge.audit import (
    AuditSampler,
    audit_sampler_from_config,
    effective_tier,
    is_tier_downgraded,
)
from prism_challenge.config import WorkerPlaneConfig
from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature

PINNED = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def _proof(
    *,
    tier: int,
    image_digest: str | None = None,
    attestation: dict[str, object] | None = None,
) -> ExecutionProof:
    return ExecutionProof(
        version=1,
        tier=tier,
        manifest_sha256="c" * 64,
        image_digest=image_digest,
        provider=ProviderInfo(name="lium", pod_id="pod-1"),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
        attestation=attestation,
    )


# --- Downgrade rules (VAL-PRISM-019 a/b) --------------------------------------------------------


def test_tier2_claim_downgrades_without_valid_attestation() -> None:
    valid = _proof(tier=2, attestation={"tdx_quote_b64": "abc"})
    null_attest = _proof(tier=2, attestation=None)
    empty_attest = _proof(tier=2, attestation={"unrelated": "x"})

    assert effective_tier(valid, pinned_image_digest=PINNED) == 2
    assert is_tier_downgraded(valid, pinned_image_digest=PINNED) is False
    # A tier-2 claim with a null / keyless attestation is effective tier 0 (never 2, never 1) --
    # even when it also carries a matching digest.
    assert effective_tier(null_attest, pinned_image_digest=PINNED) == 0
    assert effective_tier(empty_attest, pinned_image_digest=PINNED) == 0
    with_digest = _proof(tier=2, image_digest=PINNED, attestation=None)
    assert effective_tier(with_digest, pinned_image_digest=PINNED) == 0
    assert is_tier_downgraded(with_digest, pinned_image_digest=PINNED) is True


def test_tier1_claim_requires_matching_pinned_digest() -> None:
    matching = _proof(tier=1, image_digest=PINNED)
    mismatched = _proof(tier=1, image_digest=OTHER)
    missing = _proof(tier=1, image_digest=None)

    assert effective_tier(matching, pinned_image_digest=PINNED) == 1
    assert is_tier_downgraded(matching, pinned_image_digest=PINNED) is False
    assert effective_tier(mismatched, pinned_image_digest=PINNED) == 0
    assert effective_tier(missing, pinned_image_digest=PINNED) == 0
    # With no pinned digest configured, no tier-1 claim is verifiable.
    assert effective_tier(matching, pinned_image_digest=None) == 0


def test_tier0_claim_stays_tier0() -> None:
    assert effective_tier(_proof(tier=0), pinned_image_digest=PINNED) == 0
    assert is_tier_downgraded(_proof(tier=0), pinned_image_digest=PINNED) is False


# --- Sampling follows the EFFECTIVE tier (VAL-PRISM-019 statistical) -----------------------------


def _sampled_fraction(sampler: AuditSampler, proof: ExecutionProof, n: int) -> float:
    hits = sum(
        sampler.decide(
            work_unit_id=f"{proof.tier}-{i}", proof=proof, pinned_image_digest=PINNED
        ).sampled
        for i in range(n)
    )
    return hits / n


def test_sampling_statistics_follow_effective_not_claimed_tier() -> None:
    sampler = AuditSampler(
        audit_rate_tier0=0.10, audit_rate_tier1=0.05, audit_rate_tier2=0.02, seed=1234
    )
    n = 6000
    # 4-sigma binomial bound around each configured rate for this N.
    def _bound(p: float) -> float:
        return 4.0 * (p * (1.0 - p) / n) ** 0.5

    honest_t2 = _proof(tier=2, attestation={"gpu_eat_jwt": "jwt"})
    honest_t1 = _proof(tier=1, image_digest=PINNED)
    fake_t2 = _proof(tier=2, attestation=None)  # effective 0
    fake_t1 = _proof(tier=1, image_digest=OTHER)  # effective 0

    # Honest claims are sampled at their own tier's rate.
    assert abs(_sampled_fraction(sampler, honest_t2, n) - 0.02) < _bound(0.02)
    assert abs(_sampled_fraction(sampler, honest_t1, n) - 0.05) < _bound(0.05)
    # Unverifiable claims are sampled at the EFFECTIVE (tier-0) rate, NOT the lower claimed rate.
    assert abs(_sampled_fraction(sampler, fake_t2, n) - 0.10) < _bound(0.10)
    assert abs(_sampled_fraction(sampler, fake_t1, n) - 0.10) < _bound(0.10)


def test_zero_rate_never_samples_and_seed_is_reproducible() -> None:
    zero = AuditSampler(audit_rate_tier0=0.0, audit_rate_tier1=0.0, audit_rate_tier2=0.0, seed=7)
    assert all(
        not zero.should_sample(work_unit_id=f"u{i}", effective_tier=t)
        for i in range(500)
        for t in (0, 1, 2)
    )

    a = AuditSampler(audit_rate_tier0=0.10, seed=99)
    b = AuditSampler(audit_rate_tier0=0.10, seed=99)
    c = AuditSampler(audit_rate_tier0=0.10, seed=100)
    ids = [f"unit-{i}" for i in range(300)]
    sample_a = [a.should_sample(work_unit_id=i, effective_tier=0) for i in ids]
    sample_b = [b.should_sample(work_unit_id=i, effective_tier=0) for i in ids]
    sample_c = [c.should_sample(work_unit_id=i, effective_tier=0) for i in ids]
    assert sample_a == sample_b  # same seed => identical sample set
    assert sample_a != sample_c  # a different seed shifts the sample set


def test_sampler_from_config_uses_configured_rates() -> None:
    config = WorkerPlaneConfig(
        enabled=True, audit_rate_tier0=0.4, audit_rate_tier1=0.3, audit_rate_tier2=0.2
    )
    sampler = audit_sampler_from_config(config, seed=5)
    assert sampler.rate_for_tier(0) == 0.4
    assert sampler.rate_for_tier(1) == 0.3
    assert sampler.rate_for_tier(2) == 0.2
