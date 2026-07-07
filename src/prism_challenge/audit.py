"""Tier verification + probabilistic audit sampling (architecture.md 3.4/3.5).

The proof tier a worker CLAIMS is an input to the audit rate, so it must never be trusted verbatim:
a claim is only worth its tier if the backing metadata is verifiable. :func:`effective_tier` maps a
proof's CLAIMED tier onto the EFFECTIVE tier the audit scheduler actually consumes:

* claimed tier 2 -> effective 2 iff a populated attestation payload is present, else effective 0
  (a tier-2 claim with a null/empty attestation is downgraded straight to 0, never to 1);
* claimed tier 1 -> effective 1 iff the proof's ``image_digest`` equals the configured pinned
  evaluator/worker digest, else effective 0;
* claimed tier 0 (or any unknown tier) -> effective 0.

:class:`AuditSampler` then samples finalized results at the per-tier rate of their EFFECTIVE tier.
Sampling is deterministic in the sampler's ``seed`` and the per-result key, so the same seed
reproduces the same sample set (VAL-PRISM-011) and a downgraded claim is audited at its lower
(effective) rate rather than the rate it dishonestly claimed (VAL-PRISM-019).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .config import WorkerPlaneConfig
from .proof import ExecutionProof, has_attestation

#: The three proof tiers the audit rate is keyed on (architecture.md 3.4).
TIER_0, TIER_1, TIER_2 = 0, 1, 2


def effective_tier(proof: ExecutionProof, *, pinned_image_digest: str | None = None) -> int:
    """Return the VERIFIED tier for ``proof`` (never higher than what the backing metadata proves).

    A claimed tier is honoured only when its backing is verifiable; otherwise it is downgraded to
    tier 0 (architecture.md 3.4; VAL-PRISM-019). A claimed tier 2 with an unverifiable attestation
    downgrades to 0 (not 1), even if it also carries a matching image digest.
    """

    claimed = int(proof.tier)
    if claimed <= TIER_0:
        return TIER_0
    if claimed == TIER_1:
        matches = bool(pinned_image_digest) and proof.image_digest == pinned_image_digest
        return TIER_1 if matches else TIER_0
    if claimed == TIER_2:
        return TIER_2 if has_attestation(proof.attestation) else TIER_0
    # An out-of-range/unknown claimed tier is not verifiable -> conservative tier 0.
    return TIER_0


def is_tier_downgraded(proof: ExecutionProof, *, pinned_image_digest: str | None = None) -> bool:
    """Whether ``proof``'s claimed tier is higher than its verified effective tier."""

    return int(proof.tier) != effective_tier(proof, pinned_image_digest=pinned_image_digest)


@dataclass(frozen=True)
class AuditDecision:
    """The outcome of applying the audit sampler to one finalized result."""

    work_unit_id: str
    claimed_tier: int
    effective_tier: int
    sampled: bool

    @property
    def downgraded(self) -> bool:
        return self.claimed_tier != self.effective_tier


@dataclass(frozen=True)
class AuditSampler:
    """Deterministic, per-tier probabilistic sampler over finalized results (architecture.md 3.4).

    The sampled fraction of each tier converges to its configured rate; a rate of ``0.0`` yields
    exactly zero samples for that tier and ``>= 1.0`` samples every one. Sampling is a pure function
    of ``seed`` and the per-result key, so it is reproducible and order-insensitive.
    """

    audit_rate_tier0: float = 0.10
    audit_rate_tier1: float = 0.05
    audit_rate_tier2: float = 0.02
    seed: int = 0

    def rate_for_tier(self, tier: int) -> float:
        """The configured audit rate for an EFFECTIVE tier (unknown tiers fall back to tier 0)."""

        if tier == TIER_2:
            return self.audit_rate_tier2
        if tier == TIER_1:
            return self.audit_rate_tier1
        return self.audit_rate_tier0

    def should_sample(self, *, work_unit_id: str, effective_tier: int) -> bool:
        """Whether a result of ``effective_tier`` is sampled for audit (deterministic in seed)."""

        rate = self.rate_for_tier(effective_tier)
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return self._uniform(work_unit_id) < rate

    def decide(
        self,
        *,
        work_unit_id: str,
        proof: ExecutionProof,
        pinned_image_digest: str | None = None,
    ) -> AuditDecision:
        """Verify ``proof``'s tier and decide whether it is sampled at its EFFECTIVE rate."""

        tier = effective_tier(proof, pinned_image_digest=pinned_image_digest)
        return AuditDecision(
            work_unit_id=work_unit_id,
            claimed_tier=int(proof.tier),
            effective_tier=tier,
            sampled=self.should_sample(work_unit_id=work_unit_id, effective_tier=tier),
        )

    def _uniform(self, key: str) -> float:
        """A deterministic uniform draw in ``[0, 1)`` keyed on ``(seed, key)``."""

        digest = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)


def audit_sampler_from_config(worker_plane: WorkerPlaneConfig, *, seed: int = 0) -> AuditSampler:
    """Build an :class:`AuditSampler` from the prism ``worker_plane`` audit-rate config."""

    return AuditSampler(
        audit_rate_tier0=worker_plane.audit_rate_tier0,
        audit_rate_tier1=worker_plane.audit_rate_tier1,
        audit_rate_tier2=worker_plane.audit_rate_tier2,
        seed=seed,
    )


__all__ = [
    "TIER_0",
    "TIER_1",
    "TIER_2",
    "AuditDecision",
    "AuditSampler",
    "audit_sampler_from_config",
    "effective_tier",
    "is_tier_downgraded",
]
