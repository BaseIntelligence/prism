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
from typing import Protocol

from .config import WorkerPlaneConfig
from .proof import ExecutionProof, has_attestation

#: The three proof tiers the audit rate is keyed on (architecture.md 3.4).
TIER_0, TIER_1, TIER_2 = 0, 1, 2

#: Prefix that makes an audit work-unit id DISTINCT from the primary unit id (== submission_id),
#: which is reserved for the submission's own evaluation unit (VAL-PRISM-012).
AUDIT_UNIT_PREFIX = "audit:"

#: Audit-unit lifecycle (architecture.md 3.5). ``pending`` units are the only ones exposed on the
#: coordination plane (pending-only listing semantics); the rest are terminal EXCEPT ``pending`` set
#: again on a bounded re-audit after an inconclusive failure/timeout (VAL-PRISM-024).
AUDIT_STATUS_PENDING = "pending"
AUDIT_STATUS_PASSED = "passed"
AUDIT_STATUS_MISMATCH = "mismatch"
AUDIT_STATUS_FAILED = "failed"

#: Audit resolutions recorded against the unit (distinct from the lifecycle status).
AUDIT_RESOLUTION_PASS = "pass"
AUDIT_RESOLUTION_MISMATCH = "mismatch"
AUDIT_RESOLUTION_INCONCLUSIVE = "inconclusive"


def audit_unit_id_for(submission_id: str) -> str:
    """Return the audit work-unit id for ``submission_id`` (distinct from the primary unit id)."""

    return f"{AUDIT_UNIT_PREFIX}{submission_id}"


def is_audit_unit_id(work_unit_id: str) -> bool:
    """Whether ``work_unit_id`` names an audit unit rather than a primary evaluation unit."""

    return work_unit_id.startswith(AUDIT_UNIT_PREFIX)


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


class SupportsAuditResolution(Protocol):
    """The slice of :class:`~prism_challenge.repository.PrismRepository` audit resolution needs."""

    async def get_audit_unit(self, audit_unit_id: str) -> dict[str, object] | None: ...

    async def record_audit_resolution(
        self,
        *,
        audit_unit_id: str,
        status: str,
        attempts: int,
        resolution: str | None,
        resolved_manifest_sha256: str | None,
        error: str | None,
    ) -> None: ...

    async def invalidate_submission_score(self, submission_id: str, *, reason: str) -> bool: ...


@dataclass(frozen=True)
class AuditResolution:
    """The observable outcome of resolving one audit unit against a validator replay."""

    audit_unit_id: str
    submission_id: str
    status: str
    resolution: str | None
    attempts: int
    invalidated: bool
    terminal: bool

    def to_response(self) -> dict[str, object]:
        return {
            "audit_unit_id": self.audit_unit_id,
            "submission_id": self.submission_id,
            "status": self.status,
            "resolution": self.resolution,
            "attempts": self.attempts,
            "invalidated": self.invalidated,
            "terminal": self.terminal,
        }


async def resolve_audit_unit(
    repository: SupportsAuditResolution,
    *,
    audit_unit_id: str,
    replay_manifest_sha256: str | None = None,
    failed: bool = False,
    error: str | None = None,
) -> AuditResolution:
    """Resolve an audit unit from a validator's authoritative replay (architecture.md 3.5).

    The validator replay is authoritative. Outcomes:

    * an authoritative replay hash EQUAL to the audited worker hash -> ``passed`` (the audited score
      is left untouched);
    * an authoritative replay hash DIFFERENT from it -> ``mismatch``: the audited submission's score
      is invalidated (VAL-PRISM-013) and the crown/weights recomputed (VAL-PRISM-023);
    * a replay FAILURE or TIMEOUT (``failed`` / no ``replay_manifest_sha256``) NEVER confirms the
      audited result: the unit is re-audited (back to ``pending``) until ``max_attempts`` is
      exhausted, then reaches the terminal, observable ``failed`` state -- the audited submission is
      left unresolved (never silently reverted to accepted) and NO fault is attributed, because a
      fault requires an authoritative divergent manifest (VAL-PRISM-024).

    Resolving an already-terminal unit is an idempotent no-op.
    """

    unit = await repository.get_audit_unit(audit_unit_id)
    if unit is None:
        raise KeyError(f"audit unit {audit_unit_id!r} not found")

    submission_id = str(unit["submission_id"])
    status = str(unit["status"])
    attempts = int(unit["attempts"])  # type: ignore[call-overload]
    if status in (AUDIT_STATUS_PASSED, AUDIT_STATUS_MISMATCH, AUDIT_STATUS_FAILED):
        return AuditResolution(
            audit_unit_id=audit_unit_id,
            submission_id=submission_id,
            status=status,
            resolution=(str(unit["resolution"]) if unit["resolution"] is not None else None),
            attempts=attempts,
            invalidated=status == AUDIT_STATUS_MISMATCH,
            terminal=True,
        )

    max_attempts = int(unit["max_attempts"])  # type: ignore[call-overload]
    attempts += 1
    inconclusive = failed or not replay_manifest_sha256

    if inconclusive:
        exhausted = attempts >= max_attempts
        new_status = AUDIT_STATUS_FAILED if exhausted else AUDIT_STATUS_PENDING
        await repository.record_audit_resolution(
            audit_unit_id=audit_unit_id,
            status=new_status,
            attempts=attempts,
            resolution=AUDIT_RESOLUTION_INCONCLUSIVE if exhausted else None,
            resolved_manifest_sha256=None,
            error=error or "audit replay failed or timed out",
        )
        return AuditResolution(
            audit_unit_id=audit_unit_id,
            submission_id=submission_id,
            status=new_status,
            resolution=AUDIT_RESOLUTION_INCONCLUSIVE if exhausted else None,
            attempts=attempts,
            invalidated=False,
            terminal=exhausted,
        )

    audited_hash = str(unit["audited_manifest_sha256"])
    matches = replay_manifest_sha256 == audited_hash
    invalidated = False
    if matches:
        new_status = AUDIT_STATUS_PASSED
        resolution = AUDIT_RESOLUTION_PASS
    else:
        new_status = AUDIT_STATUS_MISMATCH
        resolution = AUDIT_RESOLUTION_MISMATCH
        invalidated = await repository.invalidate_submission_score(
            submission_id,
            reason=f"audit invalidated: manifest mismatch (audit_unit={audit_unit_id})",
        )
    await repository.record_audit_resolution(
        audit_unit_id=audit_unit_id,
        status=new_status,
        attempts=attempts,
        resolution=resolution,
        resolved_manifest_sha256=replay_manifest_sha256,
        error=None,
    )
    return AuditResolution(
        audit_unit_id=audit_unit_id,
        submission_id=submission_id,
        status=new_status,
        resolution=resolution,
        attempts=attempts,
        invalidated=invalidated,
        terminal=True,
    )


__all__ = [
    "AUDIT_RESOLUTION_INCONCLUSIVE",
    "AUDIT_RESOLUTION_MISMATCH",
    "AUDIT_RESOLUTION_PASS",
    "AUDIT_STATUS_FAILED",
    "AUDIT_STATUS_MISMATCH",
    "AUDIT_STATUS_PASSED",
    "AUDIT_STATUS_PENDING",
    "AUDIT_UNIT_PREFIX",
    "TIER_0",
    "TIER_1",
    "TIER_2",
    "AuditDecision",
    "AuditResolution",
    "AuditSampler",
    "SupportsAuditResolution",
    "audit_sampler_from_config",
    "audit_unit_id_for",
    "effective_tier",
    "is_audit_unit_id",
    "is_tier_downgraded",
    "resolve_audit_unit",
]
