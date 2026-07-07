"""Decentralized validator execution of an assigned prism work unit.

The assigned (online, gpu) validator pulls its single prism work unit from the master coordination
plane and runs the WHOLE re-execution by dispatching to its OWN broker-backed ``DockerExecutor``
(monkeypatched to the CPU re-exec mock in tests). The master coordinator never invokes the executor
for the prism unit - it only assigns the work (VAL-PRISM-037).

Execution reuses :class:`~prism_challenge.queue.PrismWorker`, whose container path builds the
validator's broker executor and preserves forced-random-init + prequential bits-per-byte scoring.
Driving it through :meth:`PrismWorker.process_submission` (a CAS claim on the specific submission)
makes the loop idempotent and concurrency-safe: re-running a completed/in-flight unit is a no-op
that neither re-dispatches the broker nor mutates the recorded result (VAL-PRISM-002 / 004).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .coordination import (
    PRISM_DEFAULT_CONCURRENCY,
    PRISM_WORK_UNIT_CAPABILITY,
    RESUME_CHECKPOINT_PAYLOAD_KEY,
    PrismWorkUnit,
    list_pending_prism_work_units,
    pull_assigned_work_units,
)
from .proof import (
    WorkerSigner,
    build_execution_proof_from_manifest,
    worker_signer_from_key,
)
from .queue import PrismWorker

logger = logging.getLogger(__name__)

#: Submission statuses at which a prism work unit is terminal (no re-execution, no re-dispatch).
TERMINAL_SUBMISSION_STATUSES = frozenset({"completed", "failed", "rejected"})


@dataclass(frozen=True)
class PrismWorkUnitExecution:
    """Outcome of executing (or idempotently skipping) one assigned prism work unit."""

    work_unit_id: str
    submission_id: str
    status: str
    #: True when the validator's broker was actually dispatched (False = idempotent no-op).
    executed: bool
    #: True when a fresh result was persisted by this run (False = already terminal).
    posted: bool
    #: The serialized ExecutionProof emitted for this unit's result payload, when the worker plane
    #: is enabled and a fresh successful finalization produced a manifest to bind (else ``None``).
    execution_proof: dict[str, Any] | None = None


@dataclass(frozen=True)
class PrismValidatorCycleSummary:
    """Aggregate of one validator pull/execute/post cycle."""

    pulled: int
    executed: int
    skipped: int
    completed_submissions: tuple[str, ...]
    #: ExecutionProofs emitted this cycle, keyed by ``work_unit_id`` (empty when the plane is off).
    execution_proofs: dict[str, dict[str, Any]] = field(default_factory=dict)


async def execute_work_unit(
    worker: PrismWorker,
    unit: PrismWorkUnit,
    *,
    proof_signer: WorkerSigner | None = None,
    proof_env: Mapping[str, str] | None = None,
) -> PrismWorkUnitExecution:
    """Run one assigned prism re-execution on the validator's own broker and report the outcome.

    Idempotent: :meth:`PrismWorker.process_submission` claims the submission only while it is
    pending, so a unit that already reached a terminal state is not re-dispatched and its recorded
    result is left untouched. A reassigned unit carrying ``resume_checkpoint_ref`` in its payload
    resumes from the last public HF checkpoint instead of restarting (VAL-PRISM-023).

    When the worker plane is enabled, a fresh successful finalization also emits an
    :class:`~prism_challenge.proof.ExecutionProof` bound to this unit (architecture.md 3.4).
    """

    resume_ref = unit.payload.get(RESUME_CHECKPOINT_PAYLOAD_KEY)
    result_id = await worker.process_submission(
        unit.submission_id,
        resume_checkpoint_ref=str(resume_ref) if resume_ref else None,
    )
    executed = result_id is not None
    status = await worker.repository.submission_status(unit.submission_id)
    execution_proof: dict[str, Any] | None = None
    if executed and status == "completed":
        execution_proof = await _emit_execution_proof(
            worker, unit, signer=proof_signer, env=proof_env
        )
    return PrismWorkUnitExecution(
        work_unit_id=unit.work_unit_id,
        submission_id=unit.submission_id,
        status=status or "",
        executed=executed,
        posted=executed,
        execution_proof=execution_proof,
    )


async def _emit_execution_proof(
    worker: PrismWorker,
    unit: PrismWorkUnit,
    *,
    signer: WorkerSigner | None,
    env: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    """Build the ExecutionProof for a freshly finalized unit, or ``None`` when not applicable.

    Gated on ``worker_plane.enabled``. The manifest hash is taken from the exact on-disk bytes of
    the run's ``prism_run_manifest.v2.json``; the provenance comes ONLY from the non-secret provider
    env allowlist. Held-out split config and LLM keys are never read (VAL-PRISM-008).
    """

    if not worker.settings.worker_plane.enabled:
        return None
    resolved_signer = _resolve_proof_signer(worker, signer)
    if resolved_signer is None:
        logger.warning(
            "worker plane enabled but no signing key configured; no ExecutionProof emitted "
            "for work_unit=%s",
            unit.work_unit_id,
        )
        return None
    manifest_path = await worker.repository.latest_run_manifest_path(
        unit.submission_id, worker.execution_backend
    )
    if not manifest_path:
        return None
    proof = build_execution_proof_from_manifest(
        signer=resolved_signer,
        unit_id=unit.work_unit_id,
        manifest_path=manifest_path,
        env=os.environ if env is None else env,
    )
    return proof.model_dump(mode="json")


def _resolve_proof_signer(
    worker: PrismWorker, signer: WorkerSigner | None
) -> WorkerSigner | None:
    if signer is not None:
        return signer
    key = worker.settings.worker_plane.signing_key
    if not key:
        return None
    return worker_signer_from_key(key)


async def run_validator_cycle(
    *,
    worker: PrismWorker,
    work_unit_ids: Iterable[str] | None = None,
    capabilities: Iterable[str] = (PRISM_WORK_UNIT_CAPABILITY,),
    in_flight: int | None = None,
    max_concurrency: int = PRISM_DEFAULT_CONCURRENCY,
    proof_signer: WorkerSigner | None = None,
    proof_env: Mapping[str, str] | None = None,
) -> PrismValidatorCycleSummary:
    """Run one decentralized validator cycle: pull -> execute (own broker) -> post.

    Pulls the caller's assigned, capability-matched prism units (at most
    ``max_concurrency - in_flight`` of them, so a busy validator runs one submission at a time),
    executes each on the validator's own broker, and reports which submissions completed. The pull
    and assignment are execution-free; only :func:`execute_work_unit` dispatches the broker.

    ``in_flight`` defaults to the validator's REAL in-flight draw (the count of currently-running
    submissions) so the concurrency-1 cap is enforced against reality rather than a static zero; a
    caller may override it (e.g. tests pinning a specific value).
    """

    if in_flight is None:
        in_flight = await worker.repository.count_in_flight_submissions()
    units = await list_pending_prism_work_units(worker.repository)
    pulled = pull_assigned_work_units(
        units,
        work_unit_ids=work_unit_ids,
        capabilities=capabilities,
        in_flight=in_flight,
        max_concurrency=max_concurrency,
    )
    executed = 0
    skipped = 0
    completed: list[str] = []
    execution_proofs: dict[str, dict[str, Any]] = {}
    for unit in pulled:
        outcome = await execute_work_unit(
            worker, unit, proof_signer=proof_signer, proof_env=proof_env
        )
        if outcome.executed:
            executed += 1
        else:
            skipped += 1
        if outcome.status == "completed":
            completed.append(outcome.submission_id)
        if outcome.execution_proof is not None:
            execution_proofs[outcome.work_unit_id] = outcome.execution_proof
    return PrismValidatorCycleSummary(
        pulled=len(pulled),
        executed=executed,
        skipped=skipped,
        completed_submissions=tuple(completed),
        execution_proofs=execution_proofs,
    )
