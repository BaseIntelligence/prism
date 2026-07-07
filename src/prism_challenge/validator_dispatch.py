"""Validator dispatch entrypoint for prism work units (architecture sec 4, G2).

The platform validator agent (``base validator agent``) pulls the single prism
gpu work unit from the master coordination plane and dispatches it here (selected
by ``challenge_slug``). :func:`dispatch_assignment` runs the GPU re-execution on
the validator's OWN broker by driving the production :class:`PrismWorker` (built
via :func:`prism_challenge.app.create_app`, the same construction the deployed
challenge uses) through :func:`run_validator_cycle`: the eval container runs
``network=none`` mounting only the locked train split + writable artifacts (never
val/test), with concurrency 1 enforced against the validator's real in-flight
draw.

The prism LLM review (claude-opus) routes through the master gateway using the
per-assignment scoped token, and the raw provider key is stripped from the
validator's settings (see :func:`gateway_scoped_settings`) so it never reaches
the eval host. Re-running an already-terminal submission is an idempotent no-op
(the worker's CAS claim neither re-dispatches the broker nor mutates the recorded
score).

The signature deliberately uses only plain types (no dependency on the platform
validator-agent package), so this runs against the published ``base`` while the
platform side maps it onto the validator agent's executor seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .app import create_app
from .audit import is_audit_unit_id
from .config import PrismSettings
from .config import settings as default_settings
from .evaluator.checkpoint_publisher import CheckpointPublisher
from .proof import MANIFEST_PAYLOAD_KEY, PROOF_PAYLOAD_KEY
from .validator_executor import run_primary_execution_cycle, run_validator_audit_cycle

CHALLENGE_SLUG = "prism"

_GATEWAY_TOKEN_PAYLOAD_KEYS = ("gateway_token", "BASE_GATEWAY_TOKEN")
_GATEWAY_URL_PAYLOAD_KEY = "BASE_LLM_GATEWAY_URL"
_GATEWAY_BASE_URL_PAYLOAD_KEYS = ("gateway_url", "gateway_base_url")
_LLM_GATEWAY_PATH = "/llm/v1"


class PrismGatewayConfigError(ValueError):
    """A prism assignment payload cannot yield a master gateway config.

    Raised BEFORE any broker dispatch so the validator never runs the prism LLM
    review without the master gateway (which would require a raw provider key on
    the validator).
    """


async def dispatch_assignment(
    *,
    work_unit_id: str,
    payload: Mapping[str, Any],
    broker_url: str,
    broker_token: str | None = None,
    broker_token_file: str | None = None,
    settings: PrismSettings | None = None,
    checkpoint_publisher: CheckpointPublisher | None = None,
) -> dict[str, Any]:
    """Run a pulled prism assignment on the caller's own broker (architecture.md 4).

    Routes on the UNIT TYPE so the same dispatch entrypoint serves both roles that reuse it:

    * an ``audit:`` unit (only assigned to a VALIDATOR, and only when the worker plane is on) is
      replayed + resolved (see :func:`_dispatch_audit_only`); the finalized score is untouched on a
      matching replay and invalidated + fault-recorded on a divergent one (VAL-FINAL-005);
    * any other (PRIMARY) unit is GPU-re-executed via :func:`run_primary_execution_cycle` -- this is
      the miner-funded worker path when the flag is on (it emits the ExecutionProof base reconciles)
      and the legacy validator re-execution when the flag is off.

    The autonomous validator cycle's audit-only restriction lives in :func:`run_validator_cycle`;
    here a primary unit always executes because the base assignment plane only ever routes a primary
    unit to a worker (flag on) or a legacy validator (flag off), never to an audit-only validator.
    Returns the cycle counts for the platform agent to post back to the master.
    """

    base_settings = settings or default_settings
    if base_settings.worker_plane.enabled and is_audit_unit_id(work_unit_id):
        return await _dispatch_audit_only(
            work_unit_id=work_unit_id,
            broker_url=broker_url,
            broker_token=broker_token,
            broker_token_file=broker_token_file,
            settings=base_settings,
            checkpoint_publisher=checkpoint_publisher,
        )

    effective = gateway_scoped_settings(
        base_settings,
        payload,
        broker_url=broker_url,
        broker_token=broker_token,
        broker_token_file=broker_token_file,
    )
    app = create_app(effective, checkpoint_publisher=checkpoint_publisher)
    database = app.state.database
    await database.init()
    try:
        summary = await run_primary_execution_cycle(
            worker=app.state.worker, work_unit_ids=[work_unit_id]
        )
    finally:
        await database.close()
    result: dict[str, Any] = {
        "pulled": summary.pulled,
        "executed": summary.executed,
        "skipped": summary.skipped,
        "completed_submissions": list(summary.completed_submissions),
    }
    # Emit the ExecutionProof IN the work-unit result payload at successful finalization
    # (architecture.md 3.4; VAL-PRISM-001). Absent when the worker plane is off or the unit did not
    # freshly finalize. The backing run manifest is forwarded alongside so the accepting plane can
    # recompute + verify the signed digest and reject a tampered manifest (VAL-PRISM-007).
    proof = summary.execution_proofs.get(work_unit_id)
    if proof is not None:
        result[PROOF_PAYLOAD_KEY] = proof
        manifest = summary.execution_manifests.get(work_unit_id)
        if manifest is not None:
            result[MANIFEST_PAYLOAD_KEY] = manifest
    return result


async def _dispatch_audit_only(
    *,
    work_unit_id: str,
    broker_url: str,
    broker_token: str | None,
    broker_token_file: str | None,
    settings: PrismSettings,
    checkpoint_publisher: CheckpointPublisher | None,
) -> dict[str, Any]:
    """Audit-only dispatch: replay + resolve a sampled ``audit:`` unit (VAL-FINAL-005).

    Audits skip the LLM review, so this path needs NO master gateway scoped token (unlike the
    primary path) -- only the validator's own broker for the deterministic re-execution.
    """

    effective = settings.model_copy(
        update={
            "docker_broker_url": broker_url,
            "docker_broker_token": broker_token,
            "docker_broker_token_file": broker_token_file,
        }
    )
    app = create_app(effective, checkpoint_publisher=checkpoint_publisher)
    database = app.state.database
    await database.init()
    try:
        summary = await run_validator_audit_cycle(
            worker=app.state.worker, work_unit_ids=[work_unit_id]
        )
    finally:
        await database.close()
    invalidated = sum(1 for resolution in summary.audits if resolution.invalidated)
    return {
        "pulled": summary.pulled,
        "executed": summary.executed,
        "skipped": summary.skipped,
        "completed_submissions": [],
        "is_audit": is_audit_unit_id(work_unit_id),
        "audits_resolved": len(summary.audits),
        "audits_invalidated": invalidated,
        "audit_results": [resolution.to_response() for resolution in summary.audits],
    }


def gateway_scoped_settings(
    settings: PrismSettings,
    payload: Mapping[str, Any],
    *,
    broker_url: str,
    broker_token: str | None = None,
    broker_token_file: str | None = None,
) -> PrismSettings:
    """Return settings bound to the validator's broker + the scoped gateway.

    The raw provider key is stripped so the validator routes the prism LLM review
    only through the master gateway with the per-assignment scoped token. Raises
    :class:`PrismGatewayConfigError` BEFORE any broker dispatch when the payload
    cannot yield a gateway config.
    """

    token = _first_present(payload, _GATEWAY_TOKEN_PAYLOAD_KEYS)
    if not token:
        raise PrismGatewayConfigError("prism assignment payload is missing a scoped gateway token")
    gateway_url = payload.get(_GATEWAY_URL_PAYLOAD_KEY)
    if not gateway_url:
        base = _first_present(payload, _GATEWAY_BASE_URL_PAYLOAD_KEYS)
        if base:
            gateway_url = f"{str(base).rstrip('/')}{_LLM_GATEWAY_PATH}"
    if not gateway_url:
        raise PrismGatewayConfigError(
            "prism assignment payload is missing the master gateway base URL"
        )
    return settings.model_copy(
        update={
            "docker_broker_url": broker_url,
            "docker_broker_token": broker_token,
            "docker_broker_token_file": broker_token_file,
            "llm_gateway_url": str(gateway_url),
            "llm_gateway_token": str(token),
            "llm_gateway_token_file": None,
        }
    )


def _first_present(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value:
            return value
    return None


__all__ = [
    "CHALLENGE_SLUG",
    "PrismGatewayConfigError",
    "dispatch_assignment",
    "gateway_scoped_settings",
]
