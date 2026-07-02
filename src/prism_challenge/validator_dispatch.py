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
from .config import PrismSettings
from .config import settings as default_settings
from .evaluator.checkpoint_publisher import CheckpointPublisher
from .validator_executor import run_validator_cycle

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
    """Run a pulled prism assignment's GPU re-execution on the validator's broker.

    Returns the cycle counts (pulled/executed/skipped/completed_submissions) for
    the platform validator agent to post back to the master.
    """

    effective = gateway_scoped_settings(
        settings or default_settings,
        payload,
        broker_url=broker_url,
        broker_token=broker_token,
        broker_token_file=broker_token_file,
    )
    app = create_app(effective, checkpoint_publisher=checkpoint_publisher)
    database = app.state.database
    await database.init()
    try:
        summary = await run_validator_cycle(worker=app.state.worker, work_unit_ids=[work_unit_id])
    finally:
        await database.close()
    return {
        "pulled": summary.pulled,
        "executed": summary.executed,
        "skipped": summary.skipped,
        "completed_submissions": list(summary.completed_submissions),
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
