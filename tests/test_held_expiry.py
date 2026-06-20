from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from prism_challenge.db import Database
from prism_challenge.models import SubmissionCreate, SubmissionStatus
from prism_challenge.repository import PrismRepository, now_iso

CODE = "def build_model(ctx):\n    return None\n"

EXPIRY_REASON = "review hold expired without resolution"


@pytest.fixture
async def repository(tmp_path):
    database = Database(tmp_path / "held-expiry.sqlite3")
    await database.init()
    # Tiny held timeout so an updated_at a few seconds in the past counts as stale.
    return PrismRepository(database, epoch_seconds=60, held_review_timeout_seconds=1)


async def _seed_pending(repository: PrismRepository, hotkey: str = "miner-1") -> str:
    created = await repository.create_submission(
        hotkey, SubmissionCreate(code=CODE, filename="model.py", metadata={})
    )
    return created.id


async def _status_and_error(repository: PrismRepository, submission_id: str):
    async with repository.database.connect() as conn:
        rows = await conn.execute_fetchall(
            "SELECT status, error FROM submissions WHERE id=?", (submission_id,)
        )
    row = list(rows)[0]
    return str(row["status"]), row["error"]


async def _set_updated_at(repository: PrismRepository, submission_id: str, value: str) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "UPDATE submissions SET updated_at=? WHERE id=?", (value, submission_id)
        )


async def _component_hold(repository: PrismRepository, submission_id: str):
    async with repository.database.connect() as conn:
        rows = await conn.execute_fetchall(
            "SELECT id, status FROM component_review_holds WHERE submission_id=?",
            (submission_id,),
        )
    holds = list(rows)
    return holds[0] if holds else None


async def test_stuck_llm_held_is_expired_to_rejected(repository: PrismRepository) -> None:
    """CORE: a STUCK LLM held submission (held with NO pending component hold row,
    exactly the 3 unreachable LLM-quarantine paths) must be expired to the terminal
    `rejected` state once its hold time exceeds held_review_timeout_seconds."""
    submission_id = await _seed_pending(repository)

    # Simplest of the 3 stuck paths: LLM quarantine sets status='held' but creates
    # NO component_review_holds row -> unresolvable -> stuck forever without a reaper.
    await repository.quarantine_submission_for_llm_review(
        submission_id=submission_id,
        reason="llm suspicion without evidence",
        payload={"submission_id": submission_id},
    )

    # Drive its hold time into the past (older than held_review_timeout_seconds).
    stale = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat()
    await _set_updated_at(repository, submission_id, stale)

    # Reaper runs at the top of claim_next (and standalone).
    await repository.expire_stale_held()

    status, error = await _status_and_error(repository, submission_id)
    assert status == SubmissionStatus.REJECTED.value, "stuck LLM held was never expired"
    assert error == EXPIRY_REASON


async def test_duplicate_hold_is_not_expired_and_stays_held(
    repository: PrismRepository,
) -> None:
    """GUARD (proves scoping): a live duplicate-review hold (held WITH a pending
    component_review_holds row) must NOT be expired by the stuck-LLM reaper even when
    stale -- it stays held+pending (quarantined) rather than being stranded as rejected."""
    submission_id = await _seed_pending(repository)

    await repository.hold_submission_for_duplicate_review(
        submission_id=submission_id,
        reason="duplicate of existing submission",
        report={"source_similarity": 0.99, "graph_similarity": 0.98},
    )

    stale = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat()
    await _set_updated_at(repository, submission_id, stale)

    await repository.expire_stale_held()

    # NOT expired: still held, hold row still pending.
    status, _ = await _status_and_error(repository, submission_id)
    assert status == SubmissionStatus.HELD.value, "duplicate hold was wrongly expired"
    hold = await _component_hold(repository, submission_id)
    assert hold is not None
    assert str(hold["status"]) == "pending"


async def test_fresh_stuck_held_is_not_expired(repository: PrismRepository) -> None:
    """A stuck LLM held with a RECENT updated_at (inside the grace window) must NOT
    be expired -- only stale holds past the timeout are reaped."""
    submission_id = await _seed_pending(repository)

    await repository.quarantine_submission_for_llm_review(
        submission_id=submission_id,
        reason="llm suspicion without evidence",
        payload={"submission_id": submission_id},
    )
    # Fresh hold time -> within the grace window.
    await _set_updated_at(repository, submission_id, now_iso())

    await repository.expire_stale_held()

    status, _ = await _status_and_error(repository, submission_id)
    assert status == SubmissionStatus.HELD.value, "a fresh held row must not be expired"
