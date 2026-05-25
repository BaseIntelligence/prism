from __future__ import annotations

import anyio

from prism_challenge.db import Database, loads
from prism_challenge.models import SubmissionCreate
from prism_challenge.repository import PrismRepository


async def _repo(tmp_path):
    database = Database(tmp_path / "llm-flow.sqlite3")
    await database.init()
    return PrismRepository(database, epoch_seconds=3600)


async def _submission(repo: PrismRepository) -> str:
    created = await repo.create_submission(
        "miner", SubmissionCreate(code="def build_model(ctx):\n    return None\n")
    )
    return created.id


def test_verdict_requires_mermaid(tmp_path):
    async def run() -> None:
        repo = await _repo(tmp_path)
        submission_id = await _submission(repo)

        try:
            await repo.submit_llm_verdict(
                submission_id=submission_id,
                approved=True,
                reason="looks safe",
                violations=[],
                confidence=0.7,
                raw={"reason": "looks safe", "verdict": True},
            )
        except ValueError as exc:
            assert str(exc) == (
                "llm_review_order_error: submit_mermaid required before submit_verdict"
            )
        else:
            raise AssertionError("submit_verdict succeeded before submit_mermaid")

    anyio.run(run)


def test_mermaid_submission_is_idempotent_and_audited(tmp_path):
    async def run() -> None:
        repo = await _repo(tmp_path)
        submission_id = await _submission(repo)
        mermaid = "flowchart LR\n  A[Model] --> B[Logits]"

        await repo.submit_llm_mermaid(submission_id=submission_id, mermaid=mermaid)
        await repo.submit_llm_mermaid(submission_id=submission_id, mermaid=mermaid)
        await repo.submit_llm_verdict(
            submission_id=submission_id,
            approved=True,
            reason="review accepted",
            violations=[],
            confidence=0.9,
            raw={"reason": "review accepted", "verdict": True},
        )

        async with repo.database.connect() as conn:
            events = await conn.execute_fetchall(
                "SELECT state, tool_name, payload FROM llm_review_events "
                "WHERE submission_id=? ORDER BY sequence",
                (submission_id,),
            )
            review = list(
                await conn.execute_fetchall(
                    "SELECT mermaid, final_state FROM llm_reviews WHERE submission_id=?",
                    (submission_id,),
                )
            )[0]

        rows = list(events)
        assert [row["state"] for row in rows] == [
            "mermaid_submitted",
            "verdict_submitted",
            "accepted",
        ]
        assert [row["tool_name"] for row in rows].count("SubmitMermaid") == 1
        assert loads(str(rows[0]["payload"]))["mermaid"] == mermaid
        assert review["mermaid"] == mermaid
        assert review["final_state"] == "accepted"

    anyio.run(run)
