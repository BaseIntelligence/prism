#!/usr/bin/env python
"""Local STAGING E2E smoke driver for the Prism challenge pipeline.

WHAT THIS IS
------------
A reproducible, fully LOCAL replica of the production submission pipeline. It
spins up the *real* FastAPI app (`create_app`) against a FRESH temp SQLite DB
initialised by the repo's own startup/migrations, submits a synthetic dev-signed
MINER submission, drives the worker via the internal ``process-next`` endpoint
until the submission reaches a terminal state, and then reports whether a
``scores`` row was written and whether the public leaderboard contains the id.

NO prod. NO network. NO real keys. Dev/insecure HMAC signing only
(``allow_insecure_signatures=True`` + the shared token "secret").

HOW TO RUN (headless)
---------------------
    cd /droid/prism-challenge && .venv/bin/python scripts/staging_e2e.py

Exit code is 0 on a successful *run of the harness* (the pipeline trace is
printed regardless of whether the pipeline itself is green). A non-zero exit
means the harness itself failed (e.g. submit was rejected, worker errored).

REPRODUCIBLE ENV / KNOBS
------------------------
- ``STAGING_DB``            : path to the temp SQLite file to use. If unset a
                              throwaway temp dir is created per run (recommended).
                              Never point this at a prod DB.
- ``STAGING_PUBLIC_FLAG``   : "1"/"0" to toggle ``public_submissions_enabled``
                              (default "1"). Set "0" to assert the 404 guard.
- ``STAGING_EXEC_MODE``     : submission ``metadata.execution_mode``. Default
                              "local_cpu_smoke" -- the ONLY eval path that can
                              run to completion on a local CPU box. Set to
                              "" (empty) to exercise the DEFAULT path
                              (GPU_PROXY_EVAL -> full container/finalize path,
                              queue.py:1142). EMPIRICALLY this path acquires the
                              synthetic "local-platform" GPU lease and then BLOCKS
                              on a real container eval that never returns on a
                              CPU-only box -- i.e. the default config is NOT
                              completable locally, so the local-completable smoke
                              path (local_cpu_smoke) is the harness default.
                              (Setting it empty will HANG; use only to confirm.)

SYNTHETIC MINER KEY PAIR
------------------------
There is no sr25519 key. The "key pair" is the dev-HMAC pair:
  - public id  : the hotkey string "miner-1"
  - secret     : the shared token "secret" (settings.shared_token)
The canonical message ``prism:{hotkey}:{nonce}:{ts}:{sha256(body)}`` is HMAC'd
with the secret -- exactly what ``tests/conftest.py:signed_headers`` produces.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# --- make the repo importable as a standalone script -------------------------
# pytest sets pythonpath=["src"]; a plain script must do it itself. We also add
# the tests dir so we can REUSE the existing dev-signing helper from conftest.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
_TESTS = _REPO_ROOT / "tests"
for _p in (str(_SRC), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the EXISTING dev-signing helper + sample model code from the test suite
# (no real key needed; it accepts hotkey=). This is the synthetic key path.
from conftest import VALID_CODE, signed_headers  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from prism_challenge.app import create_app  # noqa: E402
from prism_challenge.config import PrismSettings  # noqa: E402

SHARED_TOKEN = "secret"
MINER_HOTKEY = "miner-1"
VALIDATOR_HOTKEY = "val-a"
TERMINAL_STATES = {"completed", "failed", "rejected", "held"}
MAX_PROCESS_STEPS = 25


def log(line: str) -> None:
    """Single greppable trace line (prefixed STEP / NOTE / RESULT)."""
    print(line, flush=True)


def build_settings(db_path: Path, *, public_flag: bool) -> PrismSettings:
    """Replica-of-prod settings, but fully local + dev-signed + isolated DB."""
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token=SHARED_TOKEN,
        allow_insecure_signatures=True,
        public_submissions_enabled=public_flag,
        validator_hotkeys=(VALIDATOR_HOTKEY,),
        plagiarism_enabled=False,
        fineweb_sample_count=4,
    )


def submit(
    client: TestClient,
    *,
    hotkey: str,
    nonce: str,
    exec_mode: str | None,
) -> tuple[int, dict]:
    """POST a synthetic dev-signed submission as the given miner hotkey."""
    metadata: dict[str, object] = {}
    if exec_mode:
        metadata["execution_mode"] = exec_mode
    payload = {"code": VALID_CODE, "filename": "model.py", "metadata": metadata}
    body = json.dumps(payload).encode()
    headers = signed_headers(SHARED_TOKEN, body, hotkey=hotkey, nonce=nonce)
    headers["Content-Type"] = "application/json"
    resp = client.post("/v1/submissions", content=body, headers=headers)
    try:
        data = resp.json()
    except Exception:
        data = {"_raw": resp.text}
    return resp.status_code, data


def submission_status(client: TestClient, submission_id: str) -> str:
    resp = client.get(f"/v1/submissions/{submission_id}")
    if resp.status_code != 200:
        return f"<http {resp.status_code}>"
    return str(resp.json().get("status", "<missing>"))


def process_next(client: TestClient) -> str | None:
    """Invoke the worker driver exactly like prod's internal poller does."""
    resp = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": f"Bearer {SHARED_TOKEN}"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"process-next failed: http {resp.status_code} {resp.text}")
    return resp.json().get("submission_id")


def scores_row_present(db_path: Path, submission_id: str) -> bool:
    """Inspect the raw SQLite DB for a scores row (no public endpoint exists)."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM scores WHERE submission_id=?", (submission_id,)
        )
        return cur.fetchone()[0] > 0
    finally:
        conn.close()


def leaderboard_contains(client: TestClient, submission_id: str) -> bool:
    resp = client.get("/v1/leaderboard")
    if resp.status_code != 200:
        return False
    entries = resp.json().get("entries", [])
    return any(str(e.get("submission_id")) == submission_id for e in entries)


def drive_to_terminal(client: TestClient, submission_id: str) -> str:
    """Call process-next repeatedly, tracing each hop, until terminal."""
    status = submission_status(client, submission_id)
    log(f"STEP status_initial={status}")
    for i in range(1, MAX_PROCESS_STEPS + 1):
        returned = process_next(client)
        status = submission_status(client, submission_id)
        log(f"STEP process_next#{i} returned={returned} status={status}")
        if status in TERMINAL_STATES:
            return status
        if returned is None and status not in TERMINAL_STATES:
            # Nothing left claimable yet status is non-terminal: the default
            # GPU path bounces the row back to pending/queued with no local GPU.
            log(
                f"NOTE process_next returned None while status={status!r} "
                "-> submission not claimable to completion on this local box"
            )
            return status
    log(f"NOTE reached MAX_PROCESS_STEPS={MAX_PROCESS_STEPS} without terminal state")
    return status


def run_happy_path(client: TestClient, db_path: Path, exec_mode: str | None) -> None:
    log("=== HAPPY PATH: synthetic miner submission -> process_next -> leaderboard ===")
    log(f"NOTE exec_mode={exec_mode!r} (empty => default GPU_PROXY_EVAL path)")
    code, data = submit(client, hotkey=MINER_HOTKEY, nonce="miner-n1", exec_mode=exec_mode)
    if code != 200:
        log(f"STEP submit -> {code} body={data}")
        raise SystemExit(f"harness error: expected 200 from miner submit, got {code}")
    submission_id = str(data["id"])
    log(f"STEP submit -> 200 id={submission_id} hotkey={MINER_HOTKEY}")

    final_status = drive_to_terminal(client, submission_id)
    log(f"STEP terminal_status={final_status}")

    scores = scores_row_present(db_path, submission_id)
    log(f"STEP scores_row={'present' if scores else 'MISSING'}")

    in_board = leaderboard_contains(client, submission_id)
    log(f"STEP leaderboard_contains_id={'true' if in_board else 'false'}")

    # Which eval path did we exercise? Make it explicit & greppable.
    if exec_mode == "local_cpu_smoke":
        log(
            "NOTE eval_path=local_cpu_smoke (queue.py:695-699) -> marks COMPLETED "
            "WITHOUT inserting a scores row (the T13 leak)"
        )
    elif not exec_mode:
        log(
            "NOTE eval_path=default GPU_PROXY_EVAL -> full container/finalize path "
            "(queue.py:1142); not completable on a local CPU box"
        )

    if final_status == "completed" and not scores:
        log(
            "RESULT BASELINE-DEFECT: submission COMPLETED but NO scores row -> "
            "leaderboard cannot include it (T13 completed-without-scores leak)"
        )
    elif final_status == "completed" and scores and in_board:
        log("RESULT GREEN: completed + scores row + present on leaderboard")
    else:
        log(
            f"RESULT status={final_status} scores={'present' if scores else 'MISSING'} "
            f"leaderboard={'true' if in_board else 'false'}"
        )


def run_validator_guard_check(client: TestClient) -> None:
    """Bonus: the validator hotkey must be refused (403) by the anti-self guard."""
    log("=== BONUS: validator self-submission guard (expect 403) ===")
    code, _ = submit(client, hotkey=VALIDATOR_HOTKEY, nonce="val-n1", exec_mode="local_cpu_smoke")
    ok = code == 403
    log(f"STEP validator_submit -> {code} guard_ok={'true' if ok else 'false'}")


def main() -> int:
    public_flag = os.environ.get("STAGING_PUBLIC_FLAG", "1") == "1"
    exec_mode_env = os.environ.get("STAGING_EXEC_MODE", "local_cpu_smoke")
    exec_mode: str | None = exec_mode_env if exec_mode_env != "" else None

    staging_db = os.environ.get("STAGING_DB")
    tmpdir: tempfile.TemporaryDirectory | None = None
    if staging_db:
        db_path = Path(staging_db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()  # fresh DB per run; never reuse stale state
    else:
        tmpdir = tempfile.TemporaryDirectory(prefix="prism-staging-")
        db_path = Path(tmpdir.name) / "prism.sqlite3"

    log("=== Prism LOCAL STAGING E2E smoke ===")
    log(f"NOTE db_path={db_path} (fresh, isolated; initialised by app startup)")
    log(f"NOTE public_submissions_enabled={public_flag} validator_hotkeys=({VALIDATOR_HOTKEY!r},)")
    log(f"NOTE synthetic_miner_keypair: public_id={MINER_HOTKEY!r} secret=<shared_token 'secret'>")

    settings = build_settings(db_path, public_flag=public_flag)
    try:
        # TestClient context manager runs the lifespan -> database.init()/migrations.
        with TestClient(create_app(settings)) as client:
            log("STEP app_startup=ok (migrations applied to fresh DB)")
            if not public_flag:
                code, _ = submit(
                    client, hotkey=MINER_HOTKEY, nonce="off-n1", exec_mode=exec_mode
                )
                log(f"STEP flag_off submit -> {code} (expect 404 'submission route disabled')")
                return 0
            run_happy_path(client, db_path, exec_mode)
            run_validator_guard_check(client)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    log("=== DONE (harness exit 0; see RESULT line for pipeline verdict) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
