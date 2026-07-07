#!/usr/bin/env python
"""Local mission prism service: worker plane ON + admission gate + explicit CPU re-exec test mode.

Part of the cross-repo local end-to-end harness (base docs/operations/mission-harness.md). Serves
the real prism challenge API on a loopback port configured so that:

* the worker plane is ON and the admission gate requires >=1 active worker for the submitting
  hotkey (queried from the base master, reusing the shared bridge token as the internal bearer);
* the repo's OWN CPU re-exec seam is installed as EXPLICIT test-mode config
  (``worker_plane.cpu_reexec_test_mode``) so any re-execution runs deterministically on CPU with no
  GPU/Docker/broker; and
* results are finalized from the base worker plane's forwarded ExecutionProof (no self-evaluation).

CONFIG-DRIVEN (JSON path in ``argv[1]`` / ``$MISSION_PRISM_CONFIG``). NOT for production.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import uvicorn

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings


def _load_config() -> dict[str, Any]:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import os

        path = os.environ.get("MISSION_PRISM_CONFIG")
    if not path:
        raise SystemExit("usage: mission_prism.py <config.json>")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_settings(config: dict[str, Any]) -> PrismSettings:
    token = config["token"]
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{config['db_path']}",
        shared_token=token,
        shared_token_file=None,
        allow_insecure_signatures=True,
        llm_review_enabled=False,
        llm_review_required=False,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://mission-broker:8082",
        docker_broker_token=token,
        sequence_length=int(config.get("sequence_length", 16)),
        base_eval_artifact_root=Path(config["artifact_root"]),
        public_submissions_enabled=True,
        worker_plane={
            "enabled": True,
            "admission_requires_worker": bool(config.get("admission_requires_worker", True)),
            "master_base_url": config["master_base_url"],
            "cpu_reexec_test_mode": True,
            "cpu_reexec_train_data_dir": config.get("train_data_dir"),
        },
    )


def main() -> None:
    config = _load_config()
    settings = build_settings(config)
    uvicorn.run(
        create_app(settings),
        host=str(config.get("host", "127.0.0.1")),
        port=int(config["port"]),
        log_level=str(config.get("log_level", "warning")),
    )


if __name__ == "__main__":
    main()
