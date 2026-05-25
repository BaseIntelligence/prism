from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from conftest import signed_headers
from fastapi.testclient import TestClient
from test_artifact_manifest import _valid_manifest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.schemas import RUN_MANIFEST_FILENAME, ExecutionMode
from prism_challenge.sdk.executors.docker import DockerExecutorError, DockerRunResult

REMOTE_ONLY_CODE = """
def build_model(ctx):
    raise RuntimeError('fake Platform broker must not execute code in unit tests')

def get_recipe(ctx):
    return {'learning_rate': 0.0003, 'batch_size': 2}
"""


def _submit(client: TestClient, code: str, nonce: str, metadata: dict | None = None) -> str:
    body = json.dumps(
        {"code": code, "filename": "model.py", "metadata": metadata or {}},
        separators=(",", ":"),
    ).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _settings(db_path: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="platform_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://platform-docker-broker:8082",
        docker_broker_token="secret",
        platform_eval_gpu_count=2,
        platform_eval_max_gpu_count=4,
        platform_eval_gpu_type="l4",
        platform_gpu_targets=json.dumps(
            [{"id": "target-a", "server": "server-a", "gpu_count": 4}],
            separators=(",", ":"),
        ),
        component_rewards_enabled=False,
        plagiarism_enabled=False,
    )


def _gpu_manifest(mode: ExecutionMode = ExecutionMode.GPU_PROXY_EVAL) -> dict:
    manifest = _valid_manifest(mode.value)
    manifest["compute"]["gpu_count"] = 2
    manifest["compute"]["gpu_type"] = "l4"
    manifest["compute"]["gpu_server"] = "server-a"
    manifest["compute"]["gpu_device_ids"] = ["0", "1"]
    manifest["metrics"]["gpu_count"] = 2
    return manifest


def _eval_job_row(db_path: Path, submission_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM eval_jobs WHERE submission_id=? AND level='platform_gpu' "
            "ORDER BY created_at DESC LIMIT 1",
            (submission_id,),
        ).fetchall()
    finally:
        conn.close()
    assert rows
    return rows[0]


def test_gpu_job_spec_includes_platform_v10_allocation_and_artifact_contract(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(), separators=(",", ":")),
            encoding="utf-8",
        )
        captured["spec"] = spec
        captured["payload"] = json.loads((spec.mounts[0].source / "payload.json").read_text())
        captured["timeout_seconds"] = timeout_seconds
        return DockerRunResult("platform-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-job.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="gpu-job-spec")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    spec = captured["spec"]
    payload = captured["payload"]
    artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
    row = _eval_job_row(db_path, submission_id)

    assert status["status"] == "completed"
    assert status["q_arch"] > 0
    assert spec.labels["platform.job"] == submission_id
    assert spec.labels["platform.task"] == "architecture"
    assert spec.labels["prism.actual_gpu_count"] == "2"
    assert spec.labels["prism.max_gpu_count"] == "4"
    assert spec.labels["prism.gpu_type"] == "l4"
    assert spec.labels["prism.target_id"] == "target-a"
    assert spec.labels["prism.target_server"] == "server-a"
    assert spec.labels["prism.device_ids"] == "0,1"
    assert spec.env["PRISM_GPU_COUNT"] == "2"
    assert spec.env["PRISM_MAX_GPU_COUNT"] == "4"
    assert spec.env["PRISM_RUN_MANIFEST_PATH"] == f"/artifacts/{RUN_MANIFEST_FILENAME}"
    assert not artifact_mount.read_only
    assert payload["gpu_allocation"] == {
        "actual_gpu_count": 2,
        "max_gpu_count": 4,
        "gpu_type": "l4",
        "target_id": "target-a",
        "target_server": "server-a",
        "device_ids": ["0", "1"],
    }
    assert payload["artifact_output"] == {
        "mount": "/artifacts",
        "path": "/artifacts",
        "manifest_path": f"/artifacts/{RUN_MANIFEST_FILENAME}",
    }
    assert payload["execution_mode"] == ExecutionMode.GPU_PROXY_EVAL.value
    assert payload["mode_spec"]["mode"] == ExecutionMode.GPU_PROXY_EVAL.value
    assert payload["mode_spec"]["token_budget"] == 10_000_000_000
    assert payload["mode_spec"]["dataset"]["subset"] == "sample-10BT"
    assert payload["mode_spec"]["resource_profile"]["official_fixed_profile"] is True
    assert row["status"] == "completed"
    assert row["actual_gpu_count"] == 2
    assert row["requested_gpu_count"] == 2
    assert row["target_id"] == "target-a"
    assert row["target_server"] == "server-a"
    assert json.loads(row["gpu_device_ids"]) == ["0", "1"]
    assert row["artifact_output_path"] == "/artifacts"
    assert row["run_manifest_path"] == f"/artifacts/{RUN_MANIFEST_FILENAME}"
    assert captured["timeout_seconds"] == 900


def test_infrastructure_failure_not_submission_failure(tmp_path, monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        raise DockerExecutorError("Docker broker is unavailable: connection refused")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-infra.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="infra-failure")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    row = _eval_job_row(db_path, submission_id)

    assert status["status"] == "pending"
    assert status["q_arch"] is None
    assert status["q_recipe"] is None
    assert "broker is unavailable" in status["error"]
    assert row["status"] == "infra_failed"
    assert row["infra_retryable"] == 1
    assert "broker is unavailable" in row["error"]


def test_full_scale_spec_includes_frozen_dataset_resource_and_phase_targets(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        artifact_mount = next(mount for mount in spec.mounts if mount.target == "/artifacts")
        (artifact_mount.source / RUN_MANIFEST_FILENAME).write_text(
            json.dumps(_gpu_manifest(ExecutionMode.FULL_SCALE_EVAL), separators=(",", ":")),
            encoding="utf-8",
        )
        captured["payload"] = json.loads((spec.mounts[0].source / "payload.json").read_text())
        captured["spec"] = spec
        captured["timeout_seconds"] = timeout_seconds
        return DockerRunResult("platform-full-scale-job", "", "", 0)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "platform-v10-full-scale.sqlite3"
    with TestClient(create_app(_settings(db_path))) as client:
        submission_id = _submit(
            client,
            REMOTE_ONLY_CODE,
            nonce="full-scale-spec",
            metadata={"execution_mode": ExecutionMode.FULL_SCALE_EVAL.value},
        )
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    payload = captured["payload"]
    mode_spec = payload["mode_spec"]
    row = _eval_job_row(db_path, submission_id)

    assert status["status"] == "completed"
    assert captured["timeout_seconds"] == 900
    assert captured["spec"].env["PRISM_EXECUTION_MODE"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert captured["spec"].labels["prism.execution_mode"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert payload["execution_mode"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert mode_spec["mode"] == ExecutionMode.FULL_SCALE_EVAL.value
    assert mode_spec["official_score_eligible"] is True
    assert mode_spec["token_budget"] == 10_000_000_000
    assert mode_spec["parameter_target"] == 150_000_000
    assert mode_spec["dataset"]["revision"] == "fineweb-edu-contract-2026-05-25"
    assert mode_spec["dataset"]["subset"] == "sample-100BT"
    assert mode_spec["dataset"]["token_count"] == 100_000_000_000
    assert mode_spec["dataset"]["network_fallback_allowed"] is False
    assert len(mode_spec["dataset"]["train_split_fingerprint"]) == 64
    assert mode_spec["resource_profile"] == {
        "profile": "fixed_official_gpu",
        "cpus": 2.0,
        "memory": "8g",
        "gpu_count": 2,
        "max_gpu_count": 4,
        "gpu_type": "l4",
        "gpu_server": "server-a",
        "gpu_device_ids": ["0", "1"],
        "official_fixed_profile": True,
    }
    assert mode_spec["artifact_output_path"] == "/artifacts"
    assert mode_spec["run_manifest_path"] == f"/artifacts/{RUN_MANIFEST_FILENAME}"
    assert mode_spec["phases"] == [
        {
            "name": "full_scale_10b_tokens",
            "token_budget": 10_000_000_000,
            "parameter_target": 150_000_000,
            "dataset_subset": "sample-10BT",
        },
        {
            "name": "phase_2_1b_params_100b_tokens",
            "token_budget": 100_000_000_000,
            "parameter_target": 1_000_000_000,
            "dataset_subset": "sample-100BT",
        },
    ]
    assert row["status"] == "completed"
    assert row["run_manifest_path"] == f"/artifacts/{RUN_MANIFEST_FILENAME}"
