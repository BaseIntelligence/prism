"""Validator dispatch entrypoint for prism (architecture sec 4, G2).

A pulled prism gpu work unit, dispatched by the platform validator agent via
:func:`prism_challenge.validator_dispatch.dispatch_assignment`, runs the GPU
re-execution on the validator's OWN broker (exercised here via the CPU re-exec
mock). These tests lock the dispatch contract: the re-exec container runs
``network=none`` (single-node), concurrency 1 is enforced against the validator's
real in-flight draw, re-running a completed unit is an idempotent no-op, the LLM
review routes through the master gateway scoped token with the raw provider key
stripped, and a payload missing the token NEVER reaches the broker.
"""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.models import SubmissionCreate
from prism_challenge.sdk.executors.docker import DockerRunSpec
from prism_challenge.validator_dispatch import (
    PrismGatewayConfigError,
    dispatch_assignment,
    gateway_scoped_settings,
)

GATEWAY_BASE_URL = "http://master:8081"
GATEWAY_OPENROUTER_URL = f"{GATEWAY_BASE_URL}/llm/openrouter"
GATEWAY_TOKEN = "scoped-token"
BROKER_URL = "http://broker-val:8082"
BROKER_TOKEN = "val-secret"

TINY_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

TINY_TRAIN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

_SHARD_LINE = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


def _stage_train(root: Path, *, lines: int = 64) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD_LINE.format(i=i) for i in range(lines)), encoding="utf-8"
    )
    return data_dir


def _two_script_bundle(arch: str = TINY_ARCH, train: str = TINY_TRAIN) -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", arch)
        archive.writestr("training.py", train)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _settings(tmp_path: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'coord.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        llm_review_enabled=False,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
    )


def _payload(*, with_token: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "gateway_url": GATEWAY_BASE_URL,
        "OPENROUTER_BASE_URL": GATEWAY_OPENROUTER_URL,
    }
    if with_token:
        payload["gateway_token"] = GATEWAY_TOKEN
    return payload


async def _seed(settings: PrismSettings, hotkeys: list[str]) -> list[str]:
    app = create_app(settings)
    await app.state.database.init()
    ids: list[str] = []
    for hotkey in hotkeys:
        sub = await app.state.repository.create_submission(
            hotkey, SubmissionCreate(code=_two_script_bundle(), filename="project.zip")
        )
        ids.append(sub.id)
    await app.state.database.close()
    return ids


async def test_dispatch_runs_gpu_reexec_network_none(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    settings = _settings(tmp_path)
    [submission_id] = await _seed(settings, ["hk-a"])

    result = await dispatch_assignment(
        work_unit_id=submission_id,
        payload=_payload(),
        broker_url=BROKER_URL,
        broker_token=BROKER_TOKEN,
        settings=settings,
    )

    assert result["executed"] == 1
    # The validator's OWN broker ran the re-execution network-isolated, single-node.
    assert len(captured) == 1
    assert captured[0].limits.network == "none"
    assert "--nproc-per-node=1" in captured[0].command


async def test_dispatch_repost_is_idempotent(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    settings = _settings(tmp_path)
    [submission_id] = await _seed(settings, ["hk-a"])

    first = await dispatch_assignment(
        work_unit_id=submission_id,
        payload=_payload(),
        broker_url=BROKER_URL,
        settings=settings,
    )
    assert first["executed"] == 1
    runs_after_first = len(captured)

    # Re-dispatch the now-completed unit: no second broker run.
    second = await dispatch_assignment(
        work_unit_id=submission_id,
        payload=_payload(),
        broker_url=BROKER_URL,
        settings=settings,
    )
    assert second["pulled"] == 0
    assert second["executed"] == 0
    assert len(captured) == runs_after_first


async def test_dispatch_enforces_concurrency_one(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    settings = _settings(tmp_path)
    sub_a, sub_b = await _seed(settings, ["hk-a", "hk-b"])

    # Mark sub_a in-flight (a concurrent prism run already owned by this validator).
    app = create_app(settings)
    await app.state.database.init()
    claimed = await app.state.repository.claim_submission(sub_a)
    assert claimed is not None
    assert await app.state.repository.count_in_flight_submissions() == 1
    await app.state.database.close()

    # With one in-flight unit, dispatching the second pulls nothing (concurrency 1).
    result = await dispatch_assignment(
        work_unit_id=sub_b,
        payload=_payload(),
        broker_url=BROKER_URL,
        settings=settings,
    )
    assert result["pulled"] == 0
    assert result["executed"] == 0
    assert captured == []


def test_gateway_scoped_settings_strips_provider_key(tmp_path):
    settings = _settings(tmp_path).model_copy(update={"openrouter_api_key": "raw-provider-key"})
    effective = gateway_scoped_settings(
        settings, _payload(), broker_url=BROKER_URL, broker_token=BROKER_TOKEN
    )

    # The prism LLM review routes through the master gateway scoped token; the raw
    # provider key is stripped so it never reaches the validator/eval host.
    assert effective.llm_gateway_token == GATEWAY_TOKEN
    assert effective.llm_gateway_url == GATEWAY_OPENROUTER_URL
    assert effective.openrouter_api_key is None
    assert effective.openrouter_api_key_file is None
    # The re-execution is dispatched to the validator's OWN broker.
    assert effective.docker_broker_url == BROKER_URL
    assert effective.docker_broker_token == BROKER_TOKEN


def test_gateway_scoped_settings_derives_openrouter_url_from_base(tmp_path):
    # A payload carrying only the gateway base URL still yields the openrouter
    # route (never gateway=None).
    payload = {"gateway_token": GATEWAY_TOKEN, "gateway_url": GATEWAY_BASE_URL}
    effective = gateway_scoped_settings(_settings(tmp_path), payload, broker_url=BROKER_URL)
    assert effective.llm_gateway_url == GATEWAY_OPENROUTER_URL


async def test_dispatch_missing_gateway_token_never_reaches_broker(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    settings = _settings(tmp_path)
    [submission_id] = await _seed(settings, ["hk-a"])

    with pytest.raises(PrismGatewayConfigError):
        await dispatch_assignment(
            work_unit_id=submission_id,
            payload=_payload(with_token=False),
            broker_url=BROKER_URL,
            settings=settings,
        )
    assert captured == []
