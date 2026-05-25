from __future__ import annotations

import json

import pytest

from prism_challenge.evaluator.dataset import FineWebEduConfig, load_fineweb_edu_contract
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.metrics import REQUIRED_DIAGNOSTICS
from prism_challenge.evaluator.modes import run_local_cpu_smoke
from prism_challenge.evaluator.schemas import (
    ExecutionMode,
    PrismRunManifest,
    validate_run_manifest_for_official_scoring,
)

TINY_MODEL_CODE = """
import torch
from prism_challenge.evaluator.interface import TrainingRecipe

class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 8)
        self.proj = torch.nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.proj(self.emb(tokens))

def build_model(ctx):
    return TinyLanguageModel(ctx.vocab_size)

def get_recipe(ctx):
    return TrainingRecipe(learning_rate=0.001, batch_size=1)
"""


def test_local_cpu_smoke_writes_required_manifest(tmp_path) -> None:
    result = run_local_cpu_smoke(
        submission_id="smoke-submission-1",
        code=TINY_MODEL_CODE,
        code_hash="1" * 64,
        arch_hash="2" * 64,
        ctx=PrismContext(vocab_size=256, sequence_length=16, max_parameters=50_000),
        artifact_output_path=tmp_path / "artifacts",
    )

    manifest_path = tmp_path / "artifacts" / "prism_run_manifest.v1.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = PrismRunManifest.model_validate(payload)
    fixture_contract = load_fineweb_edu_contract(
        FineWebEduConfig(mode=ExecutionMode.LOCAL_CPU_SMOKE)
    )

    assert result.run_manifest_path == str(manifest_path)
    assert result.artifact_output_path == str(tmp_path / "artifacts")
    assert manifest.mode is ExecutionMode.LOCAL_CPU_SMOKE
    assert manifest.validation.passed is True
    assert manifest.validation.score_eligible is False
    assert manifest.compute.gpu_count == 0
    assert manifest.metrics.tokens_seen > 0
    assert manifest.metrics.parameter_count == manifest.model.parameter_count
    assert set(manifest.metrics.diagnostics) == set(REQUIRED_DIAGNOSTICS)
    assert manifest.metrics.loss.loss_comparable is True
    assert manifest.metrics.loss.loss_normalization_scope == "byte_normalized"
    assert manifest.metrics.benchmark_capability_metadata["status"] == "not_run"
    assert manifest.dataset.train_split_fingerprint == fixture_contract.split_fingerprints["train"]
    assert (
        manifest.dataset.validation_split_fingerprint
        == fixture_contract.split_fingerprints["validation"]
    )
    assert manifest.dataset.test_split_fingerprint == fixture_contract.split_fingerprints["test"]
    assert result.metrics["gpu_count"] == 0.0

    with pytest.raises(ValueError, match="local_cpu_smoke"):
        validate_run_manifest_for_official_scoring(payload)
