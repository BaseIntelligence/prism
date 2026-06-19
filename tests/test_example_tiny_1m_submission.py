from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.modes import run_local_cpu_smoke
from prism_challenge.evaluator.sandbox import SandboxReport, SandboxViolation, inspect_code
from prism_challenge.evaluator.schemas import (
    RUN_MANIFEST_FILENAME,
    ExecutionMode,
    PrismRunManifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = REPO_ROOT / "examples" / "tiny-1m" / "submission_singlefile.py"

REQUIRED_CONTRACT_FUNCTIONS = ("build_model", "get_recipe")

# The optimized hooks the tiny submission must expose (the "training recipe/hooks").
REQUIRED_OPTIONAL_HOOKS = (
    "configure_optimizer",
    "compute_loss",
    "train_step",
    "inference_logits",
)

# The GPU-path context (vocab=4096) used to assert the real ~1.05M parameter count.
GPU_CTX = PrismContext()

# Smoke context: tiny vocab/sequence so a genuine-but-small model trains on CPU.
SMOKE_CTX = PrismContext(vocab_size=256, sequence_length=16, max_parameters=20_000_000)


def _read_submission_source() -> str:
    assert SUBMISSION_PATH.exists(), (
        f"tiny 1M single-file submission not found at {SUBMISSION_PATH}"
    )
    return SUBMISSION_PATH.read_text(encoding="utf-8")


def _load_submission_module() -> ModuleType:
    source = _read_submission_source()
    assert source.strip(), "tiny 1M single-file submission must not be empty"
    spec = importlib.util.spec_from_file_location("tiny_1m_submission_singlefile", SUBMISSION_PATH)
    assert spec is not None and spec.loader is not None, "submission must be importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_contract_functions_present_and_callable() -> None:
    module = _load_submission_module()
    for name in REQUIRED_CONTRACT_FUNCTIONS:
        assert hasattr(module, name), f"submission is missing required function: {name}"
        assert callable(getattr(module, name)), f"{name} must be callable"


def test_optimized_hooks_present_and_callable() -> None:
    module = _load_submission_module()
    for name in REQUIRED_OPTIONAL_HOOKS:
        assert hasattr(module, name), f"submission is missing optimized hook: {name}"
        assert callable(getattr(module, name)), f"{name} must be callable"


def test_submission_is_sandbox_clean() -> None:
    source = _read_submission_source()
    try:
        report = inspect_code(source)
    except SandboxViolation as violation:  # pragma: no cover - failure path
        pytest.fail(f"tiny 1M submission must be sandbox-clean, got: {violation}")
    assert isinstance(report, SandboxReport)
    allowlist = {"collections", "dataclasses", "math", "prism_challenge", "torch", "typing"}
    assert report.imports <= allowlist
    for name in REQUIRED_CONTRACT_FUNCTIONS:
        assert f"function:{name}" in report.ast_fingerprint


def test_parameter_count_is_about_one_million() -> None:
    module = _load_submission_module()
    model = module.build_model(GPU_CTX)
    params = sum(p.numel() for p in model.parameters())
    assert 0 < params <= GPU_CTX.max_parameters, f"params {params} must be in (0, 150M]"
    assert 1_000_000 <= params <= 1_100_000, f"expected ~1.05M params, got {params}"


def test_recipe_learning_rate_in_quality_window() -> None:
    module = _load_submission_module()
    recipe = module.get_recipe(GPU_CTX)
    assert 1e-5 <= recipe.learning_rate <= 3e-3, (
        f"recipe LR {recipe.learning_rate} must stay in the q_recipe window [1e-5, 3e-3]"
    )


def test_local_cpu_smoke_manifest_shows_learning(tmp_path: Path) -> None:
    source = _read_submission_source()
    artifact_output_path = tmp_path / "artifacts"
    result = run_local_cpu_smoke(
        submission_id="tiny-1m-singlefile-smoke",
        code=source,
        code_hash="a" * 64,
        arch_hash="b" * 64,
        ctx=SMOKE_CTX,
        artifact_output_path=artifact_output_path,
    )

    manifest_path = artifact_output_path / RUN_MANIFEST_FILENAME
    assert result.run_manifest_path == str(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = PrismRunManifest.model_validate(payload)

    assert manifest.mode is ExecutionMode.LOCAL_CPU_SMOKE
    assert manifest.validation.passed is True
    assert manifest.validation.score_eligible is False

    initial_loss = manifest.metrics.loss_vs_tokens[0].loss
    final_loss = manifest.metrics.final_loss
    assert final_loss is not None, "manifest must report a final_loss"
    assert final_loss < initial_loss, (
        f"expected final_loss ({final_loss}) < initial_loss ({initial_loss})"
    )
