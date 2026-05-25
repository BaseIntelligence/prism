from __future__ import annotations

import math
from copy import deepcopy

import pytest
import torch
from pydantic import ValidationError
from test_artifact_manifest import _valid_diagnostics, _valid_manifest

from prism_challenge.evaluator.metrics import (
    REQUIRED_DIAGNOSTICS,
    collect_activation_norm_stability,
    collect_diagnostics,
    collect_representation_collapse,
    diagnostics_to_manifest,
)
from prism_challenge.evaluator.schemas import (
    DiagnosticRecord,
    ExecutionMode,
    PrismRunManifest,
    validate_run_manifest_for_official_scoring,
)


def _activations() -> dict[str, torch.Tensor]:
    return {
        "embed": torch.tensor(
            [[0.0, 1.0, 3.0, 0.0], [4.0, 0.0, 1.0, 0.5]], dtype=torch.float32
        ),
        "mlp": torch.tensor(
            [[1.0, 0.0, 2.0, 1.0], [0.0, 3.0, 0.0, 2.0]], dtype=torch.float32
        ),
    }


def _representations() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )


def _gradient_samples() -> list[torch.Tensor]:
    return [
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        torch.tensor([1.1, 1.9, 3.2, 3.8], dtype=torch.float32),
        torch.tensor([0.9, 2.1, 2.8, 4.2], dtype=torch.float32),
    ]


def _uniform_attention() -> dict[str, torch.Tensor]:
    return {"attn": torch.full((1, 2, 4, 4), 0.25, dtype=torch.float32)}


def test_emits_all_required_diagnostics() -> None:
    diagnostics = collect_diagnostics(
        activations=_activations(),
        representations=_representations(),
        gradient_samples=_gradient_samples(),
        attention_weights=_uniform_attention(),
    )

    assert set(diagnostics) == set(REQUIRED_DIAGNOSTICS)
    for metric in diagnostics.values():
        record = DiagnosticRecord.model_validate(metric.as_manifest_record())
        assert record.status in {"ok", "warning"}
        assert record.aggregate is not None
        assert math.isfinite(record.aggregate)

    payload = _valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value)
    payload["metrics"]["diagnostics"] = diagnostics_to_manifest(diagnostics)

    manifest = validate_run_manifest_for_official_scoring(payload)
    assert set(manifest.metrics.diagnostics) == set(REQUIRED_DIAGNOSTICS)


def test_attention_saturation_warns_without_rejection() -> None:
    saturated_attention = {"attn": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4)}
    diagnostics = collect_diagnostics(
        activations=_activations(),
        representations=_representations(),
        gradient_samples=_gradient_samples(),
        attention_weights=saturated_attention,
    )

    attention = diagnostics["attention_diversity"]
    assert attention.status == "warning"
    assert "attention_saturation" in attention.warnings

    payload = _valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value)
    payload["metrics"]["diagnostics"] = diagnostics_to_manifest(diagnostics)
    manifest = validate_run_manifest_for_official_scoring(payload)
    assert manifest.metrics.diagnostics["attention_diversity"].status == "warning"


def test_non_attention_models_mark_attention_not_applicable() -> None:
    diagnostics = collect_diagnostics(
        activations=_activations(),
        representations=_representations(),
        gradient_samples=_gradient_samples(),
        attention_weights=None,
    )

    attention = diagnostics["attention_diversity"]
    assert attention.status == "not_applicable"
    assert attention.aggregate is None
    assert attention.redistribution is not None
    assert attention.redistribution.enabled is True

    payload = _valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value)
    payload["metrics"]["diagnostics"] = diagnostics_to_manifest(diagnostics)
    manifest = validate_run_manifest_for_official_scoring(payload)
    assert manifest.metrics.diagnostics["attention_diversity"].aggregate is None


def test_collapse_and_activation_norm_warnings_are_health_signals() -> None:
    collapsed = collect_representation_collapse(torch.ones((4, 3), dtype=torch.float32))
    unstable = collect_activation_norm_stability(
        {
            "residual": [
                torch.ones((2, 3), dtype=torch.float32),
                torch.ones((2, 3), dtype=torch.float32) * 20.0,
            ]
        }
    )

    assert collapsed.status == "warning"
    assert "representation_collapse" in collapsed.warnings
    assert unstable.status == "warning"
    assert any(warning.startswith("activation_norm_spike") for warning in unstable.warnings)


def test_non_finite_empty_or_unredistributed_diagnostics_fail_manifest_validation() -> None:
    payload = _valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value)
    payload["metrics"]["diagnostics"] = {}
    with pytest.raises(ValidationError, match="at least 1"):
        PrismRunManifest.model_validate(payload)

    payload = _valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value)
    diagnostics = _valid_diagnostics()
    diagnostics["activation_entropy"] = deepcopy(diagnostics["activation_entropy"])
    diagnostics["activation_entropy"]["aggregate"] = float("nan")
    payload["metrics"]["diagnostics"] = diagnostics
    with pytest.raises(ValidationError, match="finite"):
        PrismRunManifest.model_validate(payload)

    payload = _valid_manifest(ExecutionMode.GPU_PROXY_EVAL.value)
    diagnostics = _valid_diagnostics()
    diagnostics["attention_diversity"] = deepcopy(diagnostics["attention_diversity"])
    diagnostics["attention_diversity"].pop("redistribution")
    payload["metrics"]["diagnostics"] = diagnostics
    with pytest.raises(ValidationError, match="redistribution"):
        PrismRunManifest.model_validate(payload)
