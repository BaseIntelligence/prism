from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

DiagnosticStatus = Literal["ok", "warning", "not_applicable"]

REQUIRED_DIAGNOSTICS = (
    "activation_entropy",
    "useful_sparsity",
    "attention_diversity",
    "representation_collapse",
    "gradient_noise_scale",
    "activation_norm_stability",
    "neuron_specialization",
)

DEFAULT_DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "activation_entropy_min": 0.05,
    "dead_sparsity_max": 0.98,
    "gradient_noise_max": 10.0,
    "min_attention_diversity": 0.05,
    "representation_collapse_max": 0.95,
    "activation_norm_spike_ratio": 3.0,
    "activation_norm_stability_min": 0.5,
}

SQL_DIAGNOSTIC_REDISTRIBUTION_POLICY_KEY = (
    "loss_comparability_policy.redistribution_policy"
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_exp_neg(value: float) -> float:
    return clamp(math.exp(-max(0.0, min(value, 20.0))))


def harmonic_mean(values: Iterable[float]) -> float:
    vals = [max(0.0, v) for v in values]
    positives = [v for v in vals if v > 0]
    if not positives or len(positives) != len(vals):
        return 0.0
    return len(positives) / sum(1.0 / v for v in positives)


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


@dataclass(frozen=True)
class DiagnosticRedistribution:
    enabled: bool
    policy_key: str | None = None
    target: str | None = None
    reason: str | None = None

    def as_manifest_record(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "policy_key": self.policy_key,
            "target": self.target,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DiagnosticMetric:
    name: str
    status: DiagnosticStatus
    aggregate: float | None
    per_layer: dict[str, float]
    warnings: tuple[str, ...] = ()
    not_applicable_reason: str | None = None
    redistribution: DiagnosticRedistribution | None = None

    def as_manifest_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "aggregate": self.aggregate,
            "per_layer": dict(self.per_layer),
            "warnings": list(self.warnings),
            "not_applicable_reason": self.not_applicable_reason,
            "redistribution": (
                self.redistribution.as_manifest_record() if self.redistribution else None
            ),
        }


TensorCollection = Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
ActivationSeries = Mapping[str, torch.Tensor | Sequence[torch.Tensor]]
GradientSample = Mapping[str, torch.Tensor] | torch.Tensor


def diagnostics_to_manifest(
    diagnostics: Mapping[str, DiagnosticMetric],
) -> dict[str, dict[str, Any]]:
    return {name: metric.as_manifest_record() for name, metric in diagnostics.items()}


def collect_diagnostics(
    *,
    activations: TensorCollection,
    representations: torch.Tensor | TensorCollection,
    gradient_samples: Sequence[GradientSample],
    attention_weights: TensorCollection | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, DiagnosticMetric]:
    active_thresholds = _thresholds(thresholds)
    return {
        "activation_entropy": collect_activation_entropy(
            activations, thresholds=active_thresholds
        ),
        "useful_sparsity": collect_useful_sparsity(
            activations, thresholds=active_thresholds
        ),
        "attention_diversity": collect_attention_diversity(
            attention_weights, thresholds=active_thresholds
        ),
        "representation_collapse": collect_representation_collapse(
            representations, thresholds=active_thresholds
        ),
        "gradient_noise_scale": collect_gradient_noise_scale(
            gradient_samples, thresholds=active_thresholds
        ),
        "activation_norm_stability": collect_activation_norm_stability(
            _activation_series(activations), thresholds=active_thresholds
        ),
        "neuron_specialization": collect_neuron_specialization(
            activations, thresholds=active_thresholds
        ),
    }


def collect_activation_entropy(
    activations: TensorCollection,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> DiagnosticMetric:
    active_thresholds = _thresholds(thresholds)
    per_layer: dict[str, float] = {}
    warnings: list[str] = []
    for name, tensor in _named_tensors(activations).items():
        values = _finite_flatten(tensor, name)
        weights = values.abs()
        total = float(weights.sum().item())
        if total <= 0.0:
            entropy = 0.0
        elif values.numel() == 1:
            entropy = 1.0
        else:
            probabilities = weights / total
            entropy_tensor = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
            entropy = float((entropy_tensor / math.log(values.numel())).item())
        per_layer[name] = clamp(entropy)
    aggregate = _mean(per_layer.values())
    if aggregate < active_thresholds["activation_entropy_min"]:
        warnings.append("dead_architecture_low_activation_entropy")
    return _metric("activation_entropy", aggregate, per_layer, warnings)


def collect_useful_sparsity(
    activations: TensorCollection,
    *,
    zero_threshold: float = 1e-6,
    thresholds: Mapping[str, float] | None = None,
) -> DiagnosticMetric:
    active_thresholds = _thresholds(thresholds)
    per_layer: dict[str, float] = {}
    warnings: list[str] = []
    for name, tensor in _named_tensors(activations).items():
        values = _finite_flatten(tensor, name)
        per_layer[name] = float((values.abs() <= zero_threshold).float().mean().item())
    aggregate = _mean(per_layer.values())
    if aggregate > active_thresholds["dead_sparsity_max"]:
        warnings.append("dead_architecture_excessive_sparsity")
    return _metric("useful_sparsity", aggregate, per_layer, warnings)


def collect_attention_diversity(
    attention_weights: TensorCollection | None,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> DiagnosticMetric:
    if not attention_weights:
        return not_applicable_metric(
            "attention_diversity",
            "architecture exposes no attention weights",
        )
    active_thresholds = _thresholds(thresholds)
    per_layer: dict[str, float] = {}
    warnings: list[str] = []
    for name, tensor in _named_tensors(attention_weights).items():
        values = _finite_tensor(tensor, name).float().abs()
        if values.ndim < 2 or values.shape[-1] < 2:
            return not_applicable_metric(
                "attention_diversity",
                "attention diversity requires a key dimension of at least two",
            )
        probabilities = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
        normalized = entropy / math.log(values.shape[-1])
        per_layer[name] = clamp(float(normalized.mean().item()))
    aggregate = _mean(per_layer.values())
    if aggregate < active_thresholds["min_attention_diversity"]:
        warnings.append("attention_saturation")
    return _metric("attention_diversity", aggregate, per_layer, warnings)


def collect_representation_collapse(
    representations: torch.Tensor | TensorCollection,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> DiagnosticMetric:
    active_thresholds = _thresholds(thresholds)
    tensors = (
        {"representations": representations}
        if isinstance(representations, torch.Tensor)
        else _named_tensors(representations)
    )
    per_layer: dict[str, float] = {}
    warnings: list[str] = []
    for name, tensor in tensors.items():
        values = _finite_tensor(tensor, name).float()
        if values.shape[-1] < 1:
            raise ValueError(f"{name} must have a non-empty feature dimension")
        rows = values.reshape(-1, values.shape[-1])
        if rows.shape[0] < 2:
            per_layer[name] = 1.0
            warnings.append("representation_collapse_insufficient_samples")
            continue
        normalized = torch.nn.functional.normalize(rows, dim=-1, eps=1e-12)
        similarity = normalized @ normalized.T
        off_diagonal = similarity[~torch.eye(rows.shape[0], dtype=torch.bool, device=rows.device)]
        per_layer[name] = clamp(float(off_diagonal.abs().mean().item()))
    aggregate = _mean(per_layer.values())
    if aggregate > active_thresholds["representation_collapse_max"]:
        warnings.append("representation_collapse")
    return _metric("representation_collapse", aggregate, per_layer, warnings)


def collect_gradient_noise_scale(
    gradient_samples: Sequence[GradientSample],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> DiagnosticMetric:
    if len(gradient_samples) < 2:
        raise ValueError("gradient noise scale requires at least two gradient samples")
    active_thresholds = _thresholds(thresholds)
    vectors = [
        _flatten_gradient_sample(sample, index)
        for index, sample in enumerate(gradient_samples)
    ]
    first_numel = vectors[0].numel()
    if any(vector.numel() != first_numel for vector in vectors):
        raise ValueError("gradient samples must flatten to the same length")
    stacked = torch.stack(vectors)
    mean_gradient = stacked.mean(dim=0)
    residual = stacked - mean_gradient
    numerator = float(residual.pow(2).mean().item())
    denominator = float(mean_gradient.pow(2).mean().item()) + 1e-12
    scale = numerator / denominator
    warnings = []
    if scale > active_thresholds["gradient_noise_max"]:
        warnings.append("gradient_noise_scale_high")
    return _metric("gradient_noise_scale", scale, {"all": scale}, warnings)


def collect_activation_norm_stability(
    activation_series: ActivationSeries,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> DiagnosticMetric:
    active_thresholds = _thresholds(thresholds)
    per_layer: dict[str, float] = {}
    warnings: list[str] = []
    for name, series_or_tensor in activation_series.items():
        tensors = (
            [series_or_tensor]
            if isinstance(series_or_tensor, torch.Tensor)
            else list(series_or_tensor)
        )
        if not tensors:
            raise ValueError(f"{name} activation series is empty")
        norms = [float(_finite_tensor(tensor, name).float().norm().item()) for tensor in tensors]
        mean_norm = sum(norms) / len(norms)
        coefficient = math.sqrt(variance(norms)) / max(mean_norm, 1e-12)
        stability = 1.0 / (1.0 + coefficient)
        per_layer[name] = clamp(stability)
        positive_norms = [norm for norm in norms if norm > 0]
        if positive_norms:
            spike_ratio = max(positive_norms) / max(min(positive_norms), 1e-12)
            if spike_ratio > active_thresholds["activation_norm_spike_ratio"]:
                warnings.append(f"activation_norm_spike:{name}")
    aggregate = _mean(per_layer.values())
    if aggregate < active_thresholds["activation_norm_stability_min"]:
        warnings.append("activation_norm_unstable")
    return _metric("activation_norm_stability", aggregate, per_layer, warnings)


def collect_neuron_specialization(
    activations: TensorCollection,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> DiagnosticMetric:
    _thresholds(thresholds)
    per_layer: dict[str, float] = {}
    for name, tensor in _named_tensors(activations).items():
        values = _finite_tensor(tensor, name).float().abs()
        if values.ndim < 1 or values.shape[-1] < 2:
            return not_applicable_metric(
                "neuron_specialization",
                "neuron specialization requires at least two feature channels",
            )
        rows = values.reshape(-1, values.shape[-1])
        neuron_usage = rows.mean(dim=0)
        mean_usage = float(neuron_usage.mean().item())
        if mean_usage <= 1e-12:
            specialization = 0.0
        else:
            specialization = float((neuron_usage.std(unbiased=False) / mean_usage).item())
        per_layer[name] = clamp(specialization)
    aggregate = _mean(per_layer.values())
    warnings = ["memory_collapse_uniform_neuron_usage"] if aggregate <= 0.01 else []
    return _metric("neuron_specialization", aggregate, per_layer, warnings)


def not_applicable_metric(name: str, reason: str) -> DiagnosticMetric:
    return DiagnosticMetric(
        name=name,
        status="not_applicable",
        aggregate=None,
        per_layer={},
        not_applicable_reason=reason,
        redistribution=DiagnosticRedistribution(
            enabled=True,
            policy_key=SQL_DIAGNOSTIC_REDISTRIBUTION_POLICY_KEY,
            target="diagnostics_health",
            reason=reason,
        ),
    )


def _metric(
    name: str, aggregate: float, per_layer: dict[str, float], warnings: Sequence[str]
) -> DiagnosticMetric:
    if not math.isfinite(aggregate):
        raise ValueError(f"{name} aggregate is non-finite")
    if not per_layer:
        raise ValueError(f"{name} must include at least one diagnostic value")
    return DiagnosticMetric(
        name=name,
        status="warning" if warnings else "ok",
        aggregate=aggregate,
        per_layer=per_layer,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _thresholds(overrides: Mapping[str, float] | None) -> dict[str, float]:
    thresholds = dict(DEFAULT_DIAGNOSTIC_THRESHOLDS)
    if overrides:
        thresholds.update({key: float(value) for key, value in overrides.items()})
    return thresholds


def _named_tensors(tensors: TensorCollection) -> dict[str, torch.Tensor]:
    if isinstance(tensors, Mapping):
        named = {str(name): tensor for name, tensor in tensors.items()}
    else:
        named = {f"layer_{index}": tensor for index, tensor in enumerate(tensors)}
    if not named:
        raise ValueError("diagnostic tensor collection is empty")
    return named


def _activation_series(activations: TensorCollection) -> ActivationSeries:
    if isinstance(activations, Mapping):
        return {name: tensor for name, tensor in activations.items()}
    return {f"layer_{index}": tensor for index, tensor in enumerate(activations)}


def _finite_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.numel() == 0:
        raise ValueError(f"{name} diagnostic tensor is empty")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} diagnostic tensor contains non-finite values")
    return tensor.detach().cpu()


def _finite_flatten(tensor: torch.Tensor, name: str) -> torch.Tensor:
    return _finite_tensor(tensor, name).float().reshape(-1)


def _flatten_gradient_sample(sample: GradientSample, index: int) -> torch.Tensor:
    if isinstance(sample, torch.Tensor):
        return _finite_flatten(sample, f"gradient_sample_{index}")
    tensors = _named_tensors(sample)
    return torch.cat(
        [
            _finite_flatten(tensor, f"gradient_sample_{index}:{name}")
            for name, tensor in tensors.items()
        ]
    )


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("cannot aggregate empty diagnostic values")
    return sum(vals) / len(vals)
