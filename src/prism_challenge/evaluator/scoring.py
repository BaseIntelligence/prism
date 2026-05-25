from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from prism_challenge.runtime_config import BenchmarkWeights, ScoreWeights

from .benchmarks.official import benchmark_sanity_component
from .interface import TrainingRecipe
from .schemas import PrismRunManifest

ArchitectureComponentName = Literal[
    "learning_scaling_dynamics",
    "standardized_lm_quality",
    "compute_efficiency",
    "parameter_efficiency",
    "diagnostics_health",
    "robustness_stability",
    "benchmark_sanity",
]
TrainingComponentName = Literal[
    "architecture_normalized_heldout_improvement",
    "learning_stability_dynamics",
    "benchmark_sanity",
    "compute_efficiency",
    "reproducibility_stability",
    "robustness_failure_behavior",
    "artifact_completeness",
]

ARCHITECTURE_SCORE_COMPONENTS: dict[str, float] = {
    "learning_scaling_dynamics": 0.35,
    "standardized_lm_quality": 0.20,
    "compute_efficiency": 0.15,
    "parameter_efficiency": 0.10,
    "diagnostics_health": 0.10,
    "robustness_stability": 0.05,
    "benchmark_sanity": 0.05,
}
TRAINING_SCORE_COMPONENTS: dict[str, float] = {
    "architecture_normalized_heldout_improvement": 0.30,
    "learning_stability_dynamics": 0.25,
    "benchmark_sanity": 0.15,
    "compute_efficiency": 0.10,
    "reproducibility_stability": 0.10,
    "robustness_failure_behavior": 0.05,
    "artifact_completeness": 0.05,
}


class ScoreValidationError(ValueError):
    def __init__(self, reasons: list[str] | tuple[str, ...]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class ScoreResult:
    q_arch: float
    q_recipe: float
    final_score: float


@dataclass(frozen=True)
class ScoreComponentDetail:
    name: str
    weight: float
    raw_value: float
    weighted_contribution: float
    source: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "weight": self.weight,
            "raw_value": self.raw_value,
            "weighted_contribution": self.weighted_contribution,
            "source": self.source,
        }


@dataclass(frozen=True)
class OfficialScoreResult:
    track: Literal["architecture", "training"]
    score: float
    components: tuple[ScoreComponentDetail, ...]
    details: dict[str, Any] = field(default_factory=dict)
    missing_reasons: tuple[str, ...] = ()

    @property
    def component_weights(self) -> dict[str, float]:
        return {component.name: component.weight for component in self.components}

    @property
    def component_values(self) -> dict[str, float]:
        return {component.name: component.raw_value for component in self.components}

    def as_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "score": self.score,
            "components": [component.as_dict() for component in self.components],
            "details": self.details,
            "missing_reasons": list(self.missing_reasons),
        }


@dataclass(frozen=True)
class RankedScore:
    submission_id: str
    score: float
    compute: float
    accepted_at: str


def score_recipe(recipe: TrainingRecipe) -> float:
    lr_score = 1.0 if 1e-5 <= recipe.learning_rate <= 3e-3 else 0.4
    batch_score = 1.0 if 1 <= recipe.batch_size <= 64 else 0.5
    opt_score = 1.0 if recipe.optimizer.lower() in {"adamw", "adam", "sgd"} else 0.5
    return max(0.0, min(1.0, 0.45 * lr_score + 0.35 * batch_score + 0.2 * opt_score))


def score_architecture_manifest(
    manifest: PrismRunManifest | dict[str, Any],
    *,
    score_weights: ScoreWeights | None = None,
    benchmark_weights: BenchmarkWeights | None = None,
) -> OfficialScoreResult:
    run_manifest = _official_manifest(manifest)
    weights = _architecture_formula(score_weights)
    benchmark_component = benchmark_sanity_component(
        run_manifest.metrics.benchmark_scores,
        benchmark_weights or BenchmarkWeights(),
        _score_weights(weights, TRAINING_SCORE_COMPONENTS),
        track="architecture",
    )
    component_values: dict[str, tuple[float, str]] = {
        "learning_scaling_dynamics": (
            _learning_scaling_dynamics(run_manifest),
            "loss.relative_loss_reduction + metrics.learning_speed_slope",
        ),
        "standardized_lm_quality": (
            _standardized_lm_quality(run_manifest),
            "loss.standardized_eval_loss",
        ),
        "compute_efficiency": (_compute_efficiency(run_manifest), "metrics.estimated_flops"),
        "parameter_efficiency": (_parameter_efficiency(run_manifest), "metrics.parameter_count"),
        "diagnostics_health": (_diagnostics_health(run_manifest), "metrics.diagnostics"),
        "robustness_stability": (_robustness_stability(run_manifest), "metrics.loss_vs_tokens"),
        "benchmark_sanity": (benchmark_component.raw_score, "metrics.benchmark_scores"),
    }
    details = {
        "architecture_id": run_manifest.architecture_id,
        "architecture_version_id": run_manifest.architecture_version_id,
        "benchmark_formula_cap": benchmark_component.formula_cap,
        "benchmark_weights": benchmark_component.benchmark_weights,
        "loss_normalization_scope": run_manifest.metrics.loss.loss_normalization_scope,
        "raw_final_loss_used": False,
    }
    if run_manifest.metrics.loss.loss_normalization_scope == "architecture_baseline":
        raise ScoreValidationError(
            [
                "architecture official scoring requires fixed_tokenizer or byte_normalized "
                "standardized loss metadata"
            ]
        )
    return _score_result("architecture", weights, component_values, details)


def score_training_manifest(
    manifest: PrismRunManifest | dict[str, Any],
    *,
    score_weights: ScoreWeights | None = None,
    benchmark_weights: BenchmarkWeights | None = None,
) -> OfficialScoreResult:
    run_manifest = _official_manifest(manifest)
    weights = _training_formula(score_weights)
    benchmark_component = benchmark_sanity_component(
        run_manifest.metrics.benchmark_scores,
        benchmark_weights or BenchmarkWeights(),
        _score_weights(ARCHITECTURE_SCORE_COMPONENTS, weights),
        track="training",
    )
    component_values: dict[str, tuple[float, str]] = {
        "architecture_normalized_heldout_improvement": (
            _clamp(run_manifest.metrics.loss.architecture_normalized_heldout_improvement),
            "loss.architecture_normalized_heldout_improvement",
        ),
        "learning_stability_dynamics": (
            _learning_stability_dynamics(run_manifest),
            "metrics.loss_vs_tokens + metrics.learning_speed_slope",
        ),
        "benchmark_sanity": (benchmark_component.raw_score, "metrics.benchmark_scores"),
        "compute_efficiency": (_compute_efficiency(run_manifest), "metrics.estimated_flops"),
        "reproducibility_stability": (
            _reproducibility_stability(run_manifest),
            "metrics.benchmark_noise_metadata",
        ),
        "robustness_failure_behavior": (
            _robustness_failure_behavior(run_manifest),
            "validation + metrics.diagnostics",
        ),
        "artifact_completeness": (_artifact_completeness(run_manifest), "artifacts"),
    }
    details = {
        "architecture_id": run_manifest.architecture_id,
        "architecture_version_id": run_manifest.architecture_version_id,
        "training_script_version_id": run_manifest.training_script_version_id,
        "benchmark_formula_cap": benchmark_component.formula_cap,
        "benchmark_weights": benchmark_component.benchmark_weights,
        "loss_normalization_scope": run_manifest.metrics.loss.loss_normalization_scope,
        "raw_final_loss_used": False,
    }
    return _score_result("training", weights, component_values, details)


def architecture_score_sort_key(score: RankedScore) -> tuple[float, float, str, str]:
    return (-score.score, score.compute, score.accepted_at, score.submission_id)


def rank_official_scores(scores: list[RankedScore]) -> list[RankedScore]:
    return sorted(scores, key=architecture_score_sort_key)


def final_score(
    *,
    q_arch: float,
    q_recipe: float,
    anti_cheat_multiplier: float,
    diversity_bonus: float,
    penalty: float,
    arch_weight: float = 0.7,
    recipe_weight: float = 0.3,
) -> ScoreResult:
    base = arch_weight * q_arch + recipe_weight * q_recipe
    novelty_gate = max(0.0, min(1.0, q_arch / 0.5))
    effective_diversity_bonus = diversity_bonus * novelty_gate
    score = max(0.0, base * anti_cheat_multiplier + effective_diversity_bonus - penalty)
    return ScoreResult(q_arch, q_recipe, score)


def _official_manifest(manifest: PrismRunManifest | dict[str, Any]) -> PrismRunManifest:
    try:
        run_manifest = (
            manifest
            if isinstance(manifest, PrismRunManifest)
            else PrismRunManifest.model_validate(manifest)
        )
        return run_manifest.require_official_scoring_ready()
    except Exception as exc:
        if isinstance(exc, ScoreValidationError):
            raise
        raise ScoreValidationError([str(exc)]) from exc


def _architecture_formula(score_weights: ScoreWeights | None) -> dict[str, float]:
    formula = score_weights.architecture_formula if score_weights else ARCHITECTURE_SCORE_COMPONENTS
    return _require_exact_formula(formula, ARCHITECTURE_SCORE_COMPONENTS, "architecture")


def _training_formula(score_weights: ScoreWeights | None) -> dict[str, float]:
    formula = score_weights.training_formula if score_weights else TRAINING_SCORE_COMPONENTS
    return _require_exact_formula(formula, TRAINING_SCORE_COMPONENTS, "training")


def _require_exact_formula(
    formula: Mapping[str, float], expected: Mapping[str, float], label: str
) -> dict[str, float]:
    missing = sorted(set(expected) - set(formula))
    extra = sorted(set(formula) - set(expected))
    mismatched = [
        key
        for key, value in expected.items()
        if key in formula and abs(float(formula[key]) - value) > 1e-9
    ]
    if missing or extra or mismatched:
        reasons = []
        if missing:
            reasons.append(f"{label} score formula missing components: {missing}")
        if extra:
            reasons.append(f"{label} score formula has unknown components: {extra}")
        if mismatched:
            reasons.append(f"{label} score formula has incorrect weights: {mismatched}")
        raise ScoreValidationError(reasons)
    return {key: float(formula[key]) for key in expected}


def _score_weights(
    architecture_formula: Mapping[str, float], training_formula: Mapping[str, float]
) -> ScoreWeights:
    return ScoreWeights(
        final_architecture_weight=0.6,
        final_recipe_weight=0.4,
        architecture_formula=dict(architecture_formula),
        training_formula=dict(training_formula),
    )


def _score_result(
    track: Literal["architecture", "training"],
    weights: Mapping[str, float],
    component_values: dict[str, tuple[float, str]],
    details: dict[str, Any],
) -> OfficialScoreResult:
    components = tuple(
        ScoreComponentDetail(
            name=name,
            weight=weight,
            raw_value=_clamp(component_values[name][0]),
            weighted_contribution=weight * _clamp(component_values[name][0]),
            source=component_values[name][1],
        )
        for name, weight in weights.items()
    )
    score = _clamp(sum(component.weighted_contribution for component in components))
    return OfficialScoreResult(track=track, score=score, components=components, details=details)


def _learning_scaling_dynamics(manifest: PrismRunManifest) -> float:
    loss = manifest.metrics.loss
    slope_score = _clamp(-manifest.metrics.learning_speed_slope * 10.0)
    return _clamp(0.6 * loss.relative_loss_reduction + 0.4 * slope_score)


def _standardized_lm_quality(manifest: PrismRunManifest) -> float:
    return _clamp(1.0 / (1.0 + manifest.metrics.loss.standardized_eval_loss))


def _compute_efficiency(manifest: PrismRunManifest) -> float:
    flops_per_token = manifest.metrics.estimated_flops / max(
        float(manifest.metrics.tokens_seen), 1.0
    )
    return _clamp(1.0 / (1.0 + flops_per_token / 1_000_000.0))


def _parameter_efficiency(manifest: PrismRunManifest) -> float:
    return _clamp(1.0 / (1.0 + manifest.metrics.parameter_count / 1_000_000_000.0))


def _diagnostics_health(manifest: PrismRunManifest) -> float:
    values = []
    for diagnostic in manifest.metrics.diagnostics.values():
        if diagnostic.status == "not_applicable":
            continue
        aggregate = _clamp(cast(float, diagnostic.aggregate))
        values.append(aggregate * (0.5 if diagnostic.status == "warning" else 1.0))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _robustness_stability(manifest: PrismRunManifest) -> float:
    points = manifest.metrics.loss_vs_tokens
    if len(points) < 2:
        return 0.0
    ordered = sorted(points, key=lambda point: point.x)
    non_increasing = sum(
        1
        for previous, current in zip(ordered, ordered[1:], strict=False)
        if current.loss <= previous.loss
    )
    monotonic_score = non_increasing / max(len(ordered) - 1, 1)
    return _clamp(0.7 * monotonic_score + 0.3 * _diagnostics_health(manifest))


def _learning_stability_dynamics(manifest: PrismRunManifest) -> float:
    return _clamp(
        0.5 * _learning_scaling_dynamics(manifest) + 0.5 * _robustness_stability(manifest)
    )


def _reproducibility_stability(manifest: PrismRunManifest) -> float:
    stderr_values = manifest.metrics.benchmark_noise_metadata.get("stderr_by_benchmark", {})
    if not isinstance(stderr_values, dict) or not stderr_values:
        return 1.0
    average_stderr = sum(abs(float(value)) for value in stderr_values.values()) / len(stderr_values)
    return _clamp(1.0 / (1.0 + average_stderr * 10.0))


def _robustness_failure_behavior(manifest: PrismRunManifest) -> float:
    if not manifest.validation.passed or manifest.validation.errors:
        return 0.0
    warning_count = sum(
        1 for diagnostic in manifest.metrics.diagnostics.values() if diagnostic.status == "warning"
    )
    return _clamp(1.0 - 0.1 * warning_count)


def _artifact_completeness(manifest: PrismRunManifest) -> float:
    required = [
        manifest.artifacts.architecture_graph.path,
        manifest.artifacts.architecture_metadata.path,
        manifest.artifacts.run_log.path,
        manifest.artifacts.metrics.path if manifest.artifacts.metrics else "",
    ]
    return sum(1 for value in required if value) / len(required)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
