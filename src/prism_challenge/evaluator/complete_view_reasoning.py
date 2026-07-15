"""Complete View P10 reasoning/logic panel fill from synthetic probe scores.

Wires challenge-owned logic suite metrics (VAL-REASON-002..008) into the
``P10_reasoning_logic`` panel shell defined by ``complete_view.reasoning_panel_shell``.

This module:

* scores MUST probes via fixture outcomes or host densify hooks
* dual-channels closed accuracy + forced CE with published chance baselines
* fills per-probe fields on a one-family or dual-family panel
* leaves suite_mean / multi-axis polar aggregation to later features
  (``reason-logic-aggregate-multiaxis``)

Architecture-agnostic. Prefer pure-torch CPU unit paths. No lm-eval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .benchmarks.logic_suite import (
    DEFAULT_LOGIC_SUITE_SEED,
    DEFAULT_TRIALS_PER_PROBE,
    LOGIC_CHANCE,
    LOGIC_PROBE_KEYS,
    LogicTaskScore,
    documented_logic_suite,
    score_suite_fixture,
)
from .complete_view import (
    COMPLETE_VIEW_REASONING_CHANCE_TABLE,
    REASONING_REL_FLOOR,
    REASONING_SUITE_ID,
    build_complete_view,
    reasoning_panel_shell,
)
from .official_comparison import OfficialScoreRecord
from .scorecard_suite import relative_to_chance

DeviceHint = Literal["cpu", "cuda", "auto", "fixture"]


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


@dataclass(frozen=True)
class FamilyReasoningScores:
    """One-family P10 probe metric bundle (acc + relative + forced CE)."""

    probes: dict[str, LogicTaskScore]
    device: str = "fixture"
    seeds: tuple[int, ...] | None = None
    suite_id: str = REASONING_SUITE_ID

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "device": self.device,
            "seeds": None if self.seeds is None else list(self.seeds),
            "probes": {k: v.as_dict() for k, v in self.probes.items()},
        }

    def accuracy_map(self) -> dict[str, float]:
        return {k: float(v.accuracy) for k, v in self.probes.items()}

    def relative_map(self) -> dict[str, float]:
        return {k: float(v.relative) for k, v in self.probes.items()}

    def forced_ce_map(self) -> dict[str, float | None]:
        return {k: v.forced_ce for k, v in self.probes.items()}


def family_reasoning_from_scores(
    scores: Mapping[str, LogicTaskScore],
    *,
    device: str = "fixture",
    seeds: Sequence[int] | None = None,
) -> FamilyReasoningScores:
    """Pack an existing probe→score map into a family payload."""
    probes: dict[str, LogicTaskScore] = {}
    for key in LOGIC_PROBE_KEYS:
        if key not in scores:
            raise KeyError(f"missing MUST probe score for {key}")
        probes[key] = scores[key]
    return FamilyReasoningScores(
        probes=probes,
        device=device,
        seeds=None if seeds is None else tuple(int(s) for s in seeds),
    )


def family_reasoning_fixture(
    *,
    accuracy_by_probe: Mapping[str, float] | None = None,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    device: str = "fixture",
    seeds: Sequence[int] | None = None,
) -> FamilyReasoningScores:
    """CPU fixture path for one family (dual channel metrics for all MUST probes)."""
    scores = score_suite_fixture(
        n_trials=n_trials,
        suite_seed=suite_seed,
        accuracy_by_probe=accuracy_by_probe,
        device=device,
    )
    return family_reasoning_from_scores(scores, device=device, seeds=seeds)


def _probe_entry_from_score(
    score: LogicTaskScore, *, side_status: str = "filled"
) -> dict[str, Any]:
    return {
        "accuracy": score.accuracy,
        "relative_to_chance": score.relative,
        "forced_ce": score.forced_ce,
        "chance": score.chance,
        "trials": score.trials,
        "device": score.device,
        "status": side_status,
        "detail": dict(score.detail) if score.detail is not None else None,
    }


def fill_reasoning_panel(
    panel: dict[str, Any] | None = None,
    *,
    a: FamilyReasoningScores | None = None,
    b: FamilyReasoningScores | None = None,
    status_if_partial: str = "partial",
) -> dict[str, Any]:
    """Fill (or create) a P10 panel dict from one/both family score bundles.

    Aggregate suite_mean / floor_pass intentional left for
    ``reason-logic-aggregate-multiaxis``; this function fills **per-probe**
    dual-score metrics required by VAL-REASON-002..008.
    """
    out = reasoning_panel_shell(status="not_run") if panel is None else dict(panel)
    # deep-copy probes map lightly
    probes_in = out.get("probes")
    probes: dict[str, Any] = {}
    if isinstance(probes_in, Mapping):
        for k, v in probes_in.items():
            probes[str(k)] = dict(v) if isinstance(v, Mapping) else v
    else:
        probes = {}

    filled_any = False
    for key in LOGIC_PROBE_KEYS:
        entry = dict(probes.get(key) or {})
        entry.setdefault("matrix_id", None)
        entry.setdefault("chance", COMPLETE_VIEW_REASONING_CHANCE_TABLE.get(key))
        entry.setdefault("acc", {"a": None, "b": None})
        entry.setdefault("rel_to_chance", {"a": None, "b": None})
        entry.setdefault("forced_ce", {"a": None, "b": None})
        entry.setdefault("by_length", {})
        entry.setdefault("trials", 0)
        if not isinstance(entry["acc"], dict):
            entry["acc"] = {"a": None, "b": None}
        if not isinstance(entry["rel_to_chance"], dict):
            entry["rel_to_chance"] = {"a": None, "b": None}
        if not isinstance(entry["forced_ce"], dict):
            entry["forced_ce"] = {"a": None, "b": None}

        trials = 0
        if a is not None and key in a.probes:
            sa = a.probes[key]
            entry["acc"]["a"] = sa.accuracy
            entry["rel_to_chance"]["a"] = sa.relative
            entry["forced_ce"]["a"] = sa.forced_ce
            entry["chance"] = sa.chance
            entry["side_a"] = _probe_entry_from_score(sa)
            trials = max(trials, int(sa.trials))
            filled_any = True
        if b is not None and key in b.probes:
            sb = b.probes[key]
            entry["acc"]["b"] = sb.accuracy
            entry["rel_to_chance"]["b"] = sb.relative
            entry["forced_ce"]["b"] = sb.forced_ce
            entry["chance"] = sb.chance
            entry["side_b"] = _probe_entry_from_score(sb)
            trials = max(trials, int(sb.trials))
            filled_any = True

        both = (
            entry["acc"].get("a") is not None
            and entry["acc"].get("b") is not None
            and math.isfinite(float(entry["acc"]["a"]))
            and math.isfinite(float(entry["acc"]["b"]))
        )
        one = entry["acc"].get("a") is not None or entry["acc"].get("b") is not None
        if both:
            entry["status"] = "filled"
            entry["reason"] = None
        elif one:
            entry["status"] = status_if_partial
            entry["reason"] = "partial_one_family_only"
        # else keep not_run shell
        entry["trials"] = trials
        probes[key] = entry

    out["probes"] = probes
    out["chance_table"] = dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE)
    out["suite_id"] = REASONING_SUITE_ID
    out["rel_floor"] = REASONING_REL_FLOOR
    out["scoring"] = {
        "closed_choice_accuracy": True,
        "forced_ce": True,
        "chance_baselines": True,
        "architecture_agnostic_logits_only": True,
    }
    seeds_a = None if a is None else a.seeds
    seeds_b = None if b is None else b.seeds
    if seeds_a is not None or seeds_b is not None:
        out["seeds"] = {
            "a": None if seeds_a is None else list(seeds_a),
            "b": None if seeds_b is None else list(seeds_b),
        }
    devices = []
    if a is not None:
        devices.append(a.device)
    if b is not None:
        devices.append(b.device)
    if devices and len(set(devices)) == 1:
        out["device"] = devices[0]
    else:
        out["device"] = (devices or ["fixture"])[0]

    # Panel-level status: filled only when every MUST probe has dual sides; else partial/not_run.
    all_both = all(
        isinstance(probes.get(k), Mapping) and probes[k].get("status") == "filled"
        for k in LOGIC_PROBE_KEYS
    )
    any_probe = any(
        isinstance(probes.get(k), Mapping)
        and probes[k].get("status") in ("filled", "partial", status_if_partial)
        for k in LOGIC_PROBE_KEYS
    )
    if all_both:
        out["status"] = "filled"
        out["reason"] = None
    elif any_probe or filled_any:
        out["status"] = status_if_partial
        out["reason"] = "probe_suite_partial_or_one_family"
    # Aggregates stay null for the aggregate feature (suite_mean).
    ag = dict(out.get("aggregates") or {})
    ag.setdefault("logic_acc_macro", {"a": None, "b": None})
    ag.setdefault("logic_rel_macro", {"a": None, "b": None})
    ag.setdefault("logic_ce_macro", {"a": None, "b": None})
    ag.setdefault("logic_below_chance_count", {"a": None, "b": None})
    ag.setdefault("logic_floor_pass", {"a": None, "b": None})
    ag.setdefault("suite_mean", {"a": None, "b": None})
    ag["status"] = "not_run" if out["status"] != "filled" else "pending_aggregate"
    ag["reason"] = (
        "suite_mean deferred to reason-logic-aggregate-multiaxis"
        if out["status"] in ("filled", status_if_partial, "partial")
        else "not_run_until_probe_suite_aggregate"
    )
    out["aggregates"] = ag
    out["meta"] = {
        "suite_doc": documented_logic_suite(),
        "chance_table": dict(LOGIC_CHANCE),
        "host_densify_trained_state": True,
        "no_lm_eval": True,
    }
    return out


def build_complete_view_with_reasoning(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    family_a: FamilyReasoningScores | None = None,
    family_b: FamilyReasoningScores | None = None,
    score_class: str = "fixture",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build complete_view.v1.3 document with P10 filled from family reasoning scores."""
    doc = build_complete_view(a, b, score_class=score_class, **kwargs)
    panels = dict(doc.get("panels") or {})
    panels["P10_reasoning_logic"] = fill_reasoning_panel(
        panels.get("P10_reasoning_logic"),
        a=family_a,
        b=family_b,
    )
    doc["panels"] = panels
    return doc


@dataclass(frozen=True)
class DualFamilyReasoning:
    """Convenience dual-side fixture holder for VAL-REASON unit tests."""

    a: FamilyReasoningScores
    b: FamilyReasoningScores
    notes: tuple[str, ...] = field(default_factory=tuple)

    def panel(self) -> dict[str, Any]:
        return fill_reasoning_panel(a=self.a, b=self.b)


def dual_family_reasoning_fixture(
    *,
    a_acc: Mapping[str, float] | None = None,
    b_acc: Mapping[str, float] | None = None,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed_a: int = DEFAULT_LOGIC_SUITE_SEED,
    suite_seed_b: int = DEFAULT_LOGIC_SUITE_SEED + 7,
    device: str = "fixture",
    seeds: Sequence[int] = (1337, 2027, 4242),
) -> DualFamilyReasoning:
    """Symmetric dual-family synthetic accuracies for unit fixtures."""
    fa = family_reasoning_fixture(
        accuracy_by_probe=a_acc,
        n_trials=n_trials,
        suite_seed=suite_seed_a,
        device=device,
        seeds=seeds,
    )
    fb = family_reasoning_fixture(
        accuracy_by_probe=b_acc,
        n_trials=n_trials,
        suite_seed=suite_seed_b,
        device=device,
        seeds=seeds,
    )
    return DualFamilyReasoning(
        a=fa,
        b=fb,
        notes=(
            "synthetic_fixture_not_trained_state",
            "REAL-PROVIDER TEE BLOCKED",
            "not_gsm8k_mmlu",
        ),
    )


def probe_metric_bundle(
    scores_a: Mapping[str, LogicTaskScore] | FamilyReasoningScores,
    scores_b: Mapping[str, LogicTaskScore] | FamilyReasoningScores,
) -> dict[str, dict[str, Any]]:
    """Compact dual-side metrics dict (acc/rel/ce) per probe for tests & reports."""

    def _as_map(
        s: Mapping[str, LogicTaskScore] | FamilyReasoningScores,
    ) -> Mapping[str, LogicTaskScore]:
        if isinstance(s, FamilyReasoningScores):
            return s.probes
        return s

    ma, mb = _as_map(scores_a), _as_map(scores_b)
    out: dict[str, dict[str, Any]] = {}
    for key in LOGIC_PROBE_KEYS:
        sa, sb = ma[key], mb[key]
        out[key] = {
            "chance": sa.chance,
            "acc": {"a": sa.accuracy, "b": sb.accuracy},
            "rel_to_chance": {"a": sa.relative, "b": sb.relative},
            "forced_ce": {"a": sa.forced_ce, "b": sb.forced_ce},
            "trials": {"a": sa.trials, "b": sb.trials},
        }
    return out


def assert_probe_dual_channel(score: LogicTaskScore) -> None:
    """Raise AssertionError if a MUST probe score lacks dual channel fields."""
    if score.probe not in LOGIC_PROBE_KEYS:
        raise AssertionError(f"unknown probe {score.probe}")
    if not math.isfinite(score.accuracy) or not (0.0 <= score.accuracy <= 1.0):
        raise AssertionError(f"{score.probe}: accuracy out of range: {score.accuracy}")
    ch = LOGIC_CHANCE[score.probe]
    if abs(score.chance - ch) > 1e-9:
        raise AssertionError(f"{score.probe}: chance {score.chance} != table {ch}")
    expected_rel = relative_to_chance(score.accuracy, score.chance)
    if abs(score.relative - expected_rel) > 1e-9:
        raise AssertionError(
            f"{score.probe}: relative_to_chance {score.relative} != {expected_rel}"
        )
    if score.forced_ce is None or not math.isfinite(float(score.forced_ce)):
        raise AssertionError(f"{score.probe}: forced_ce missing/non-finite")
    if score.trials <= 0:
        raise AssertionError(f"{score.probe}: trials must be > 0")


__all__ = [
    "DualFamilyReasoning",
    "FamilyReasoningScores",
    "assert_probe_dual_channel",
    "build_complete_view_with_reasoning",
    "dual_family_reasoning_fixture",
    "family_reasoning_fixture",
    "family_reasoning_from_scores",
    "fill_reasoning_panel",
    "probe_metric_bundle",
]
