"""Scale-eval ladder product pin guards + Complete View densify entrypoints.

P0 product surface (VAL-SCALE-018 partial; pin plumbing for P0→P3):

* Multi-seed K≥3 default ProtocolPin fields for public / non-provisional cups
* Multi-family host compare under one matched explore pin
* Complete View long_ctx + sample_eff densify entrypoints (host-side, $0 GPU)

LAB-GPU train cups are separate lab features. This module never invents metrics,
never ranks on wall-clock, and never reintroduces a Prism tee package.
Emission remains heldout-primary + bpb secondary (research Complete View is
non-emission unless an explicit protocol v2 annex lands elsewhere).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .complete_view_eff import (
    FamilyEffStability,
    build_complete_view_with_eff_stability,
)
from .complete_view_longctx import (
    FamilyLongCtxQuality,
    build_complete_view_with_longctx_quality,
)
from .multi_family_compare import (
    FRONTIER_FAIR_EVAL_FAMILY_IDS,
    explore_protocol_pin,
    run_multi_family_lab_gpu_host_compare,
    run_multi_family_official_compare,
)
from .official_comparison import (
    OFFICIAL_DEFAULT_BATCH_SIZE,
    OFFICIAL_DEFAULT_SEEDS,
    OFFICIAL_DEFAULT_SEQ_LEN,
    OFFICIAL_DEFAULT_TOKEN_BUDGET,
    OFFICIAL_DEFAULT_TOKENIZER,
    OFFICIAL_DEFAULT_VAL_BYTE_BUDGET,
    OFFICIAL_EXPLORE_PARAM_CAP,
    OFFICIAL_EXPLORE_STAGE,
    OFFICIAL_MIN_PUBLIC_SEEDS,
    OFFICIAL_WALL_CLOCK_NEVER_RANKS,
    OfficialScoreRecord,
    ProtocolPin,
    protocol_budget_constants,
)

# P0 cup defaults (explore tiny, short-ctx seq, 500k tokens, public K=3 seeds).
SCALE_P0_SEEDS: tuple[int, ...] = tuple(OFFICIAL_DEFAULT_SEEDS)  # (1337, 2027, 4242)
SCALE_P0_TOKEN_BUDGET: int = int(OFFICIAL_DEFAULT_TOKEN_BUDGET)  # 500_000
SCALE_P0_SEQ_LEN: int = int(OFFICIAL_DEFAULT_SEQ_LEN)  # 128
SCALE_P0_BATCH_SIZE: int = int(OFFICIAL_DEFAULT_BATCH_SIZE)  # 4
SCALE_P0_PARAM_STAGE: str = str(OFFICIAL_EXPLORE_STAGE)
SCALE_P0_PARAM_CAP: int = int(OFFICIAL_EXPLORE_PARAM_CAP)
SCALE_P0_CORE_FAMILY_IDS: tuple[str, ...] = (
    "deeploop-tiny-1m",
    "mamba-tiny-1m",
    "transformer-tiny-1m",
    "kda-tiny-1m",
)

DensifyPanel = Literal["long_ctx", "sample_eff", "both"]


@dataclass(frozen=True)
class ScalePinGuardResult:
    """Outcome of :func:`assert_public_multi_seed_pin` / :func:`scale_pin_public_ok`."""

    ok: bool
    seed_count: int
    min_public_seeds: int
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "seed_count": self.seed_count,
            "min_public_seeds": self.min_public_seeds,
            "reasons": list(self.reasons),
        }


def scale_p0_protocol_pin(
    *,
    seeds: Sequence[int] | None = None,
    token_budget: int | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    require_public_k: bool = True,
) -> ProtocolPin:
    """Matched explore ProtocolPin for the P0 multi-seed scale-eval cup.

    Defaults freeze K≥3 public seeds (1337/2027/4242), seq=128, token_budget=500k,
    explore 124M ceiling, gpt2 tokenizer, heldout_delta primary. Callers may pass
    K=1 for provisional lab labels only when ``require_public_k=False``.
    """
    seed_tuple = tuple(int(s) for s in seeds) if seeds is not None else SCALE_P0_SEEDS
    if require_public_k and len(seed_tuple) < OFFICIAL_MIN_PUBLIC_SEEDS:
        raise ValueError(
            f"public scale pin requires K≥{OFFICIAL_MIN_PUBLIC_SEEDS} seeds; "
            f"got K={len(seed_tuple)} (set require_public_k=False for provisional lab)"
        )
    pin = explore_protocol_pin(
        seeds=seed_tuple,
        token_budget=int(token_budget) if token_budget is not None else SCALE_P0_TOKEN_BUDGET,
    )
    # explore_protocol_pin inherits default seq/batch; freeze P0 knobs explicitly.
    from dataclasses import replace

    return replace(
        pin,
        seq_len=int(seq_len) if seq_len is not None else SCALE_P0_SEQ_LEN,
        batch_size=int(batch_size) if batch_size is not None else SCALE_P0_BATCH_SIZE,
        param_cap=SCALE_P0_PARAM_CAP,
        param_ladder_stage=SCALE_P0_PARAM_STAGE,
        tokenizer=OFFICIAL_DEFAULT_TOKENIZER,
        val_byte_budget=OFFICIAL_DEFAULT_VAL_BYTE_BUDGET,
        primary_form="heldout_delta",
        force_iter_train_batches=True,
        require_trained_state=True,
    )


def scale_pin_fields(pin: ProtocolPin | None = None) -> dict[str, Any]:
    """Documented pin field surface for scale-eval operators / regression tests."""
    active = pin if pin is not None else scale_p0_protocol_pin()
    d = active.as_dict()
    return {
        "protocol_id": d["protocol_id"],
        "token_budget": d["token_budget"],
        "seeds": list(d["seeds"]),
        "seed_count": len(d["seeds"]),
        "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
        "seq_len": d["seq_len"],
        "batch_size": d["batch_size"],
        "tokenizer": d["tokenizer"],
        "vocab_size": d["vocab_size"],
        "param_cap": d["param_cap"],
        "param_ladder_stage": d["param_ladder_stage"],
        "val_byte_budget": d["val_byte_budget"],
        "primary_form": d["primary_form"],
        "wall_clock_never_ranks": bool(d.get("wall_clock_never_ranks", True)),
        "force_iter_train_batches": d["force_iter_train_batches"],
        "require_trained_state": d["require_trained_state"],
        "official_wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
    }


def scale_pin_public_ok(pin: ProtocolPin) -> ScalePinGuardResult:
    """Check pin is eligible for public non-provisional multi-seed claims (K≥3)."""
    reasons: list[str] = []
    seeds = tuple(pin.seeds)
    k = len(seeds)
    if k < OFFICIAL_MIN_PUBLIC_SEEDS:
        reasons.append(f"seed_count_below_public_min:K={k}<{OFFICIAL_MIN_PUBLIC_SEEDS}")
    if len(set(seeds)) != k:
        reasons.append("duplicate_seeds")
    if pin.primary_form != "heldout_delta" and pin.primary_form != "val_bpb_trained":
        reasons.append(f"unknown_primary_form:{pin.primary_form}")
    if not OFFICIAL_WALL_CLOCK_NEVER_RANKS:
        reasons.append("wall_clock_rank_flag_broken")
    if int(pin.token_budget) <= 0:
        reasons.append("non_positive_token_budget")
    if int(pin.seq_len) <= 0:
        reasons.append("non_positive_seq_len")
    return ScalePinGuardResult(
        ok=not reasons,
        seed_count=k,
        min_public_seeds=OFFICIAL_MIN_PUBLIC_SEEDS,
        reasons=tuple(reasons),
    )


def assert_public_multi_seed_pin(pin: ProtocolPin) -> None:
    """Raise ``ValueError`` when pin fails public K≥3 / matched-field guards."""
    result = scale_pin_public_ok(pin)
    if not result.ok:
        raise ValueError("public multi-seed pin guard failed: " + ";".join(result.reasons))


def densify_entrypoints() -> dict[str, Any]:
    """Machine-readable map of Complete View densify APIs for scale-eval operators.

    Prefer host densify on existing ``trained_state`` / fixture families before new
    Lium trains. Entry points are pure product imports (no GPU required for fixture
    densify; LAB-GPU artifact densify is best-effort host CPU).
    """
    return {
        "schema": "prism_scale_densify_entrypoints.v1",
        "long_ctx": {
            "module": "prism_challenge.evaluator.complete_view_longctx",
            "build_view": "build_complete_view_with_longctx_quality",
            "panels": "build_longctx_quality_panels",
            "fixture_family": "fixture_family_longctx_quality",
            "multi_t_suite": "multi_t_long_ctx_suite",
            "multi_seed_val_bpb": "multi_seed_val_bpb_trained",
            "notes": (
                "Host densify long_ctx panel on K≥3 trained_state or fixtures; "
                "does not rewrite emission heldout-primary rank."
            ),
        },
        "sample_eff": {
            "module": "prism_challenge.evaluator.complete_view_eff",
            "build_view": "build_complete_view_with_eff_stability",
            "panels": "build_eff_stability_panels",
            "fixture_family": "fixture_family_eff_stability",
            "dense_from_stream": "dense_sample_efficiency_from_stream",
            "train_series_stability": "densify_stability_from_train_series",
            "notes": (
                "sample_eff / train_series densify is residual scientific; "
                "never sole-primary over heldout/bpb."
            ),
        },
        "multi_family_host_compare": {
            "module": "prism_challenge.evaluator.multi_family_compare",
            "run_lab_gpu": "run_multi_family_lab_gpu_host_compare",
            "run_fixture": "run_multi_family_official_compare",
            "explore_pin": "explore_protocol_pin",
            "scale_p0_pin": "scale_p0_protocol_pin",
            "core_families_p0": list(SCALE_P0_CORE_FAMILY_IDS),
            "frontier_families": list(FRONTIER_FAIR_EVAL_FAMILY_IDS),
        },
        "scale_helpers": {
            "module": "prism_challenge.evaluator.scale_eval",
            "p0_pin": "scale_p0_protocol_pin",
            "pin_fields": "scale_pin_fields",
            "public_ok": "scale_pin_public_ok",
            "densify_pair": "densify_complete_view_pair",
            "host_compare": "run_scale_multi_family_host_compare",
        },
        "rank_guards": {
            "primary": "heldout_delta (higher better)",
            "secondary": "prequential bpb (lower better, Prism-recomputed)",
            "anti": ["memorization_flag", "step0_anomaly", "miner_self_report_ignored"],
            "wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
            "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
            "tee_package": "absent (provider trust + IMAGE_PIN only)",
        },
        "protocol_budget": protocol_budget_constants(),
    }


def densify_complete_view_pair(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    panel: DensifyPanel = "both",
    fam_long_a: FamilyLongCtxQuality | None = None,
    fam_long_b: FamilyLongCtxQuality | None = None,
    fam_eff_a: FamilyEffStability | None = None,
    fam_eff_b: FamilyEffStability | None = None,
    score_class: str = "fixture",
    **kwargs: Any,
) -> dict[str, Any]:
    """Single densify entrypoint for Complete View long_ctx and/or sample_eff panels.

    Pass pre-built :class:`FamilyLongCtxQuality` / :class:`FamilyEffStability` packs
    from host densify on trained_state (or the existing fixture builders under
    ``complete_view_longctx`` / ``complete_view_eff``). Omitting packs still yields a
    valid complete_view document with null/not-run honesty on empty panels.
    Does **not** rewrite emission heldout-primary rank.
    """
    if panel == "long_ctx":
        return build_complete_view_with_longctx_quality(
            a,
            b,
            fam_a=fam_long_a,
            fam_b=fam_long_b,
            score_class=score_class,
            **kwargs,
        )
    if panel == "sample_eff":
        return build_complete_view_with_eff_stability(
            a,
            b,
            fam_a=fam_eff_a,
            fam_b=fam_eff_b,
            score_class=score_class,
            **kwargs,
        )
    # both: long_ctx first, then overlay eff/stability panels.
    base = build_complete_view_with_longctx_quality(
        a,
        b,
        fam_a=fam_long_a,
        fam_b=fam_long_b,
        score_class=score_class,
        **kwargs,
    )
    from .complete_view import COMPLETE_VIEW_PANEL_KEYS

    long_panels = {
        k: v for k, v in (base.get("panels") or {}).items() if k in COMPLETE_VIEW_PANEL_KEYS
    }
    return build_complete_view_with_eff_stability(
        a,
        b,
        fam_a=fam_eff_a,
        fam_b=fam_eff_b,
        panels_override=long_panels,
        score_class=score_class,
        **kwargs,
    )


def run_scale_multi_family_host_compare(
    output_dir: Path | str,
    *,
    artifacts_root: Path | str | None = None,
    family_ids: Sequence[str] | None = None,
    pin: ProtocolPin | None = None,
    seeds: Sequence[int] | None = None,
    package: bool = True,
    write_report: bool = True,
    allow_partial: bool = True,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    """Multi-family host compare under the scale-eval P0 pin (fixture or LAB-GPU).

    * ``fixture_mode=True`` (default when no artifacts_root): synthetic multi-family
      Official compare under matched pin — no GPU / no Lium.
    * ``artifacts_root`` set: host recompute from LAB-GPU manifests (missing →
      BLOCKED_with_reason, never invented).
    """
    ids = tuple(family_ids) if family_ids is not None else SCALE_P0_CORE_FAMILY_IDS
    active_pin = (
        pin
        if pin is not None
        else scale_p0_protocol_pin(
            seeds=seeds if seeds is not None else None,
            require_public_k=True,
        )
    )
    assert_public_multi_seed_pin(active_pin)
    seed_tuple = tuple(int(s) for s in (seeds if seeds is not None else active_pin.seeds))

    if artifacts_root is not None and not fixture_mode:
        return run_multi_family_lab_gpu_host_compare(
            artifacts_root,
            output_dir,
            family_ids=ids,
            seeds=seed_tuple,
            pin=active_pin,
            package=package,
            write_report=write_report,
            allow_partial=allow_partial,
        )
    return run_multi_family_official_compare(
        output_dir,
        family_ids=ids,
        pin=active_pin,
        package=package,
        write_report=write_report,
    )


def tee_package_absent() -> bool:
    """True when Prism tee package path is gone (scale-eval + NO TEE residual)."""
    from pathlib import Path as _Path

    # Prefer filesystem check over import so a stale pyc cannot fool us alone.
    root = _Path(__file__).resolve().parents[1]  # .../prism_challenge
    return not (root / "tee").exists()


def scale_product_snapshot() -> dict[str, Any]:
    """Compact snapshot for evidence packs (no secrets, no spend)."""
    pin = scale_p0_protocol_pin()
    guard = scale_pin_public_ok(pin)
    return {
        "schema": "prism_scale_product_snapshot.v1",
        "pin": scale_pin_fields(pin),
        "public_guard": guard.as_dict(),
        "densify_entrypoints": densify_entrypoints(),
        "core_families_p0": list(SCALE_P0_CORE_FAMILY_IDS),
        "tee_package_absent": tee_package_absent(),
        "wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
        "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
    }
