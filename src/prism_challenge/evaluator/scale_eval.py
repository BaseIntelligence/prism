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

from . import param_ladder as _param_ladder
from .complete_view_eff import (
    FamilyEffStability,
    build_complete_view_with_eff_stability,
)
from .complete_view_longctx import (
    FamilyLongCtxQuality,
    build_complete_view_with_longctx_quality,
)
from .interface import PrismContext
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
    OFFICIAL_DEFAULT_VOCAB_SIZE,
    OFFICIAL_EXPLORE_PARAM_CAP,
    OFFICIAL_EXPLORE_STAGE,
    OFFICIAL_MIN_PUBLIC_SEEDS,
    OFFICIAL_PROMOTE_PARAM_CAP,
    OFFICIAL_PROMOTE_STAGE,
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

# P1 scale defaults (VAL-SCALE-006): seq≥256 (512 target when VRAM allows), tokens≥1M.
SCALE_P1_SEEDS: tuple[int, ...] = SCALE_P0_SEEDS
SCALE_P1_SEQ_LEN: int = 256
SCALE_P1_SEQ_LEN_TARGET: int = 512
SCALE_P1_TOKEN_BUDGET: int = 1_000_000
SCALE_P1_TOKEN_BUDGET_HIGH: int = 2_000_000
SCALE_P1_BATCH_SIZE: int = SCALE_P0_BATCH_SIZE
SCALE_P1_PARAM_STAGE: str = SCALE_P0_PARAM_STAGE
SCALE_P1_PARAM_CAP: int = SCALE_P0_PARAM_CAP
SCALE_P1_CORE_FAMILY_IDS: tuple[str, ...] = SCALE_P0_CORE_FAMILY_IDS
SCALE_P1_SEQ_LEN_MIN: int = 256
SCALE_P1_TOKEN_BUDGET_MIN: int = 1_000_000

# P2 promote 350M ladder defaults (VAL-SCALE-011): same matched package pin as P1
# explore floor, but param_ladder_stage=promote and param_cap=350M. Crown candidates
# are deeploop + explore runner-up + transformer baseline (kda optional when budget).
SCALE_P2_SEEDS: tuple[int, ...] = SCALE_P0_SEEDS
SCALE_P2_SEQ_LEN: int = SCALE_P1_SEQ_LEN  # keep P1 seq floor for fair promote re-eval
SCALE_P2_TOKEN_BUDGET: int = SCALE_P1_TOKEN_BUDGET
SCALE_P2_BATCH_SIZE: int = SCALE_P0_BATCH_SIZE
SCALE_P2_PARAM_STAGE: str = str(OFFICIAL_PROMOTE_STAGE)
SCALE_P2_PARAM_CAP: int = int(OFFICIAL_PROMOTE_PARAM_CAP)
# Minimum crown set for promote confirm/revoke (add kda when budget allows).
SCALE_P2_CROWN_FAMILY_IDS: tuple[str, ...] = (
    "deeploop-tiny-1m",
    "mamba-tiny-1m",  # P1 explore crown / best runner relative to deeploop lineage
    "transformer-tiny-1m",
)
SCALE_P2_CORE_FAMILY_IDS: tuple[str, ...] = SCALE_P2_CROWN_FAMILY_IDS + ("kda-tiny-1m",)
SCALE_P2_SEQ_LEN_MIN: int = SCALE_P1_SEQ_LEN_MIN
SCALE_P2_TOKEN_BUDGET_MIN: int = SCALE_P1_TOKEN_BUDGET_MIN

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
    return explore_protocol_pin(
        seeds=seed_tuple,
        token_budget=int(token_budget) if token_budget is not None else SCALE_P0_TOKEN_BUDGET,
        seq_len=int(seq_len) if seq_len is not None else SCALE_P0_SEQ_LEN,
        batch_size=int(batch_size) if batch_size is not None else SCALE_P0_BATCH_SIZE,
    )


def scale_p1_protocol_pin(
    *,
    seeds: Sequence[int] | None = None,
    token_budget: int | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    require_public_k: bool = True,
    require_p1_floor: bool = True,
) -> ProtocolPin:
    """Matched explore ProtocolPin for P1 scaled seq/token cups (VAL-SCALE-006).

    Defaults: K≥3 public seeds, seq=256 (raise to 512 via ``seq_len`` when VRAM
    allows), token_budget=1_000_000 (raise to 2M via ``token_budget``). Does **not**
    hardcode seq=128; pin fields pass through explore/official/lab harness paths.
    Emission ranking keys unchanged (heldout primary + bpb secondary).
    """
    seed_tuple = tuple(int(s) for s in seeds) if seeds is not None else SCALE_P1_SEEDS
    if require_public_k and len(seed_tuple) < OFFICIAL_MIN_PUBLIC_SEEDS:
        raise ValueError(
            f"public scale pin requires K≥{OFFICIAL_MIN_PUBLIC_SEEDS} seeds; "
            f"got K={len(seed_tuple)} (set require_public_k=False for provisional lab)"
        )
    resolved_seq = int(seq_len) if seq_len is not None else SCALE_P1_SEQ_LEN
    resolved_budget = int(token_budget) if token_budget is not None else SCALE_P1_TOKEN_BUDGET
    pin = explore_protocol_pin(
        seeds=seed_tuple,
        token_budget=resolved_budget,
        seq_len=resolved_seq,
        batch_size=int(batch_size) if batch_size is not None else SCALE_P1_BATCH_SIZE,
    )
    if require_p1_floor:
        assert_scale_p1_pin_floor(pin)
    return pin


def assert_scale_p1_pin_floor(pin: ProtocolPin) -> None:
    """Raise when pin is below P1 product floors (seq≥256, token_budget≥1M)."""
    reasons: list[str] = []
    if int(pin.seq_len) < SCALE_P1_SEQ_LEN_MIN:
        reasons.append(
            f"seq_len={pin.seq_len}<{SCALE_P1_SEQ_LEN_MIN} "
            f"(P1 floor; target {SCALE_P1_SEQ_LEN_TARGET})"
        )
    if int(pin.token_budget) < SCALE_P1_TOKEN_BUDGET_MIN:
        reasons.append(f"token_budget={pin.token_budget}<{SCALE_P1_TOKEN_BUDGET_MIN} (P1 floor)")
    if reasons:
        raise ValueError("P1 scale pin floor failed: " + "; ".join(reasons))


def promote_protocol_pin(
    *,
    seeds: Sequence[int] | None = None,
    token_budget: int | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    k_label: str | None = None,
) -> ProtocolPin:
    """Matched ProtocolPin for promote-stage (350M) scale-eval cups (VAL-SCALE-011).

    Same tokenizer/seeds/budget contract as explore pins, but ``param_ladder_stage``
    is ``promote`` and ``param_cap`` is the 350M promote ceiling. Does not silently
    fall back to explore 124M.
    """
    del k_label  # report metadata only
    base = explore_protocol_pin(
        seeds=seeds,
        token_budget=token_budget,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    return ProtocolPin(
        protocol_id=base.protocol_id,
        token_budget=int(base.token_budget),
        seeds=tuple(int(s) for s in base.seeds),
        param_cap=int(OFFICIAL_PROMOTE_PARAM_CAP),
        param_ladder_stage=str(OFFICIAL_PROMOTE_STAGE),
        seq_len=int(base.seq_len),
        batch_size=int(base.batch_size),
        tokenizer=str(base.tokenizer),
        vocab_size=int(base.vocab_size),
        scored_nproc=int(base.scored_nproc),
        val_byte_budget=int(base.val_byte_budget),
        force_iter_train_batches=True,
        require_trained_state=True,
        primary_form="heldout_delta",
        require_train_series=False,
        wall_clock_seconds=base.wall_clock_seconds,
        step_budget=base.step_budget,
        gap_threshold_bpb=base.gap_threshold_bpb,
    )


def scale_p2_protocol_pin(
    *,
    seeds: Sequence[int] | None = None,
    token_budget: int | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    require_public_k: bool = True,
    require_p2_floor: bool = True,
) -> ProtocolPin:
    """Matched promote ProtocolPin for P2 350M cup (VAL-SCALE-011).

    Defaults: K≥3 public seeds, seq=256, token_budget=1_000_000, stage=promote,
    param_cap=350M. Seq/budget floors match P1 so promote is a ladder-stage raise
    on the same package pin, not a silent seq=128 trap.
    """
    seed_tuple = tuple(int(s) for s in seeds) if seeds is not None else SCALE_P2_SEEDS
    if require_public_k and len(seed_tuple) < OFFICIAL_MIN_PUBLIC_SEEDS:
        raise ValueError(
            f"public scale pin requires K≥{OFFICIAL_MIN_PUBLIC_SEEDS} seeds; "
            f"got K={len(seed_tuple)} (set require_public_k=False for provisional lab)"
        )
    resolved_seq = int(seq_len) if seq_len is not None else SCALE_P2_SEQ_LEN
    resolved_budget = int(token_budget) if token_budget is not None else SCALE_P2_TOKEN_BUDGET
    pin = promote_protocol_pin(
        seeds=seed_tuple,
        token_budget=resolved_budget,
        seq_len=resolved_seq,
        batch_size=int(batch_size) if batch_size is not None else SCALE_P2_BATCH_SIZE,
    )
    if require_p2_floor:
        assert_scale_p2_pin_floor(pin)
    return pin


def assert_scale_p2_pin_floor(pin: ProtocolPin) -> None:
    """Raise when pin is not a valid P2 promote pin (seq/budget floors + promote stage)."""
    reasons: list[str] = []
    if int(pin.seq_len) < SCALE_P2_SEQ_LEN_MIN:
        reasons.append(f"seq_len={pin.seq_len}<{SCALE_P2_SEQ_LEN_MIN} (P2 floor)")
    if int(pin.token_budget) < SCALE_P2_TOKEN_BUDGET_MIN:
        reasons.append(f"token_budget={pin.token_budget}<{SCALE_P2_TOKEN_BUDGET_MIN} (P2 floor)")
    stage = str(pin.param_ladder_stage).strip().lower()
    if stage != str(OFFICIAL_PROMOTE_STAGE):
        reasons.append(f"param_ladder_stage={pin.param_ladder_stage!r} != 'promote'")
    if int(pin.param_cap) < int(OFFICIAL_PROMOTE_PARAM_CAP):
        reasons.append(
            f"param_cap={pin.param_cap}<{OFFICIAL_PROMOTE_PARAM_CAP} (promote 350M ceiling)"
        )
    # Coerce stage label via ladder helper (fail-closed on unknown).
    _param_ladder.normalize_param_ladder_stage(pin.param_ladder_stage)
    if reasons:
        raise ValueError("P2 promote pin floor failed: " + "; ".join(reasons))


def protocol_pin_context_fields(pin: ProtocolPin) -> dict[str, Any]:
    """Map ProtocolPin knobs onto PrismContext / worker-plane field names.

    Ensures seq_len and token_budget pass through without seq=128-only traps.
    """
    return {
        "sequence_length": int(pin.seq_len),
        "token_budget": int(pin.token_budget),
        "max_parameters": int(pin.param_cap),
        "param_ladder_stage": str(pin.param_ladder_stage),
        "seed": int(pin.seeds[0]) if pin.seeds else 1337,
        "vocab_size": int(pin.vocab_size) if pin.vocab_size else OFFICIAL_DEFAULT_VOCAB_SIZE,
        "step_budget": pin.step_budget,
    }


def prism_context_from_protocol_pin(
    pin: ProtocolPin,
    *,
    seed: int | None = None,
    max_layers: int = 96,
    **overrides: Any,
) -> PrismContext:
    """Build a :class:`PrismContext` that honors pin seq_len and token_budget.

    Used by official compare / lab harness / worker-plane path wiring so raised
    P1 pin values are not dropped when constructing eval context.
    """
    fields = protocol_pin_context_fields(pin)
    if seed is not None:
        fields["seed"] = int(seed)
    fields["max_layers"] = int(max_layers)
    # Drop None step_budget so PrismContext default applies cleanly.
    if fields.get("step_budget") is None:
        fields.pop("step_budget", None)
    fields.update(overrides)
    return PrismContext(**fields)


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
            "scale_p1_pin": "scale_p1_protocol_pin",
            "scale_p2_pin": "scale_p2_protocol_pin",
            "promote_pin": "promote_protocol_pin",
            "core_families_p0": list(SCALE_P0_CORE_FAMILY_IDS),
            "core_families_p1": list(SCALE_P1_CORE_FAMILY_IDS),
            "crown_families_p2": list(SCALE_P2_CROWN_FAMILY_IDS),
            "core_families_p2": list(SCALE_P2_CORE_FAMILY_IDS),
            "frontier_families": list(FRONTIER_FAIR_EVAL_FAMILY_IDS),
        },
        "scale_helpers": {
            "module": "prism_challenge.evaluator.scale_eval",
            "p0_pin": "scale_p0_protocol_pin",
            "p1_pin": "scale_p1_protocol_pin",
            "p2_pin": "scale_p2_protocol_pin",
            "promote_pin": "promote_protocol_pin",
            "pin_fields": "scale_pin_fields",
            "public_ok": "scale_pin_public_ok",
            "p1_floor": "assert_scale_p1_pin_floor",
            "p2_floor": "assert_scale_p2_pin_floor",
            "context_from_pin": "prism_context_from_protocol_pin",
            "pin_to_context": "protocol_pin_context_fields",
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
        "p1_ladder": {
            "seq_len_min": SCALE_P1_SEQ_LEN_MIN,
            "seq_len_default": SCALE_P1_SEQ_LEN,
            "seq_len_target": SCALE_P1_SEQ_LEN_TARGET,
            "token_budget_min": SCALE_P1_TOKEN_BUDGET_MIN,
            "token_budget_default": SCALE_P1_TOKEN_BUDGET,
            "token_budget_high": SCALE_P1_TOKEN_BUDGET_HIGH,
            "notes": (
                "Raise ProtocolPin.seq_len / token_budget and settings.sequence_length / "
                "token_budget together; no seq=128-only trap on explore/official/lab paths."
            ),
        },
        "p2_ladder": {
            "param_ladder_stage": SCALE_P2_PARAM_STAGE,
            "param_cap": SCALE_P2_PARAM_CAP,
            "seq_len_default": SCALE_P2_SEQ_LEN,
            "token_budget_default": SCALE_P2_TOKEN_BUDGET,
            "crown_families": list(SCALE_P2_CROWN_FAMILY_IDS),
            "core_families": list(SCALE_P2_CORE_FAMILY_IDS),
            "notes": (
                "Promote 350M confirm/revoke cup uses matched pin (seq/budget from P1 floors) "
                "with param_ladder_stage=promote; wall never ranks."
            ),
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
    pin_p1 = scale_p1_protocol_pin()
    pin_p2 = scale_p2_protocol_pin()
    guard = scale_pin_public_ok(pin)
    guard_p1 = scale_pin_public_ok(pin_p1)
    guard_p2 = scale_pin_public_ok(pin_p2)
    return {
        "schema": "prism_scale_product_snapshot.v1",
        "pin": scale_pin_fields(pin),
        "p1_pin": scale_pin_fields(pin_p1),
        "p2_pin": scale_pin_fields(pin_p2),
        "public_guard": guard.as_dict(),
        "public_guard_p1": guard_p1.as_dict(),
        "public_guard_p2": guard_p2.as_dict(),
        "densify_entrypoints": densify_entrypoints(),
        "core_families_p0": list(SCALE_P0_CORE_FAMILY_IDS),
        "core_families_p1": list(SCALE_P1_CORE_FAMILY_IDS),
        "crown_families_p2": list(SCALE_P2_CROWN_FAMILY_IDS),
        "core_families_p2": list(SCALE_P2_CORE_FAMILY_IDS),
        "p1_ladder": {
            "seq_len_min": SCALE_P1_SEQ_LEN_MIN,
            "seq_len_default": SCALE_P1_SEQ_LEN,
            "seq_len_target": SCALE_P1_SEQ_LEN_TARGET,
            "token_budget_min": SCALE_P1_TOKEN_BUDGET_MIN,
            "token_budget_default": SCALE_P2_TOKEN_BUDGET,
            "token_budget_high": SCALE_P1_TOKEN_BUDGET_HIGH,
        },
        "p2_ladder": {
            "param_ladder_stage": SCALE_P2_PARAM_STAGE,
            "param_cap": SCALE_P2_PARAM_CAP,
            "seq_len_default": SCALE_P2_SEQ_LEN,
            "token_budget_default": SCALE_P2_TOKEN_BUDGET,
        },
        "tee_package_absent": tee_package_absent(),
        "wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
        "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
    }
