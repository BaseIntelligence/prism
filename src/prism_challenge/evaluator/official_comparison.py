"""Prism Official Comparison Protocol v1 ranking helpers.

Architecture/lab surface that ranks two (or more) challenge-owned score records under
the protocol defined in ``docs/official-comparison.md``:

* PRIMARY: held-out generalization (``heldout_delta`` preferred; higher better)
* SECONDARY: Prism-recomputed prequential bits-per-byte (lower better)
* Anti-overfit: memorization gap flag + step-0 anomaly remain active
* Wall-clock and miner self-report may be *recorded* but never enter the rank key

Production leaderboard scoring (``score_prequential_bpb`` / ``final_score`` with bpb
primary) is intentionally left unchanged. Official Comparison is a separate comparison
mode used by offline a-vs-b harnesses and pure unit fixtures.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from .scoring import (
    HELDOUT_DELTA_BPB_EPSILON,
    MEMORIZATION_GAP_THRESHOLD_BPB,
    MEMORIZATION_PENALTY_FACTOR,
    PrequentialBpbScore,
    score_prequential_bpb,
)

# --- Protocol pin constants (VAL-COMP-006 / docs ProtocolPin sketch) ---------------------------
PROTOCOL_ID = "prism_official_compare.v1"
PROTOCOL_SCHEMA = "prism_official_compare.v1"
PRIMARY_FORM_HELDOUT_DELTA: Literal["heldout_delta"] = "heldout_delta"
PRIMARY_FORM_VAL_BPB: Literal["val_bpb_trained"] = "val_bpb_trained"
PrimaryForm = Literal["heldout_delta", "val_bpb_trained"]

# Multi-metric scorecard annex v1.1 (docs §14 / VAL-SCORE-*). Additive on PROTOCOL_ID;
# not a sole weighted-crown rewrite of emission leaderboard.
SCORECARD_ID = "multimetric.v1.1"
SCORECARD_SCHEMA = "prism_scorecard_annex.v1.1"
SCORECARD_TIERS: tuple[str, ...] = ("V", "P", "S", "R")

# Near-tie bands (docs §5.5). Secondary reuses the held-out epsilon scale.
OFFICIAL_EPS_HELDOUT = HELDOUT_DELTA_BPB_EPSILON  # 5e-3
OFFICIAL_EPS_BPB = HELDOUT_DELTA_BPB_EPSILON  # 5e-3
# Polar band on long-ctx suite mean accuracy (docs §14.5).
OFFICIAL_EPS_LONG_CTX = 0.02
# Seed-scale absolute long-ctx floor when suite is enabled (design §3.4).
OFFICIAL_LONG_CTX_FLOOR = 0.15

# Matched-budget protocol defaults (fair fixed pin for ArchCompare / TrainCompare).
OFFICIAL_PARAM_CAP = 150_000_000
OFFICIAL_DEFAULT_TOKEN_BUDGET = 500_000
OFFICIAL_DEFAULT_VAL_BYTE_BUDGET = 65_536
OFFICIAL_DEFAULT_SEEDS: tuple[int, ...] = (1337, 2027, 4242)
OFFICIAL_MIN_PUBLIC_SEEDS = 3
OFFICIAL_DEFAULT_SEQ_LEN = 128
OFFICIAL_DEFAULT_BATCH_SIZE = 4
OFFICIAL_DEFAULT_TOKENIZER = "gpt2"
OFFICIAL_DEFAULT_VOCAB_SIZE = 50_304
OFFICIAL_SCORED_NPROC = 1
# Wall-clock remains a safety watchdog only (docs §4 #6, §5.4, VAL-COMP-011).
OFFICIAL_WALL_CLOCK_NEVER_RANKS = True
OFFICIAL_MEMORIZATION_GAP_THRESHOLD_BPB = MEMORIZATION_GAP_THRESHOLD_BPB
OFFICIAL_MEMORIZATION_PENALTY_FACTOR = MEMORIZATION_PENALTY_FACTOR

# Honesty residual for prior LAB-GPU K=1 short-ctx observations (VAL-SCORE-012 related).
SCORECARD_PROVISIONAL_HONESTY_NOTE = (
    "prior LAB-GPU K=1 short-ctx mamba heldout lead is provisional only; "
    "scorecard required for full claim language"
)

CompareWinner = Literal["a", "b", "tie"]
CompareReason = Literal[
    "invalid",
    "step0_anomaly",
    "primary_heldout",
    "secondary_bpb",
    "anti_overfit",
    "multi_seed_residual",
    "tie",
    "tie_polar",
]
AxisLead = Literal["a", "b", "tie", "missing"]


@dataclass(frozen=True)
class ProtocolPin:
    """Frozen matched-budget pin both sides of an official compare must share.

    ``wall_clock_seconds`` is diagnostic/safety only and never a rank key.
    """

    protocol_id: str = PROTOCOL_ID
    token_budget: int = OFFICIAL_DEFAULT_TOKEN_BUDGET
    step_budget: int | None = None
    wall_clock_seconds: float | None = 1200.0
    seeds: tuple[int, ...] = OFFICIAL_DEFAULT_SEEDS
    param_cap: int = OFFICIAL_PARAM_CAP
    seq_len: int = OFFICIAL_DEFAULT_SEQ_LEN
    batch_size: int = OFFICIAL_DEFAULT_BATCH_SIZE
    tokenizer: str = OFFICIAL_DEFAULT_TOKENIZER
    vocab_size: int = OFFICIAL_DEFAULT_VOCAB_SIZE
    scored_nproc: int = OFFICIAL_SCORED_NPROC
    val_byte_budget: int = OFFICIAL_DEFAULT_VAL_BYTE_BUDGET
    gap_threshold_bpb: float = OFFICIAL_MEMORIZATION_GAP_THRESHOLD_BPB
    primary_form: PrimaryForm = PRIMARY_FORM_HELDOUT_DELTA
    force_iter_train_batches: bool = True
    require_trained_state: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "token_budget": self.token_budget,
            "step_budget": self.step_budget,
            "wall_clock_seconds": self.wall_clock_seconds,
            "seeds": list(self.seeds),
            "param_cap": self.param_cap,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "tokenizer": self.tokenizer,
            "vocab_size": self.vocab_size,
            "scored_nproc": self.scored_nproc,
            "val_byte_budget": self.val_byte_budget,
            "gap_threshold_bpb": self.gap_threshold_bpb,
            "primary_form": self.primary_form,
            "force_iter_train_batches": self.force_iter_train_batches,
            "require_trained_state": self.require_trained_state,
            "wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
        }


@dataclass(frozen=True)
class OfficialScoreRecord:
    """One side of an official comparison (single-seed or multi-seed aggregate).

    Rank uses only challenge-owned fields: primary held-out form, recomputed bpb,
    anti-overfit flags, and optional multi-seed residual. Diagnostics (wall-clock,
    miner self-report) may be present for observability but are ignored by
    :func:`official_rank_key` and :func:`compare_official`.

    Scorecard annex v1.1 adds optional multi-metric fields (long-ctx, sample_eff,
    efficiency, validity, stability). These never rewrite production leaderboard
    emission; they feed :func:`build_scorecard_annex` / polar conflict only.
    """

    label: str
    bpb: float
    primary_form: PrimaryForm = PRIMARY_FORM_HELDOUT_DELTA
    heldout_delta: float | None = None
    val_bpb_trained: float | None = None
    memorization_flag: bool = False
    train_heldout_gap: float | None = None
    step0_anomaly: bool = False
    valid: bool = True
    seed_count: int = 1
    bpb_std: float | None = None
    overfit_rate: float = 0.0
    # Diagnostics: never part of official_rank_key
    wall_clock_seconds: float | None = None
    miner_reported_bpb: float | None = None
    miner_reported_final_score: float | None = None
    flags: tuple[str, ...] = ()
    # --- multimetric.v1.1 scorecard fields (additive; placeholders by default) ---
    # Validity (V) per-side residual flags. None means "not yet measured / inherit.".
    stop_token_budget: bool | None = None
    finite_bpb: bool | None = None
    param_cap_ok: bool | None = None
    matched_pin: bool | None = None
    challenge_authored: bool = True
    force_instrument: bool | None = None
    # Multi-seed residual (K and scales for heldout/bpb when multi-seed).
    heldout_std: float | None = None
    # Long-ctx suite (P) — null when suite disabled / not-run.
    long_ctx_score: float | None = None
    long_ctx_needle: float | None = None
    long_ctx_mqar: float | None = None
    long_ctx_induction_copy: float | None = None
    lag_nll: float | None = None
    long_ctx_enabled: bool = False
    long_ctx_floor_pass: bool | None = None
    # Sample-efficiency placeholders (mark vector / AUC filled by later suite track).
    sample_eff_auc: float | None = None
    sample_eff_marks: tuple[float, ...] | None = None
    # Efficiency annex (S) — diagnostic Pareto only.
    params: int | None = None
    peak_vram_gib: float | None = None
    tokens_per_s: float | None = None
    # Stability residual (R).
    nan_inf_events: int | None = None
    grad_spike_rate: float | None = None
    instability_flag: bool = False

    @property
    def primary_value(self) -> float | None:
        if self.primary_form == PRIMARY_FORM_HELDOUT_DELTA:
            return self.heldout_delta
        return self.val_bpb_trained

    @property
    def is_public_multi_seed(self) -> bool:
        """True when clean seed_count meets public non-provisional K≥3 (VAL-SCORE-008)."""
        return (
            self.valid and not self.step0_anomaly and self.seed_count >= OFFICIAL_MIN_PUBLIC_SEEDS
        )

    @property
    def multi_seed_provisional(self) -> bool:
        """True when K_clean < public minimum (provisional lab posture only)."""
        return self.seed_count < OFFICIAL_MIN_PUBLIC_SEEDS

    def primary_is_better(self, other: OfficialScoreRecord, *, eps: float) -> bool | None:
        """Return True if self strictly beats other on the primary axis by more than eps.

        None means missing primary on either side (secondary decides / invalid residual).
        """
        a = self.primary_value
        b = other.primary_value
        if a is None or b is None:
            return None
        if self.primary_form != other.primary_form:
            return None
        if self.primary_form == PRIMARY_FORM_HELDOUT_DELTA:
            # Higher heldout_delta is better.
            if a > b + eps:
                return True
            if b > a + eps:
                return False
            return None  # near-tie treated as not strict
        # val_bpb_trained: lower is better
        if a < b - eps:
            return True
        if b < a - eps:
            return False
        return None


@dataclass(frozen=True)
class CompareResult:
    """Deterministic outcome of :func:`compare_official` / :func:`compare_official_scorecard`."""

    winner: CompareWinner
    reason: CompareReason
    rule: str = "heldout_primary_then_bpb_secondary"
    eps_heldout: float = OFFICIAL_EPS_HELDOUT
    eps_bpb: float = OFFICIAL_EPS_BPB
    detail: str = ""
    # Scorecard polar annex (VAL-SCORE-003): false special-cases when not polar.
    tie_polar: bool = False
    crown_allowed: bool = True
    eps_long_ctx: float = OFFICIAL_EPS_LONG_CTX
    scorecard_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "winner": self.winner,
            "reason": self.reason,
            "rule": self.rule,
            "eps_heldout": self.eps_heldout,
            "eps_bpb": self.eps_bpb,
            "detail": self.detail,
            "tie_polar": self.tie_polar,
            "crown_allowed": self.crown_allowed,
        }
        if self.scorecard_id is not None:
            payload["scorecard_id"] = self.scorecard_id
            payload["eps_long_ctx"] = self.eps_long_ctx
        return payload


@dataclass(frozen=True)
class ValidityGateRecord:
    """Validity (V) tier residual for one side or the compare pair (VAL-SCORE-004)."""

    stop_token_budget: bool
    finite_bpb: bool
    step0_clean: bool
    param_cap: bool
    matched_pin: bool
    multi_seed_K: int
    multi_seed_public: bool
    multi_seed_provisional: bool
    challenge_authored: bool
    force_instrument: bool
    ok: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_token_budget": self.stop_token_budget,
            "finite_bpb": self.finite_bpb,
            "step0_clean": self.step0_clean,
            "param_cap": self.param_cap,
            "matched_pin": self.matched_pin,
            "multi_seed_K": self.multi_seed_K,
            "multi_seed_public": self.multi_seed_public,
            "multi_seed_provisional": self.multi_seed_provisional,
            "challenge_authored": self.challenge_authored,
            "force_instrument": self.force_instrument,
            "ok": self.ok,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PolarConflictResult:
    """Short-gen vs long-ctx polar conflict decision (VAL-SCORE-003)."""

    tie_polar: bool
    crown_allowed: bool
    short_gen_lead: AxisLead
    long_ctx_lead: AxisLead
    reason: str | None
    eps_heldout: float = OFFICIAL_EPS_HELDOUT
    eps_long_ctx: float = OFFICIAL_EPS_LONG_CTX
    long_ctx_enabled_and_filled: bool = False
    floor_veto_a: bool = False
    floor_veto_b: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "tie_polar": self.tie_polar,
            "crown_allowed": self.crown_allowed,
            "short_gen_lead": self.short_gen_lead,
            "long_ctx_lead": self.long_ctx_lead,
            "reason": self.reason,
            "eps_heldout": self.eps_heldout,
            "eps_long_ctx": self.eps_long_ctx,
            "long_ctx_enabled_and_filled": self.long_ctx_enabled_and_filled,
            "floor_veto_a": self.floor_veto_a,
            "floor_veto_b": self.floor_veto_b,
        }


def official_record_from_score(
    score: PrequentialBpbScore,
    *,
    label: str = "run",
    primary_form: PrimaryForm = PRIMARY_FORM_HELDOUT_DELTA,
    wall_clock_seconds: float | None = None,
    miner_reported_bpb: float | None = None,
    miner_reported_final_score: float | None = None,
    seed_count: int = 1,
    bpb_std: float | None = None,
    overfit_rate: float | None = None,
    stop_token_budget: bool | None = None,
    param_cap_ok: bool | None = None,
    matched_pin: bool | None = None,
    challenge_authored: bool = True,
    force_instrument: bool | None = None,
    long_ctx_score: float | None = None,
    long_ctx_needle: float | None = None,
    long_ctx_mqar: float | None = None,
    long_ctx_induction_copy: float | None = None,
    lag_nll: float | None = None,
    long_ctx_enabled: bool = False,
    sample_eff_auc: float | None = None,
    sample_eff_marks: tuple[float, ...] | None = None,
    params: int | None = None,
    peak_vram_gib: float | None = None,
    tokens_per_s: float | None = None,
    nan_inf_events: int | None = None,
    grad_spike_rate: float | None = None,
    instability_flag: bool = False,
    heldout_std: float | None = None,
) -> OfficialScoreRecord:
    """Project a challenge-owned :class:`PrequentialBpbScore` into an official record.

    Miner-reported numbers may be attached as diagnostics only. Scorecard fields
    default to honest placeholders (null / not-run) unless provided by the caller.
    """
    finite = math.isfinite(score.bpb) and score.bpb > 0.0
    valid = finite and not score.anomaly and score.anti_cheat_multiplier > 0.0
    long_ctx_floor_pass: bool | None = None
    if long_ctx_enabled and long_ctx_score is not None and math.isfinite(long_ctx_score):
        long_ctx_floor_pass = float(long_ctx_score) >= OFFICIAL_LONG_CTX_FLOOR
    # Missing primary under the requested form still yields a record; compare falls through.
    return OfficialScoreRecord(
        label=label,
        bpb=float(score.bpb),
        primary_form=primary_form,
        heldout_delta=score.heldout_delta,
        val_bpb_trained=score.val_bpb_trained,
        memorization_flag=bool(score.memorization_flag),
        train_heldout_gap=score.train_heldout_gap,
        step0_anomaly=bool(score.anomaly),
        valid=valid,
        seed_count=seed_count,
        bpb_std=bpb_std,
        overfit_rate=(
            float(overfit_rate)
            if overfit_rate is not None
            else (1.0 if score.memorization_flag else 0.0)
        ),
        wall_clock_seconds=wall_clock_seconds,
        miner_reported_bpb=miner_reported_bpb,
        miner_reported_final_score=miner_reported_final_score,
        flags=tuple(score.flags),
        stop_token_budget=stop_token_budget,
        finite_bpb=finite,
        param_cap_ok=param_cap_ok,
        matched_pin=matched_pin,
        challenge_authored=challenge_authored,
        force_instrument=force_instrument,
        heldout_std=heldout_std,
        long_ctx_score=long_ctx_score,
        long_ctx_needle=long_ctx_needle,
        long_ctx_mqar=long_ctx_mqar,
        long_ctx_induction_copy=long_ctx_induction_copy,
        lag_nll=lag_nll,
        long_ctx_enabled=long_ctx_enabled,
        long_ctx_floor_pass=long_ctx_floor_pass,
        sample_eff_auc=sample_eff_auc,
        sample_eff_marks=sample_eff_marks,
        params=params,
        peak_vram_gib=peak_vram_gib,
        tokens_per_s=tokens_per_s,
        nan_inf_events=nan_inf_events,
        grad_spike_rate=grad_spike_rate,
        instability_flag=instability_flag,
    )


def official_record_from_manifest(
    manifest: Mapping[str, Any],
    *,
    label: str = "run",
    primary_form: PrimaryForm = PRIMARY_FORM_HELDOUT_DELTA,
    miner_reported: Mapping[str, Any] | None = None,
    skip_heldout: bool = False,
) -> OfficialScoreRecord:
    """Build an official record from a challenge-owned v2 manifest via Prism recompute.

    Always recomputes secondary bpb (and anti-overfit) with :func:`score_prequential_bpb`.
    Any miner-provided metrics mapping is accepted only as non-authoritative diagnostics
    (VAL-COMP-003): miner self-report never becomes the rank key.
    """
    score = score_prequential_bpb(manifest, skip_heldout=skip_heldout)
    wall_clock: float | None = None
    compute = manifest.get("compute")
    if isinstance(compute, Mapping):
        raw_wc = compute.get("wall_clock_seconds")
        if isinstance(raw_wc, int | float) and not isinstance(raw_wc, bool):
            wall_clock = float(raw_wc)
    miner_bpb: float | None = None
    miner_fs: float | None = None
    if isinstance(miner_reported, Mapping):
        raw_bpb = miner_reported.get("bpb", miner_reported.get("prequential_bpb"))
        if isinstance(raw_bpb, int | float) and not isinstance(raw_bpb, bool):
            miner_bpb = float(raw_bpb)
        raw_fs = miner_reported.get("final_score")
        if isinstance(raw_fs, int | float) and not isinstance(raw_fs, bool):
            miner_fs = float(raw_fs)
    return official_record_from_score(
        score,
        label=label,
        primary_form=primary_form,
        wall_clock_seconds=wall_clock,
        miner_reported_bpb=miner_bpb,
        miner_reported_final_score=miner_fs,
    )


def aggregate_official_records(
    records: Iterable[OfficialScoreRecord],
    *,
    label: str,
    primary_form: PrimaryForm = PRIMARY_FORM_HELDOUT_DELTA,
) -> OfficialScoreRecord:
    """Mean multi-seed aggregate used by residual multi-seed official claims (docs §5.1/5.2).

    Invalid / step-0 seeds are dropped from the mean; if none remain, the aggregate is
    marked invalid so compare fails closed. Multi-seed K is ``seed_count`` of clean
    seeds; heldout standard deviation is reported for scorecard residual (VAL-SCORE-008).
    """
    clean = [r for r in records if r.valid and not r.step0_anomaly]
    material = list(records)
    if not clean:
        return OfficialScoreRecord(
            label=label,
            bpb=float("inf"),
            primary_form=primary_form,
            heldout_delta=None,
            val_bpb_trained=None,
            memorization_flag=True,
            step0_anomaly=True,
            valid=False,
            seed_count=0,
            overfit_rate=1.0,
            flags=("no_clean_seeds",),
            finite_bpb=False,
            challenge_authored=all(r.challenge_authored for r in material) if material else True,
            instability_flag=any(r.instability_flag for r in material),
        )
    bpb_vals = [float(r.bpb) for r in clean]
    mean_bpb = sum(bpb_vals) / len(bpb_vals)
    variance = sum((v - mean_bpb) ** 2 for v in bpb_vals) / len(bpb_vals)
    bpb_std = math.sqrt(variance)
    overfit_rate = sum(1.0 for r in clean if r.memorization_flag) / len(clean)

    if primary_form == PRIMARY_FORM_HELDOUT_DELTA:
        deltas = [float(r.heldout_delta) for r in clean if r.heldout_delta is not None]
        mean_delta = (sum(deltas) / len(deltas)) if deltas else None
        if len(deltas) > 1:
            d_mean = mean_delta if mean_delta is not None else 0.0
            heldout_std = math.sqrt(sum((d - d_mean) ** 2 for d in deltas) / len(deltas))
        else:
            heldout_std = 0.0 if len(deltas) == 1 else None
        mean_val = None
    else:
        vals = [float(r.val_bpb_trained) for r in clean if r.val_bpb_trained is not None]
        mean_val = (sum(vals) / len(vals)) if vals else None
        mean_delta = None
        heldout_std = None

    # Gap mean for residual diagnostics only.
    gaps = [float(r.train_heldout_gap) for r in clean if r.train_heldout_gap is not None]
    mean_gap = (sum(gaps) / len(gaps)) if gaps else None
    memo = overfit_rate > 0.5 or (
        mean_gap is not None and mean_gap > OFFICIAL_MEMORIZATION_GAP_THRESHOLD_BPB
    )

    def _all_or_none(attr: str) -> bool | None:
        vals_attr = [getattr(r, attr) for r in clean]
        if all(v is True for v in vals_attr):
            return True
        if all(v is False for v in vals_attr):
            return False
        if all(v is None for v in vals_attr):
            return None
        # Mixed: treat as False when any measured False, else True if any True.
        if any(v is False for v in vals_attr):
            return False
        if any(v is True for v in vals_attr):
            return True
        return None

    def _mean_optional(attr: str) -> float | None:
        nums = [
            float(getattr(r, attr))
            for r in clean
            if getattr(r, attr) is not None and math.isfinite(float(getattr(r, attr)))
        ]
        if not nums:
            return None
        return sum(nums) / len(nums)

    long_ctx_enabled = any(r.long_ctx_enabled for r in clean)
    long_ctx_score = _mean_optional("long_ctx_score")
    long_ctx_floor_pass: bool | None = None
    if long_ctx_enabled and long_ctx_score is not None:
        long_ctx_floor_pass = float(long_ctx_score) >= OFFICIAL_LONG_CTX_FLOOR

    def _aggregate_sample_eff_marks(
        records: list[OfficialScoreRecord],
    ) -> tuple[float, ...] | None:
        """Mean mark vector across seeds when every clean seed reports same-length marks."""
        mark_lists = [
            list(r.sample_eff_marks)
            for r in records
            if r.sample_eff_marks is not None and len(r.sample_eff_marks) > 0
        ]
        if not mark_lists:
            return None
        width = len(mark_lists[0])
        if any(len(m) != width for m in mark_lists):
            return None
        means: list[float] = []
        for col in range(width):
            col_vals = [float(m[col]) for m in mark_lists if math.isfinite(float(m[col]))]
            if not col_vals:
                return None
            means.append(sum(col_vals) / len(col_vals))
        return tuple(means)

    nan_events = [r.nan_inf_events for r in clean if r.nan_inf_events is not None]
    nan_sum = sum(nan_events) if nan_events else None
    instability = any(r.instability_flag for r in clean) or (nan_sum is not None and nan_sum > 0)

    params_vals = [r.params for r in clean if r.params is not None]
    params_mean = int(round(sum(params_vals) / len(params_vals))) if params_vals else None

    return OfficialScoreRecord(
        label=label,
        bpb=mean_bpb,
        primary_form=primary_form,
        heldout_delta=mean_delta,
        val_bpb_trained=mean_val,
        memorization_flag=memo,
        train_heldout_gap=mean_gap,
        step0_anomaly=False,
        valid=True,
        seed_count=len(clean),
        bpb_std=bpb_std,
        overfit_rate=overfit_rate,
        flags=tuple(sorted({f for r in clean for f in r.flags})),
        stop_token_budget=_all_or_none("stop_token_budget"),
        finite_bpb=True,
        param_cap_ok=_all_or_none("param_cap_ok"),
        matched_pin=_all_or_none("matched_pin"),
        challenge_authored=all(r.challenge_authored for r in clean),
        force_instrument=_all_or_none("force_instrument"),
        heldout_std=heldout_std,
        long_ctx_score=long_ctx_score,
        long_ctx_needle=_mean_optional("long_ctx_needle"),
        long_ctx_mqar=_mean_optional("long_ctx_mqar"),
        long_ctx_induction_copy=_mean_optional("long_ctx_induction_copy"),
        lag_nll=_mean_optional("lag_nll"),
        long_ctx_enabled=long_ctx_enabled,
        long_ctx_floor_pass=long_ctx_floor_pass,
        sample_eff_auc=_mean_optional("sample_eff_auc"),
        sample_eff_marks=_aggregate_sample_eff_marks(clean),
        params=params_mean,
        peak_vram_gib=_mean_optional("peak_vram_gib"),
        tokens_per_s=_mean_optional("tokens_per_s"),
        nan_inf_events=nan_sum,
        grad_spike_rate=_mean_optional("grad_spike_rate"),
        instability_flag=instability,
    )


def official_rank_key(
    record: OfficialScoreRecord,
    *,
    eps_heldout: float = OFFICIAL_EPS_HELDOUT,
    eps_bpb: float = OFFICIAL_EPS_BPB,
) -> tuple[Any, ...]:
    """Total ascending sort key: smaller key ranks better under Official Comparison.

    Order (docs §5.5):
    1. validity / step-0 (invalid and step-0 lose)
    2. primary held-out (higher heldout_delta better; lower val_bpb better)
    3. secondary prequential bpb (lower better)
    4. anti-overfit (no memorization flag better; lower overfit_rate; lower gap)
    5. multi-seed residual (lower bpb_std better)
    6. label for total order

    Wall-clock and miner-reported fields are **never** part of this key (VAL-COMP-011/003).
    """
    del eps_heldout, eps_bpb  # included in signature for API symmetry; bands used by compare
    # Intentionally do NOT surface ``record.wall_clock_seconds`` or miner fields.
    _ = record.wall_clock_seconds
    _ = record.miner_reported_bpb
    _ = record.miner_reported_final_score

    invalid = 0 if (record.valid and not record.step0_anomaly) else 1
    step0 = 1 if record.step0_anomaly else 0

    if record.primary_form == PRIMARY_FORM_HELDOUT_DELTA:
        # Higher delta better → sort by -delta; missing primary goes to last among valid.
        if record.heldout_delta is None or not math.isfinite(record.heldout_delta):
            primary_missing = 1
            primary_sort = 0.0
        else:
            primary_missing = 0
            primary_sort = -float(record.heldout_delta)
    else:
        if record.val_bpb_trained is None or not math.isfinite(record.val_bpb_trained):
            primary_missing = 1
            primary_sort = 0.0
        else:
            primary_missing = 0
            primary_sort = float(record.val_bpb_trained)

    bpb_sort = float(record.bpb) if math.isfinite(record.bpb) else float("inf")
    memo = 1 if record.memorization_flag else 0
    overfit = float(record.overfit_rate)
    gap = (
        float(record.train_heldout_gap)
        if record.train_heldout_gap is not None and math.isfinite(record.train_heldout_gap)
        else 0.0
    )
    residual = (
        float(record.bpb_std)
        if record.bpb_std is not None and math.isfinite(record.bpb_std)
        else 0.0
    )
    return (
        invalid,
        step0,
        primary_missing,
        primary_sort,
        bpb_sort,
        memo,
        overfit,
        gap,
        residual,
        record.label,
    )


def compare_official(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    eps_heldout: float = OFFICIAL_EPS_HELDOUT,
    eps_bpb: float = OFFICIAL_EPS_BPB,
) -> CompareResult:
    """Pure deterministic A-vs-B comparison under Official Comparison Protocol v1.

    Never consults wall-clock or miner self-report diagnostics. Step-0 / invalid sides lose
    to a clean opponent; both invalid → tie on the invalid residual.
    """
    ka = official_rank_key(a, eps_heldout=eps_heldout, eps_bpb=eps_bpb)
    kb = official_rank_key(b, eps_heldout=eps_heldout, eps_bpb=eps_bpb)

    # 1) Validity / step-0
    a_ok = a.valid and not a.step0_anomaly
    b_ok = b.valid and not b.step0_anomaly
    if a_ok and not b_ok:
        return CompareResult(
            winner="a",
            reason="step0_anomaly" if b.step0_anomaly else "invalid",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail=f"{b.label} disqualified (step0/invalid)",
        )
    if b_ok and not a_ok:
        return CompareResult(
            winner="b",
            reason="step0_anomaly" if a.step0_anomaly else "invalid",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail=f"{a.label} disqualified (step0/invalid)",
        )
    if not a_ok and not b_ok:
        return CompareResult(
            winner="tie",
            reason="invalid",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail="both sides invalid or step0",
        )

    # 2) PRIMARY held-out (with near-tie band)
    primary = a.primary_is_better(b, eps=eps_heldout)
    if primary is True:
        return CompareResult(
            winner="a",
            reason="primary_heldout",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail=f"primary {a.primary_form}: {a.label} better than {b.label}",
        )
    if primary is False:
        return CompareResult(
            winner="b",
            reason="primary_heldout",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail=f"primary {b.primary_form}: {b.label} better than {a.label}",
        )

    # 3) SECONDARY recomputed bpb (lower better, with near-tie band)
    if math.isfinite(a.bpb) and math.isfinite(b.bpb):
        if a.bpb < b.bpb - eps_bpb:
            return CompareResult(
                winner="a",
                reason="secondary_bpb",
                eps_heldout=eps_heldout,
                eps_bpb=eps_bpb,
                detail=f"bpb {a.bpb:.6f} < {b.bpb:.6f}",
            )
        if b.bpb < a.bpb - eps_bpb:
            return CompareResult(
                winner="b",
                reason="secondary_bpb",
                eps_heldout=eps_heldout,
                eps_bpb=eps_bpb,
                detail=f"bpb {b.bpb:.6f} < {a.bpb:.6f}",
            )

    # 4) Anti-overfit residual
    if a.memorization_flag != b.memorization_flag:
        winner: CompareWinner = "a" if (not a.memorization_flag) else "b"
        return CompareResult(
            winner=winner,
            reason="anti_overfit",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail="memorization_flag residual",
        )
    if a.overfit_rate + 1e-12 < b.overfit_rate:
        return CompareResult(
            winner="a",
            reason="anti_overfit",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail="lower overfit_rate",
        )
    if b.overfit_rate + 1e-12 < a.overfit_rate:
        return CompareResult(
            winner="b",
            reason="anti_overfit",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail="lower overfit_rate",
        )

    # 5) Multi-seed residual (lower bpb_std when both sides report multi-seed aggregates)
    a_std = a.bpb_std
    b_std = b.bpb_std
    if (
        a.seed_count > 1
        and b.seed_count > 1
        and a_std is not None
        and b_std is not None
        and math.isfinite(a_std)
        and math.isfinite(b_std)
    ):
        if a_std < b_std - 1e-12:
            return CompareResult(
                winner="a",
                reason="multi_seed_residual",
                eps_heldout=eps_heldout,
                eps_bpb=eps_bpb,
                detail="lower multi-seed bpb variance",
            )
        if b_std < a_std - 1e-12:
            return CompareResult(
                winner="b",
                reason="multi_seed_residual",
                eps_heldout=eps_heldout,
                eps_bpb=eps_bpb,
                detail="lower multi-seed bpb variance",
            )

    # Scientific TIE under the documented near-tie bands. Labels and wall-clock are not
    # consulted (rank_leaderboard earliest-commit total order is a different surface).
    _ = (ka, kb)  # preserved for callers that inspect pure rank keys separately
    return CompareResult(
        winner="tie",
        reason="tie",
        eps_heldout=eps_heldout,
        eps_bpb=eps_bpb,
        detail="primary+secondary+anti-overfit residual equal",
    )


def rank_official(records: Iterable[OfficialScoreRecord]) -> list[OfficialScoreRecord]:
    """Sort official records best-first via :func:`official_rank_key`."""
    return sorted(records, key=official_rank_key)


def protocol_budget_constants() -> dict[str, Any]:
    """Surface matched budget + denominator knobs for pin / docs / tests (VAL-COMP-006)."""
    pin = ProtocolPin()
    return cast(
        dict[str, Any],
        {
            **pin.as_dict(),
            "eps_heldout": OFFICIAL_EPS_HELDOUT,
            "eps_bpb": OFFICIAL_EPS_BPB,
            "eps_long_ctx": OFFICIAL_EPS_LONG_CTX,
            "long_ctx_floor": OFFICIAL_LONG_CTX_FLOOR,
            "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
            "memorization_gap_threshold_bpb": OFFICIAL_MEMORIZATION_GAP_THRESHOLD_BPB,
            "memorization_penalty_factor": OFFICIAL_MEMORIZATION_PENALTY_FACTOR,
            "wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
            "protocol_schema": PROTOCOL_SCHEMA,
            "scorecard_id": SCORECARD_ID,
            "scorecard_schema": SCORECARD_SCHEMA,
            "scorecard_tiers": list(SCORECARD_TIERS),
        },
    )


def _bool_gate(value: bool | None, *, default: bool) -> bool:
    """Map optional measured gate → bool; unmeasured inherits ``default`` honestly."""
    if value is None:
        return default
    return bool(value)


def evaluate_validity_gates(
    record: OfficialScoreRecord,
    *,
    matched_pin: bool | None = None,
    param_cap_ok: bool | None = None,
    force_instrument: bool | None = None,
    stop_token_budget: bool | None = None,
    require_public_multi_seed: bool = False,
) -> ValidityGateRecord:
    """Record Validity (V) gates for one official score side (VAL-SCORE-004).

    Unmeasured optional fields inherit safe defaults for fixture/lab paths:
    stop_token_budget / param_cap / force_instrument / matched_pin default True
    when the caller did not equip them, because challenge-owned synthetic paths
    already stop on token_budget under the protocol pin. Explicit False always
    fails the gate.
    """
    reasons: list[str] = []
    finite = _bool_gate(
        record.finite_bpb,
        default=(math.isfinite(record.bpb) and record.bpb > 0.0),
    )
    step0_clean = not record.step0_anomaly and record.valid
    stop_ok = _bool_gate(
        stop_token_budget if stop_token_budget is not None else record.stop_token_budget,
        default=True,
    )
    param_ok = _bool_gate(
        param_cap_ok if param_cap_ok is not None else record.param_cap_ok,
        default=True,
    )
    pin_ok = _bool_gate(
        matched_pin if matched_pin is not None else record.matched_pin,
        default=True,
    )
    force_ok = _bool_gate(
        force_instrument if force_instrument is not None else record.force_instrument,
        default=True,
    )
    challenge_ok = bool(record.challenge_authored)
    multi_k = int(record.seed_count)
    public = multi_k >= OFFICIAL_MIN_PUBLIC_SEEDS and step0_clean and finite
    provisional = multi_k < OFFICIAL_MIN_PUBLIC_SEEDS

    if not stop_ok:
        reasons.append("stop_token_budget")
    if not finite:
        reasons.append("finite_bpb")
    if not step0_clean:
        reasons.append("step0_clean")
    if not param_ok:
        reasons.append("param_cap")
    if not pin_ok:
        reasons.append("matched_pin")
    if not challenge_ok:
        reasons.append("challenge_authored")
    if not force_ok:
        reasons.append("force_instrument")
    if require_public_multi_seed and provisional:
        reasons.append("multi_seed_K_provisional")

    # Provisional multi-seed alone does not fail V.ok for lab paths unless required.
    hard_reasons = [r for r in reasons if r != "multi_seed_K_provisional"]
    ok = not hard_reasons and (
        "multi_seed_K_provisional" not in reasons or not require_public_multi_seed
    )

    return ValidityGateRecord(
        stop_token_budget=stop_ok,
        finite_bpb=finite,
        step0_clean=step0_clean,
        param_cap=param_ok,
        matched_pin=pin_ok,
        multi_seed_K=multi_k,
        multi_seed_public=public,
        multi_seed_provisional=provisional,
        challenge_authored=challenge_ok,
        force_instrument=force_ok,
        ok=ok,
        reasons=tuple(reasons),
    )


def evaluate_pair_validity(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    matched_pin: bool = True,
    require_public_multi_seed: bool = False,
) -> dict[str, Any]:
    """Pair-level V residual used by scorecard annex (both sides + conjunction)."""
    va = evaluate_validity_gates(
        a, matched_pin=matched_pin, require_public_multi_seed=require_public_multi_seed
    )
    vb = evaluate_validity_gates(
        b, matched_pin=matched_pin, require_public_multi_seed=require_public_multi_seed
    )
    k_a = va.multi_seed_K
    k_b = vb.multi_seed_K
    k_min = min(k_a, k_b)
    public = va.multi_seed_public and vb.multi_seed_public and k_min >= OFFICIAL_MIN_PUBLIC_SEEDS
    provisional = k_min < OFFICIAL_MIN_PUBLIC_SEEDS
    return {
        "a": va.as_dict(),
        "b": vb.as_dict(),
        "matched_pin": matched_pin,
        "stop_token_budget": va.stop_token_budget and vb.stop_token_budget,
        "finite_bpb": va.finite_bpb and vb.finite_bpb,
        "step0_clean": va.step0_clean and vb.step0_clean,
        "param_cap": va.param_cap and vb.param_cap,
        "challenge_authored": va.challenge_authored and vb.challenge_authored,
        "force_instrument": va.force_instrument and vb.force_instrument,
        "multi_seed_K": k_min,
        "multi_seed_public": public,
        "multi_seed_provisional": provisional,
        "ok": va.ok and vb.ok and matched_pin,
    }


def _axis_lead(better: bool | None) -> AxisLead:
    if better is True:
        return "a"
    if better is False:
        return "b"
    return "tie"


def _long_ctx_is_better(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    eps_long_ctx: float,
) -> bool | None:
    """Higher long_ctx_score is better. None when either side missing / not filled."""
    sa = a.long_ctx_score
    sb = b.long_ctx_score
    if sa is None or sb is None:
        return None
    if not (math.isfinite(sa) and math.isfinite(sb)):
        return None
    if sa > sb + eps_long_ctx:
        return True
    if sb > sa + eps_long_ctx:
        return False
    return None  # near-tie


def detect_polar_conflict(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    eps_heldout: float = OFFICIAL_EPS_HELDOUT,
    eps_long_ctx: float = OFFICIAL_EPS_LONG_CTX,
    long_ctx_floor: float = OFFICIAL_LONG_CTX_FLOOR,
) -> PolarConflictResult:
    """Detect short-gen vs long-ctx polar conflict (VAL-SCORE-003 / docs §14.5).

    Polar does **not** fire when long-ctx is disabled or both long-ctx scores are
    null/not-run — v1 heldout-primary rank is then preserved (VAL-SCORE-002).
    """
    short_better = a.primary_is_better(b, eps=eps_heldout)
    short_lead = _axis_lead(short_better)

    suite_enabled = bool(a.long_ctx_enabled or b.long_ctx_enabled)
    a_score = a.long_ctx_score
    b_score = b.long_ctx_score
    filled = (
        suite_enabled
        and a_score is not None
        and b_score is not None
        and math.isfinite(a_score)
        and math.isfinite(b_score)
    )
    if not filled:
        return PolarConflictResult(
            tie_polar=False,
            crown_allowed=True,
            short_gen_lead=short_lead,
            long_ctx_lead="missing",
            reason=None,
            eps_heldout=eps_heldout,
            eps_long_ctx=eps_long_ctx,
            long_ctx_enabled_and_filled=False,
        )

    floor_fail_a = float(a_score) < long_ctx_floor
    floor_fail_b = float(b_score) < long_ctx_floor
    # Prefer explicit per-record floor flag when set.
    if a.long_ctx_floor_pass is False:
        floor_fail_a = True
    if a.long_ctx_floor_pass is True:
        floor_fail_a = False
    if b.long_ctx_floor_pass is False:
        floor_fail_b = True
    if b.long_ctx_floor_pass is True:
        floor_fail_b = False

    long_better = _long_ctx_is_better(a, b, eps_long_ctx=eps_long_ctx)
    long_lead = _axis_lead(long_better)

    # Floor form: one side fails long_ctx floor, the other passes — long-ctx competence
    # disagreement. If short-gen still favors the floor-failing side, polar fires.
    asymmetric_floor = floor_fail_a != floor_fail_b
    if asymmetric_floor:
        long_ctx_competent: AxisLead = "b" if floor_fail_a else "a"
        if short_lead in ("a", "b") and short_lead != long_ctx_competent:
            return PolarConflictResult(
                tie_polar=True,
                crown_allowed=False,
                short_gen_lead=short_lead,
                long_ctx_lead=long_ctx_competent,
                reason=(
                    "long_ctx_floor_veto_asymmetric:"
                    f"short_gen={short_lead},long_ctx_competent={long_ctx_competent}"
                ),
                eps_heldout=eps_heldout,
                eps_long_ctx=eps_long_ctx,
                long_ctx_enabled_and_filled=True,
                floor_veto_a=floor_fail_a,
                floor_veto_b=floor_fail_b,
            )
        # Floor disagreement but short-gen does not reverse → still polar-safe
        # only if short and long careful; treat floor-asymmetric + short missing as no crown.
        if short_lead == "tie" and floor_fail_a != floor_fail_b:
            return PolarConflictResult(
                tie_polar=True,
                crown_allowed=False,
                short_gen_lead=short_lead,
                long_ctx_lead=long_ctx_competent,
                reason="long_ctx_floor_veto_asymmetric_short_near_tie",
                eps_heldout=eps_heldout,
                eps_long_ctx=eps_long_ctx,
                long_ctx_enabled_and_filled=True,
                floor_veto_a=floor_fail_a,
                floor_veto_b=floor_fail_b,
            )

    # Pure axis disagreement: A better short, B better long (beyond both ε).
    if short_lead in ("a", "b") and long_lead in ("a", "b") and short_lead != long_lead:
        return PolarConflictResult(
            tie_polar=True,
            crown_allowed=False,
            short_gen_lead=short_lead,
            long_ctx_lead=long_lead,
            reason=f"axis_disagree:short_gen={short_lead},long_ctx={long_lead}",
            eps_heldout=eps_heldout,
            eps_long_ctx=eps_long_ctx,
            long_ctx_enabled_and_filled=True,
            floor_veto_a=floor_fail_a,
            floor_veto_b=floor_fail_b,
        )

    return PolarConflictResult(
        tie_polar=False,
        crown_allowed=True,
        short_gen_lead=short_lead,
        long_ctx_lead=long_lead,
        reason=None,
        eps_heldout=eps_heldout,
        eps_long_ctx=eps_long_ctx,
        long_ctx_enabled_and_filled=True,
        floor_veto_a=floor_fail_a,
        floor_veto_b=floor_fail_b,
    )


def compare_official_scorecard(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    eps_heldout: float = OFFICIAL_EPS_HELDOUT,
    eps_bpb: float = OFFICIAL_EPS_BPB,
    eps_long_ctx: float = OFFICIAL_EPS_LONG_CTX,
    long_ctx_floor: float = OFFICIAL_LONG_CTX_FLOOR,
) -> CompareResult:
    """A-vs-B compare with multimetric.v1.1 polar overlay (VAL-SCORE-002/003).

    1. Run pure v1 :func:`compare_official` (heldout-primary then bpb secondary).
    2. If short-gen vs long-ctx polar-conflict, override to TIE_POLAR /
       ``crown_allowed=false`` while keeping the scorecard vector for callers.
    3. When long-ctx is disabled / not filled, return the v1 result unchanged.
    """
    base = compare_official(a, b, eps_heldout=eps_heldout, eps_bpb=eps_bpb)
    polar = detect_polar_conflict(
        a,
        b,
        eps_heldout=eps_heldout,
        eps_long_ctx=eps_long_ctx,
        long_ctx_floor=long_ctx_floor,
    )
    if polar.tie_polar:
        return CompareResult(
            winner="tie",
            reason="tie_polar",
            rule="multimetric.v1.1_tie_polar",
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            detail=polar.reason or "TIE_POLAR: short-gen vs long-ctx disagree",
            tie_polar=True,
            crown_allowed=False,
            eps_long_ctx=eps_long_ctx,
            scorecard_id=SCORECARD_ID,
        )
    # No polar conflict: preserve v1 winner / reason; annotate crown + scorecard id.
    return CompareResult(
        winner=base.winner,
        reason=base.reason,
        rule=base.rule,
        eps_heldout=base.eps_heldout,
        eps_bpb=base.eps_bpb,
        detail=base.detail,
        tie_polar=False,
        crown_allowed=True,
        eps_long_ctx=eps_long_ctx,
        scorecard_id=SCORECARD_ID,
    )


def _side_scorecard_vector(record: OfficialScoreRecord) -> dict[str, Any]:
    """Publish one side's multi-metric vector (honest nulls when suite not-run)."""
    return {
        "label": record.label,
        "short_gen": {
            "primary_form": record.primary_form,
            "heldout_delta": record.heldout_delta,
            "val_bpb_trained": record.val_bpb_trained,
            "heldout_std": record.heldout_std,
        },
        "secondary_bpb": record.bpb,
        "bpb_std": record.bpb_std,
        "long_ctx": {
            "enabled": record.long_ctx_enabled,
            "suite_mean": record.long_ctx_score,
            "needle": record.long_ctx_needle,
            "mqar": record.long_ctx_mqar,
            "induction_copy": record.long_ctx_induction_copy,
            "lag_nll": record.lag_nll,
            "floor_pass": record.long_ctx_floor_pass,
        },
        "sample_efficiency": {
            "auc": record.sample_eff_auc,
            "marks": list(record.sample_eff_marks) if record.sample_eff_marks is not None else None,
        },
        "memorization": {
            "memo_gap": record.train_heldout_gap,
            "memorization_flag": record.memorization_flag,
            "overfit_rate": record.overfit_rate,
        },
        "efficiency": {
            "params": record.params,
            "peak_vram_gib": record.peak_vram_gib,
            "tokens_per_s": record.tokens_per_s,
            "wall_clock_seconds": record.wall_clock_seconds,
        },
        "stability": {
            "nan_inf_events": record.nan_inf_events,
            "grad_spike_rate": record.grad_spike_rate,
            "instability_flag": record.instability_flag,
            "step0_anomaly": record.step0_anomaly,
        },
        "multi_seed": {
            "K": record.seed_count,
            "public": record.is_public_multi_seed,
            "provisional": record.multi_seed_provisional,
        },
    }


def build_scorecard_annex(
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    compare: CompareResult | None = None,
    matched_pin: bool = True,
    eps_heldout: float = OFFICIAL_EPS_HELDOUT,
    eps_bpb: float = OFFICIAL_EPS_BPB,
    eps_long_ctx: float = OFFICIAL_EPS_LONG_CTX,
    long_ctx_floor: float = OFFICIAL_LONG_CTX_FLOOR,
) -> dict[str, Any]:
    """Build the additive multimetric.v1.1 scorecard annex block (VAL-SCORE-010).

    Prefer attaching this under ``scorecard`` on ``prism_compare_report.v1`` rather
    than rewriting emission leaderboard fields.
    """
    polar = detect_polar_conflict(
        a,
        b,
        eps_heldout=eps_heldout,
        eps_long_ctx=eps_long_ctx,
        long_ctx_floor=long_ctx_floor,
    )
    if compare is None:
        compare = compare_official_scorecard(
            a,
            b,
            eps_heldout=eps_heldout,
            eps_bpb=eps_bpb,
            eps_long_ctx=eps_long_ctx,
            long_ctx_floor=long_ctx_floor,
        )
    pair_v = evaluate_pair_validity(a, b, matched_pin=matched_pin)
    k_min = int(pair_v["multi_seed_K"])
    long_enabled = bool(a.long_ctx_enabled or b.long_ctx_enabled)
    return {
        "scorecard_id": SCORECARD_ID,
        "scorecard_schema": SCORECARD_SCHEMA,
        "tiers": list(SCORECARD_TIERS),
        "multi_seed": {
            "K": k_min,
            "K_a": a.seed_count,
            "K_b": b.seed_count,
            "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
            "public": bool(pair_v["multi_seed_public"]),
            "provisional": bool(pair_v["multi_seed_provisional"]),
        },
        "validity": {
            "stop_token_budget": pair_v["stop_token_budget"],
            "finite_bpb": pair_v["finite_bpb"],
            "step0_clean": pair_v["step0_clean"],
            "param_cap": pair_v["param_cap"],
            "matched_pin": pair_v["matched_pin"],
            "challenge_authored": pair_v["challenge_authored"],
            "force_instrument": pair_v["force_instrument"],
            "multi_seed_K": k_min,
            "ok": pair_v["ok"],
            "sides": {"a": pair_v["a"], "b": pair_v["b"]},
        },
        "short_gen": {
            "heldout_delta_a": a.heldout_delta,
            "heldout_delta_b": b.heldout_delta,
            "val_bpb_trained_a": a.val_bpb_trained,
            "val_bpb_trained_b": b.val_bpb_trained,
            "lead": polar.short_gen_lead,
            "eps_heldout": eps_heldout,
        },
        "long_ctx": {
            "enabled": long_enabled,
            "needle": {"a": a.long_ctx_needle, "b": b.long_ctx_needle},
            "mqar": {"a": a.long_ctx_mqar, "b": b.long_ctx_mqar},
            "induction_copy": {
                "a": a.long_ctx_induction_copy,
                "b": b.long_ctx_induction_copy,
            },
            "lag_nll": {"a": a.lag_nll, "b": b.lag_nll},
            "suite_mean": {"a": a.long_ctx_score, "b": b.long_ctx_score},
            "floor": long_ctx_floor,
            "floor_pass": {
                "a": a.long_ctx_floor_pass,
                "b": b.long_ctx_floor_pass,
            },
            "floors_relative_to_chance": True,
            "floors": {
                "absolute_suite_mean_floor": long_ctx_floor,
                "relative_floor": 0.05,
                "chance_baselines": {
                    "needle": 0.25,
                    "mqar": 1.0 / 16.0,
                    "induction_copy": 0.05,
                },
                "relative_floor_tasks": ["needle", "mqar"],
                "note": (
                    "Seed-scale long-ctx floors: absolute suite mean ≥ "
                    f"{long_ctx_floor}; relative_to_chance ≥ 0.05 on needle and mqar "
                    "when suite enabled."
                ),
            },
            "lead": polar.long_ctx_lead,
            "eps_long_ctx": eps_long_ctx,
        },
        "sample_efficiency": {
            "a": {
                "auc": a.sample_eff_auc,
                "marks": list(a.sample_eff_marks) if a.sample_eff_marks else None,
            },
            "b": {
                "auc": b.sample_eff_auc,
                "marks": list(b.sample_eff_marks) if b.sample_eff_marks else None,
            },
        },
        "memorization": {
            "memo_gap_a": a.train_heldout_gap,
            "memo_gap_b": b.train_heldout_gap,
            "memorization_flag_a": a.memorization_flag,
            "memorization_flag_b": b.memorization_flag,
            "overfit_rate_a": a.overfit_rate,
            "overfit_rate_b": b.overfit_rate,
            "threshold_bpb": OFFICIAL_MEMORIZATION_GAP_THRESHOLD_BPB,
        },
        "efficiency": {
            "params": {"a": a.params, "b": b.params},
            "peak_vram_gib": {"a": a.peak_vram_gib, "b": b.peak_vram_gib},
            "tokens_per_s": {"a": a.tokens_per_s, "b": b.tokens_per_s},
            "wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
            "sole_rank_forbidden": True,
            "flops_diagnostic_only": True,
            "overrides_scientific_axes": False,
            "overrides_polar_rule": False,
        },
        "stability": {
            "nan_inf_events": {"a": a.nan_inf_events, "b": b.nan_inf_events},
            "grad_spike_rate": {"a": a.grad_spike_rate, "b": b.grad_spike_rate},
            "instability_flag": {
                "a": a.instability_flag,
                "b": b.instability_flag,
            },
            "step0_anomaly": {"a": a.step0_anomaly, "b": b.step0_anomaly},
            "bpb_std": {"a": a.bpb_std, "b": b.bpb_std},
            "heldout_std": {"a": a.heldout_std, "b": b.heldout_std},
        },
        "polar": polar.as_dict(),
        "vector": {"a": _side_scorecard_vector(a), "b": _side_scorecard_vector(b)},
        "ranking_overlay": {
            "winner": compare.winner,
            "reason": compare.reason,
            "rule": compare.rule,
            "tie_polar": compare.tie_polar,
            "crown_allowed": compare.crown_allowed,
            "default_v1_preserved_when_no_polar_conflict": not compare.tie_polar,
            "authoritative_claim": ("TIE_POLAR" if compare.tie_polar else compare.reason),
        },
        "honesty_note": SCORECARD_PROVISIONAL_HONESTY_NOTE,
    }


def attach_scorecard_to_report(
    report: dict[str, Any],
    a: OfficialScoreRecord,
    b: OfficialScoreRecord,
    *,
    compare: CompareResult | None = None,
    matched_pin: bool = True,
) -> dict[str, Any]:
    """Attach scorecard annex onto a ``prism_compare_report.v1`` dict (additive).

    Updates ranking when polar conflict forces TIE_POLAR so operators see both the
    scorecard polar block and a consistent ranking surface.
    """
    annex = build_scorecard_annex(a, b, compare=compare, matched_pin=matched_pin)
    overlay = annex["ranking_overlay"]
    ranking = dict(report.get("ranking") or {})
    ranking["winner"] = overlay["winner"]
    ranking["reason"] = overlay["reason"]
    ranking["rule"] = overlay["rule"]
    ranking["tie_polar"] = overlay["tie_polar"]
    ranking["crown_allowed"] = overlay["crown_allowed"]
    ranking["default_v1_preserved_when_no_polar_conflict"] = overlay[
        "default_v1_preserved_when_no_polar_conflict"
    ]
    ranking["authoritative_claim"] = overlay["authoritative_claim"]
    if overlay["tie_polar"]:
        ranking["outcome_label"] = {
            **dict(ranking.get("outcome_label") or {}),
            "winner_side": "tie",
            "winner_label": "TIE_POLAR",
            "crown_allowed": False,
        }
    out = {
        **report,
        "scorecard_id": SCORECARD_ID,
        "scorecard": annex,
        "ranking": ranking,
        "honesty_note": SCORECARD_PROVISIONAL_HONESTY_NOTE,
    }
    return out
