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

# Near-tie bands (docs §5.5). Secondary reuses the held-out epsilon scale.
OFFICIAL_EPS_HELDOUT = HELDOUT_DELTA_BPB_EPSILON  # 5e-3
OFFICIAL_EPS_BPB = HELDOUT_DELTA_BPB_EPSILON  # 5e-3

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

CompareWinner = Literal["a", "b", "tie"]
CompareReason = Literal[
    "invalid",
    "step0_anomaly",
    "primary_heldout",
    "secondary_bpb",
    "anti_overfit",
    "multi_seed_residual",
    "tie",
]


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

    @property
    def primary_value(self) -> float | None:
        if self.primary_form == PRIMARY_FORM_HELDOUT_DELTA:
            return self.heldout_delta
        return self.val_bpb_trained

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
    """Deterministic outcome of :func:`compare_official`."""

    winner: CompareWinner
    reason: CompareReason
    rule: str = "heldout_primary_then_bpb_secondary"
    eps_heldout: float = OFFICIAL_EPS_HELDOUT
    eps_bpb: float = OFFICIAL_EPS_BPB
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "rule": self.rule,
            "eps_heldout": self.eps_heldout,
            "eps_bpb": self.eps_bpb,
            "detail": self.detail,
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
) -> OfficialScoreRecord:
    """Project a challenge-owned :class:`PrequentialBpbScore` into an official record.

    Miner-reported numbers may be attached as diagnostics only.
    """
    valid = (
        math.isfinite(score.bpb)
        and score.bpb > 0.0
        and not score.anomaly
        and score.anti_cheat_multiplier > 0.0
    )
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
    marked invalid so compare fails closed.
    """
    clean = [r for r in records if r.valid and not r.step0_anomaly]
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
        )
    bpb_vals = [float(r.bpb) for r in clean]
    mean_bpb = sum(bpb_vals) / len(bpb_vals)
    variance = sum((v - mean_bpb) ** 2 for v in bpb_vals) / len(bpb_vals)
    bpb_std = math.sqrt(variance)
    overfit_rate = sum(1.0 for r in clean if r.memorization_flag) / len(clean)

    if primary_form == PRIMARY_FORM_HELDOUT_DELTA:
        deltas = [float(r.heldout_delta) for r in clean if r.heldout_delta is not None]
        mean_delta = (sum(deltas) / len(deltas)) if deltas else None
        mean_val = None
    else:
        vals = [float(r.val_bpb_trained) for r in clean if r.val_bpb_trained is not None]
        mean_val = (sum(vals) / len(vals)) if vals else None
        mean_delta = None

    # Gap mean for residual diagnostics only.
    gaps = [float(r.train_heldout_gap) for r in clean if r.train_heldout_gap is not None]
    mean_gap = (sum(gaps) / len(gaps)) if gaps else None
    memo = overfit_rate > 0.5 or (
        mean_gap is not None and mean_gap > OFFICIAL_MEMORIZATION_GAP_THRESHOLD_BPB
    )
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
            "min_public_seeds": OFFICIAL_MIN_PUBLIC_SEEDS,
            "memorization_gap_threshold_bpb": OFFICIAL_MEMORIZATION_GAP_THRESHOLD_BPB,
            "memorization_penalty_factor": OFFICIAL_MEMORIZATION_PENALTY_FACTOR,
            "wall_clock_never_ranks": OFFICIAL_WALL_CLOCK_NEVER_RANKS,
            "protocol_schema": PROTOCOL_SCHEMA,
        },
    )
