from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# --- Prequential bits-per-byte primary scoring (architecture.md section 5) -------------------
# The authoritative score is the prequential / online compression metric in bits-per-byte: the
# AREA UNDER the from-scratch online loss curve (integrated over the whole single-pass run),
# normalized by the number of raw UTF-8 BYTES covered (tokenizer-agnostic by construction). It is
# always recomputed from the CHALLENGE-OWNED prism_run_manifest.v2 capture; miner-reported numbers
# are ignored. The legacy raw-loss ``standardized_lm_quality`` term is retired from the score.
NATS_TO_BITS = 1.0 / math.log(2.0)
# A finite/positive sanity band for bits-per-byte; anything outside it is treated as a degenerate
# (non-scorable) run rather than silently ranked.
BPB_SANE_MAX = 64.0

# --- Held-out delta tie-breaker + anti-memorization gap (architecture.md sections 5, 6) ----------
# The held-out delta-over-random-init (``bpb(random-init twin on val) - bpb(trained on val)``) is a
# TIE-BREAKER: a larger improvement ranks better, but it must not override the primary bpb axis, so
# it contributes only a tiny term folded into ``final_score`` (a lexicographic refinement is the
# job of the leaderboard-determinism feature). The train-vs-held-out gap flags memorization: an
# excessive gap penalizes the score so a memorizer ranks below an equivalent non-memorizing run.
HELDOUT_DELTA_TIE_BREAK_WEIGHT = 1e-3
MEMORIZATION_GAP_THRESHOLD_BPB = 1.0
MEMORIZATION_PENALTY_FACTOR = 0.5

class ScoreValidationError(ValueError):
    def __init__(self, reasons: list[str] | tuple[str, ...]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class PrequentialBpbScore:
    """Challenge-computed prequential bits-per-byte primary score (architecture.md section 5).

    ``bpb`` is the prequential code-length integrated over the WHOLE single-pass online-loss curve
    divided by the raw UTF-8 BYTES covered (tokenizer-agnostic). ``final_score`` is a documented
    monotone transform where a SMALLER bpb yields a BETTER (larger) final_score, so the existing
    leaderboard ``ORDER BY final_score DESC`` ranks better learners higher. A step-0 / smuggled-
    weights anomaly drives the anti-cheat multiplier to zero so an anomalously-low bpb is flagged
    rather than rewarded.
    """

    bpb: float
    final_score: float
    covered_bytes: int
    sum_neg_log2_likelihood_bits: float
    cumulative_codelength_bits: float
    tokens_consumed: int
    online_loss_samples: int
    step0_loss: float | None
    anti_cheat_multiplier: float
    anomaly: bool
    flags: tuple[str, ...]
    # Held-out delta tie-breaker + anti-memorization gap (architecture.md sections 5, 6). These are
    # ``None`` when no secret val split was scored for the run (held-out simply skipped).
    heldout_delta: float | None = None
    val_bpb_trained: float | None = None
    val_bpb_random_init: float | None = None
    train_heldout_gap: float | None = None
    memorization_flag: bool = False
    memorization_penalty: float = 1.0

    def metrics_payload(self) -> dict[str, Any]:
        """Flat metrics for the ``scores`` row (challenge-computed; no raw-loss term)."""
        payload: dict[str, Any] = {
            "prequential_bpb": self.bpb,
            "bits_per_byte": self.bpb,
            "final_score": self.final_score,
            "total_bytes_covered": float(self.covered_bytes),
            "covered_bytes": float(self.covered_bytes),
            "sum_neg_log2_likelihood_bits": self.sum_neg_log2_likelihood_bits,
            "cumulative_codelength_bits": self.cumulative_codelength_bits,
            "tokens_consumed": float(self.tokens_consumed),
            "online_loss_samples": float(self.online_loss_samples),
            "anti_cheat_multiplier": self.anti_cheat_multiplier,
            "step0_anomaly": float(self.anomaly),
            "memorization_flag": float(self.memorization_flag),
            "memorization_penalty": self.memorization_penalty,
        }
        if self.heldout_delta is not None:
            payload["heldout_delta"] = self.heldout_delta
            payload["held_out_delta"] = self.heldout_delta
        if self.val_bpb_trained is not None:
            payload["val_bpb_trained"] = self.val_bpb_trained
        if self.val_bpb_random_init is not None:
            payload["val_bpb_random_init"] = self.val_bpb_random_init
        if self.train_heldout_gap is not None:
            payload["train_heldout_gap"] = self.train_heldout_gap
            payload["train_val_gap"] = self.train_heldout_gap
        return payload

    def manifest_score_block(self) -> dict[str, Any]:
        """Challenge-authored ``score`` block merged into prism_run_manifest.v2.json."""
        block: dict[str, Any] = {
            "schema": "prism_score.v2",
            "primary_metric": "prequential_bpb",
            "prequential_bpb": self.bpb,
            "bits_per_byte": self.bpb,
            "final_score": self.final_score,
            "lower_is_better": True,
            "covered_bytes": self.covered_bytes,
            "total_bytes_covered": self.covered_bytes,
            "sum_neg_log2_likelihood_bits": self.sum_neg_log2_likelihood_bits,
            "cumulative_codelength_bits": self.cumulative_codelength_bits,
            "tokens_consumed": self.tokens_consumed,
            "compute_normalization": "tokens_bytes",
            "wall_clock_term": False,
            "anti_cheat_multiplier": self.anti_cheat_multiplier,
            "anomaly": self.anomaly,
            "flags": list(self.flags),
            "tie_breaker": "heldout_delta",
            "memorization_flag": self.memorization_flag,
            "memorization_penalty": self.memorization_penalty,
            "miner_reported_ignored": True,
        }
        if self.heldout_delta is not None:
            block["heldout_delta"] = self.heldout_delta
            block["held_out_delta"] = self.heldout_delta
        if self.val_bpb_trained is not None:
            block["val_bpb_trained"] = self.val_bpb_trained
        if self.val_bpb_random_init is not None:
            block["val_bpb_random_init"] = self.val_bpb_random_init
        if self.train_heldout_gap is not None:
            block["train_heldout_gap"] = self.train_heldout_gap
        return block


def bpb_to_final_score(bpb: float) -> float:
    """Monotone-decreasing transform of bits-per-byte: lower bpb -> higher (better) final_score."""
    return 1.0 / (1.0 + max(0.0, float(bpb)))


# --- Deterministic leaderboard ordering + final tie-break (architecture.md section 5) ------------
# ``final_score`` already folds the primary prequential bpb and the held-out-delta tie-breaker into
# one monotone number (lower bpb / larger delta => larger final_score, so ORDER BY final_score DESC
# ranks better learners first). When two submissions are near-equal on BOTH axes their final_score
# is (near-)equal; the FINAL deterministic tie-break is EARLIEST-COMMIT-WINS (then submission id) so
# the leaderboard order is a TOTAL, reproducible order. The tie epsilon stays far below
# ``HELDOUT_DELTA_TIE_BREAK_WEIGHT`` so a genuine held-out-delta difference still orders ahead of
# the commit-time tie-break (the delta tie-break is never collapsed).
LEADERBOARD_TIE_EPSILON = 1e-9


@dataclass(frozen=True)
class LeaderboardRow:
    """A scored, completed submission competing for a leaderboard rank."""

    submission_id: str
    hotkey: str
    final_score: float
    accepted_at: str


def leaderboard_rank_key(row: LeaderboardRow) -> tuple[int, str, str]:
    """Total, deterministic leaderboard sort key (ascending => better rank first).

    ``final_score`` is quantized onto an epsilon grid so near-equal scores share a bucket; the
    higher bucket ranks first (``-bucket``), and a same-bucket tie is resolved by earliest commit
    (``accepted_at`` ascending) then ``submission_id`` ascending.
    """
    score = float(row.final_score)
    bucket = math.floor(score / LEADERBOARD_TIE_EPSILON + 0.5) if math.isfinite(score) else 0
    return (-bucket, row.accepted_at, row.submission_id)


def rank_leaderboard(rows: Iterable[LeaderboardRow]) -> list[LeaderboardRow]:
    """Order leaderboard rows by bpb/learning with the deterministic earliest-commit tie-break."""
    return sorted(rows, key=leaderboard_rank_key)


def score_prequential_bpb(
    manifest: Mapping[str, Any], *, sane_max: float = BPB_SANE_MAX
) -> PrequentialBpbScore:
    """Compute the prequential bits-per-byte score from the challenge-owned v2 manifest.

    ``bpb = (sum over consumed tokens of -log2 p(token)) / total_bytes_covered`` where the
    numerator is the token-weighted online (predict-then-train) negative log-likelihood the
    challenge captured itself. Raises ``ScoreValidationError`` for a degenerate (zero-coverage,
    non-finite, or out-of-band) run so it never collapses into a fabricated/0-that-ranks score.
    """
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ScoreValidationError(["v2 manifest is missing a metrics block"])
    covered_bytes = _manifest_covered_bytes(manifest, metrics)
    if covered_bytes <= 0:
        raise ScoreValidationError(["prequential scoring requires covered_bytes > 0"])
    sum_nll_nats = float(metrics.get("sum_neg_log_likelihood_nats", 0.0))
    online_loss = metrics.get("online_loss")
    online_samples = len(online_loss) if isinstance(online_loss, list) else 0
    if online_samples == 0:
        raise ScoreValidationError(["prequential scoring requires a captured online-loss stream"])
    cumulative_codelength_bits = sum_nll_nats * NATS_TO_BITS
    bpb = cumulative_codelength_bits / covered_bytes
    flags: list[str] = []
    if not math.isfinite(bpb):
        raise ScoreValidationError(["prequential bpb is not finite"])
    if bpb <= 0.0:
        raise ScoreValidationError(["prequential bpb must be positive"])
    if bpb > sane_max:
        flags.append("bpb_out_of_band")
    anti_cheat = manifest.get("anti_cheat")
    anti_cheat = anti_cheat if isinstance(anti_cheat, Mapping) else {}
    step0_anomaly = bool(anti_cheat.get("step0_anomaly", False))
    if step0_anomaly:
        flags.append("step0_anomaly")
    if bool(anti_cheat.get("nan_inf_detected", False)):
        flags.append("nan_inf_detected")
    anti_cheat_multiplier = 0.0 if step0_anomaly else 1.0
    heldout = _read_heldout(manifest, metrics, anti_cheat, train_bpb=bpb)
    if heldout.memorization_flag:
        flags.append("memorization_gap")
    # The held-out delta refines ranking only as a TIE-BREAKER (tiny additive term, monotone in the
    # delta) so a strictly lower bpb is never ranked worse purely on the primary axis; an excessive
    # train-vs-held-out gap multiplies in a memorization penalty so a memorizer ranks below an
    # equivalent non-memorizing run.
    base = bpb_to_final_score(bpb)
    tie_break = (
        HELDOUT_DELTA_TIE_BREAK_WEIGHT * math.tanh(heldout.delta)
        if heldout.delta is not None
        else 0.0
    )
    final_score_value = (base * heldout.penalty + tie_break) * anti_cheat_multiplier
    step0_loss = metrics.get("step0_loss")
    return PrequentialBpbScore(
        bpb=bpb,
        final_score=final_score_value,
        covered_bytes=covered_bytes,
        sum_neg_log2_likelihood_bits=cumulative_codelength_bits,
        cumulative_codelength_bits=cumulative_codelength_bits,
        tokens_consumed=int(metrics.get("predicted_tokens", metrics.get("tokens_seen", 0)) or 0),
        online_loss_samples=online_samples,
        step0_loss=float(step0_loss) if isinstance(step0_loss, int | float) else None,
        anti_cheat_multiplier=anti_cheat_multiplier,
        anomaly=step0_anomaly,
        flags=tuple(flags),
        heldout_delta=heldout.delta,
        val_bpb_trained=heldout.val_bpb_trained,
        val_bpb_random_init=heldout.val_bpb_random_init,
        train_heldout_gap=heldout.gap,
        memorization_flag=heldout.memorization_flag,
        memorization_penalty=heldout.penalty,
    )


@dataclass(frozen=True)
class _HeldoutView:
    delta: float | None
    val_bpb_trained: float | None
    val_bpb_random_init: float | None
    gap: float | None
    memorization_flag: bool
    penalty: float


def _read_heldout(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    anti_cheat: Mapping[str, Any],
    *,
    train_bpb: float,
) -> _HeldoutView:
    """Read the host-computed held-out delta + anti-memorization gap from the v2 manifest.

    The held-out delta + gap are populated host-side (``evaluator/heldout.py``) into the metrics /
    score blocks. When absent (no secret val split scored) the run is graded on prequential bpb
    alone with no tie-break and no penalty.
    """
    score_block = manifest.get("score")
    score_block = score_block if isinstance(score_block, Mapping) else {}
    delta = _coerce_float(_first_present(metrics, score_block, ("heldout_delta", "held_out_delta")))
    val_trained = _coerce_float(_first_present(metrics, score_block, ("val_bpb_trained",)))
    val_random = _coerce_float(_first_present(metrics, score_block, ("val_bpb_random_init",)))
    gap = _coerce_float(
        _first_present(metrics, score_block, ("train_heldout_gap", "train_val_gap"))
    )
    if gap is None and val_trained is not None and math.isfinite(train_bpb):
        gap = val_trained - train_bpb
    explicit_flag = bool(
        metrics.get("memorization_flag")
        or score_block.get("memorization_flag")
        or anti_cheat.get("memorization_flag")
    )
    memorization_flag = explicit_flag or (gap is not None and gap > MEMORIZATION_GAP_THRESHOLD_BPB)
    penalty = MEMORIZATION_PENALTY_FACTOR if memorization_flag else 1.0
    return _HeldoutView(
        delta=delta,
        val_bpb_trained=val_trained,
        val_bpb_random_init=val_random,
        gap=gap,
        memorization_flag=memorization_flag,
        penalty=penalty,
    )


def _first_present(
    primary: Mapping[str, Any], secondary: Mapping[str, Any], keys: tuple[str, ...]
) -> Any:
    for source in (primary, secondary):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def _coerce_float(value: Any) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _manifest_covered_bytes(manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> int:
    for source in (metrics, manifest.get("data")):
        if isinstance(source, Mapping):
            value = source.get("covered_bytes")
            if isinstance(value, int | float) and not isinstance(value, bool):
                return int(value)
    return 0
