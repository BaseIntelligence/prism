"""Dual-family Official Comparison CPU/fixture harness (no NVIDIA required).

Packages the registered Transformer + Mamba seed families as unknown-style two-script
zips under one ProtocolPin, builds challenge-owned synthetic score manifests (Prism
recomputes secondary bpb), and emits a ``prism_compare_report.v1`` via
:func:`compare_official`.

This is the lab/offline ArchCompare surface for VAL-COMP-009. Long multi-step GPU
pair trains remain **DEFERRED** when the host lacks NVIDIA (see
:func:`gpu_verification_status`). Production leaderboard scoring is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from prism_challenge.seed_packaging import (
    REQUIRED_ENTRY_SCRIPTS,
    SEED_FAMILIES,
    PackedSeed,
    get_family,
    package_seed_zip,
)

from .official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA,
    CompareResult,
    OfficialScoreRecord,
    ProtocolPin,
    aggregate_official_records,
    compare_official,
    official_record_from_manifest,
    protocol_budget_constants,
)

REPORT_SCHEMA = "prism_compare_report.v1"
CompareMode = Literal["ArchCompare", "TrainCompare", "SystemCompare"]

# Registered dual-family pair for the fixture harness (unknown architecture style).
SIDE_A_FAMILY_ID = "transformer-tiny-1m"
SIDE_B_FAMILY_ID = "mamba-tiny-1m"

DeviceClass = Literal["cpu", "cuda", "fixture"]


@dataclass(frozen=True)
class SynthSeedMetrics:
    """One seed's challenge-owned capture shell used to recompute official scores.

    Values are synthetic (unit fixtures). They must look like a Prism host capture,
    never like a miner self-report trust root.
    """

    seed: int
    bpb: float
    heldout_delta: float
    covered_bytes: int = 65_536
    train_heldout_gap: float = 0.15
    memorization_flag: bool | None = None
    step0_anomaly: bool = False
    wall_clock_seconds: float = 120.0
    val_bpb_trained: float | None = None
    val_bpb_random_init: float | None = None


@dataclass(frozen=True)
class FamilySynthProfile:
    """Per-family multi-seed synthetic profile under the matched protocol pin."""

    family_id: str
    label: str
    architecture_family: str
    seeds: tuple[SynthSeedMetrics, ...]


# Deterministic dual-family fixtures: Transformer generalizes better on primary
# held-out axis; secondary bpb is intentionally weaker than Mamba so production
# leaderboard (bpb-primary) would disagree — Official Comparison invert is visible.
DEFAULT_TRANSFORMER_PROFILE = FamilySynthProfile(
    family_id=SIDE_A_FAMILY_ID,
    label="transformer-tiny-1m",
    architecture_family="transformer",
    seeds=(
        SynthSeedMetrics(seed=1337, bpb=1.80, heldout_delta=0.95, train_heldout_gap=0.12),
        SynthSeedMetrics(seed=2027, bpb=1.85, heldout_delta=0.90, train_heldout_gap=0.14),
        SynthSeedMetrics(seed=4242, bpb=1.75, heldout_delta=1.00, train_heldout_gap=0.10),
    ),
)
DEFAULT_MAMBA_PROFILE = FamilySynthProfile(
    family_id=SIDE_B_FAMILY_ID,
    label="mamba-tiny-1m",
    architecture_family="mamba",
    seeds=(
        SynthSeedMetrics(seed=1337, bpb=1.10, heldout_delta=0.25, train_heldout_gap=0.18),
        SynthSeedMetrics(seed=2027, bpb=1.15, heldout_delta=0.20, train_heldout_gap=0.20),
        SynthSeedMetrics(seed=4242, bpb=1.05, heldout_delta=0.30, train_heldout_gap=0.16),
    ),
)


def default_protocol_pin(*, device_class: DeviceClass = "fixture") -> ProtocolPin:
    """Matched ProtocolPin shared by both harness sides (fairness pins)."""
    del device_class  # pin is device-agnostic; device class is report metadata only
    return ProtocolPin(
        protocol_id=PROTOCOL_ID,
        token_budget=protocol_budget_constants()["token_budget"],
        seeds=OFFICIAL_DEFAULT_SEEDS,
        param_cap=int(protocol_budget_constants()["param_cap"]),
        seq_len=int(protocol_budget_constants()["seq_len"]),
        batch_size=int(protocol_budget_constants()["batch_size"]),
        tokenizer=str(protocol_budget_constants()["tokenizer"]),
        vocab_size=int(protocol_budget_constants()["vocab_size"]),
        scored_nproc=int(protocol_budget_constants()["scored_nproc"]),
        val_byte_budget=int(protocol_budget_constants()["val_byte_budget"]),
        force_iter_train_batches=True,
        require_trained_state=True,
        primary_form="heldout_delta",
    )


def protocol_pin_hash(pin: ProtocolPin) -> str:
    """Stable pin hash for report / operator evidence."""
    payload = json.dumps(pin.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def gpu_verification_status() -> dict[str, Any]:
    """Probe host GPU readiness; never invent a GPU PASS.

    Returns classification DEFERRED when NVIDIA is absent (this mission host profile).
    """
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_devices = sorted(Path("/dev").glob("nvidia*"))
    has_nvidia_runtime = False
    docker = shutil.which("docker")
    if docker is not None:
        # Best-effort: presence of nvidia runtime listed by docker info is informative only.
        # We do not execute GPU containers here.
        try:
            proc = subprocess.run(
                [docker, "info", "--format", "{{json .Runtimes}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0 and "nvidia" in (proc.stdout or "").lower():
                has_nvidia_runtime = True
        except (OSError, subprocess.TimeoutExpired):
            has_nvidia_runtime = False

    available = bool(nvidia_smi) and bool(nvidia_devices) and has_nvidia_runtime
    if available:
        status = "AVAILABLE"
        reason = "nvidia-smi, /dev/nvidia*, and docker nvidia runtime all present"
    else:
        status = "DEFERRED"
        reasons: list[str] = []
        if not nvidia_smi:
            reasons.append("nvidia-smi not found")
        if not nvidia_devices:
            reasons.append("/dev/nvidia* absent")
        if not has_nvidia_runtime:
            reasons.append("docker nvidia runtime not advertised")
        reason = "; ".join(reasons) if reasons else "GPU path not ready"
    return {
        "status": status,
        "reason": reason,
        "nvidia_smi": nvidia_smi is not None,
        "nvidia_device_count": len(nvidia_devices),
        "docker_nvidia_runtime": has_nvidia_runtime,
        "claim_gpu_pass": False,  # harness never elevates this to PASS
    }


def synth_challenge_manifest(
    metrics: SynthSeedMetrics,
    *,
    pin: ProtocolPin,
    family_id: str,
    device: DeviceClass = "fixture",
) -> dict[str, Any]:
    """Build a challenge-owned v2 manifest shell from synthetic seed metrics.

    Secondary bpb is always re-derived by :func:`score_prequential_bpb` via
    ``official_record_from_manifest`` (miner self-report fields are absent).
    """
    covered = int(metrics.covered_bytes)
    sum_nll_nats = float(metrics.bpb) * covered * math.log(2.0)
    body: dict[str, Any] = {
        "schema_version": "prism_run_manifest.v2",
        "protocol": {
            "protocol_id": pin.protocol_id,
            "protocol_hash": protocol_pin_hash(pin),
            "family_id": family_id,
            "seed": metrics.seed,
            "token_budget": pin.token_budget,
            "tokenizer": pin.tokenizer,
            "param_cap": pin.param_cap,
            "unknown_style": True,
        },
        "data": {
            "covered_bytes": covered,
            "single_pass": True,
            "stopped_reason": "token_budget",
        },
        "metrics": {
            "online_loss": [2.0, 1.5, 1.0],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "covered_bytes": covered,
            "predicted_tokens": max(1, covered // 2),
            "step0_loss": 2.0,
            "consumed_batches": 3,
            "random_init_baseline_nats": math.log(256),
            "nan_inf_batches": 0,
            "heldout_delta": metrics.heldout_delta,
            "held_out_delta": metrics.heldout_delta,
            "train_heldout_gap": metrics.train_heldout_gap,
        },
        "anti_cheat": {
            "step0_anomaly": metrics.step0_anomaly,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
        "compute": {
            "schema": "prism_compute.v1",
            "gpu_count": 0 if device != "cuda" else 1,
            "world_size": pin.scored_nproc,
            "nproc_per_node": pin.scored_nproc,
            "device": "cpu" if device != "cuda" else "cuda",
            "wall_clock_seconds": metrics.wall_clock_seconds,
        },
    }
    metrics_map = body["metrics"]
    assert isinstance(metrics_map, dict)
    if metrics.val_bpb_trained is not None:
        metrics_map["val_bpb_trained"] = metrics.val_bpb_trained
    if metrics.val_bpb_random_init is not None:
        metrics_map["val_bpb_random_init"] = metrics.val_bpb_random_init
    if metrics.memorization_flag is not None:
        metrics_map["memorization_flag"] = metrics.memorization_flag
    return body


def package_unknown_style_pair(
    output_dir: Path | str,
    *,
    side_a_family_id: str = SIDE_A_FAMILY_ID,
    side_b_family_id: str = SIDE_B_FAMILY_ID,
) -> dict[str, PackedSeed]:
    """Package both families as two-script submission zips under the shared outer contract.

    Both packages are treated as *unknown style*: same required entry scripts, same
    fingerprint surface, no family-specific score path. Seed packaging already exists;
    this couples packaging to the official compare pin.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = {
        "a": package_seed_zip(side_a_family_id, out),
        "b": package_seed_zip(side_b_family_id, out),
    }
    # Fairness: required entry names identical across families.
    a_entries = set(pairs["a"].entry_names)
    b_entries = set(pairs["b"].entry_names)
    missing_a = [n for n in REQUIRED_ENTRY_SCRIPTS if n not in a_entries]
    missing_b = [n for n in REQUIRED_ENTRY_SCRIPTS if n not in b_entries]
    if missing_a or missing_b:
        raise RuntimeError(f"unknown-style package missing entries a={missing_a} b={missing_b}")
    return pairs


def records_for_profile(
    profile: FamilySynthProfile,
    *,
    pin: ProtocolPin,
    device: DeviceClass = "fixture",
    miner_reported: Mapping[str, Any] | None = None,
) -> list[OfficialScoreRecord]:
    """Project one family synth profile into official score records (per seed)."""
    records: list[OfficialScoreRecord] = []
    for seed_metrics in profile.seeds:
        if seed_metrics.seed not in pin.seeds:
            # Protocol fairness: only pin seeds may contribute.
            continue
        manifest = synth_challenge_manifest(
            seed_metrics,
            pin=pin,
            family_id=profile.family_id,
            device=device,
        )
        rec = official_record_from_manifest(
            manifest,
            label=f"{profile.label}:seed={seed_metrics.seed}",
            primary_form=pin.primary_form,
            miner_reported=miner_reported,
        )
        records.append(rec)
    return records


def aggregate_side(
    profile: FamilySynthProfile,
    *,
    pin: ProtocolPin,
    device: DeviceClass = "fixture",
    miner_reported: Mapping[str, Any] | None = None,
) -> OfficialScoreRecord:
    """Multi-seed official aggregate for one package side."""
    per_seed = records_for_profile(profile, pin=pin, device=device, miner_reported=miner_reported)
    return aggregate_official_records(
        per_seed,
        label=profile.label,
        primary_form=pin.primary_form,
    )


def _side_bundle_block(packed: PackedSeed, aggregate: OfficialScoreRecord) -> dict[str, Any]:
    family = get_family(packed.family_id)
    return {
        "label": aggregate.label,
        "family_id": packed.family_id,
        "architecture_family": family.architecture_family,
        "unknown_style": True,
        "bundle_hash": packed.content_sha256,
        "zip_path": str(packed.zip_path),
        "entry_names": list(packed.entry_names),
        "size_bytes": packed.size_bytes,
        "mean_heldout_delta": aggregate.heldout_delta,
        "mean_bpb": aggregate.bpb,
        "std_bpb": aggregate.bpb_std,
        "overfit_rate": aggregate.overfit_rate,
        "memorization_flag": aggregate.memorization_flag,
        "step0_anomaly": aggregate.step0_anomaly,
        "valid": aggregate.valid,
        "seed_count": aggregate.seed_count,
        "wall_clock_seconds": aggregate.wall_clock_seconds,
        "miner_reported_bpb": aggregate.miner_reported_bpb,
    }


def build_compare_report(
    *,
    pin: ProtocolPin,
    side_a: OfficialScoreRecord,
    side_b: OfficialScoreRecord,
    packed: Mapping[str, PackedSeed],
    result: CompareResult,
    mode: CompareMode = "ArchCompare",
    device_class: DeviceClass = "fixture",
    gpu: Mapping[str, Any] | None = None,
    validity_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Emit a ``prism_compare_report.v1`` document (docs §10 sketch)."""
    gpu_info = dict(gpu) if gpu is not None else gpu_verification_status()
    validity_ok = side_a.valid and side_b.valid and not validity_reasons
    return {
        "schema": REPORT_SCHEMA,
        "protocol_id": pin.protocol_id,
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_hash": protocol_pin_hash(pin),
        "mode": mode,
        "primary_form": pin.primary_form,
        "device_class": device_class,
        "pin": pin.as_dict(),
        "side_a": _side_bundle_block(packed["a"], side_a),
        "side_b": _side_bundle_block(packed["b"], side_b),
        "seeds": list(pin.seeds),
        "aggregate": {
            "a": {
                "mean_heldout_delta": side_a.heldout_delta,
                "mean_bpb": side_a.bpb,
                "std_bpb": side_a.bpb_std,
                "overfit_rate": side_a.overfit_rate,
                "valid": side_a.valid,
                "seed_count": side_a.seed_count,
            },
            "b": {
                "mean_heldout_delta": side_b.heldout_delta,
                "mean_bpb": side_b.bpb,
                "std_bpb": side_b.bpb_std,
                "overfit_rate": side_b.overfit_rate,
                "valid": side_b.valid,
                "seed_count": side_b.seed_count,
            },
        },
        "ranking": {
            "winner": result.winner,
            "reason": result.reason,
            "rule": result.rule,
            "eps_heldout": result.eps_heldout,
            "eps_bpb": result.eps_bpb,
            "detail": result.detail,
            # Label-facing outcome: clear A vs B for operators/tests.
            "outcome_label": {
                "a": side_a.label,
                "b": side_b.label,
                "winner_side": result.winner,
                "winner_label": (
                    side_a.label
                    if result.winner == "a"
                    else side_b.label
                    if result.winner == "b"
                    else "tie"
                ),
            },
        },
        "validity": {
            "ok": validity_ok,
            "reasons": list(validity_reasons),
            "wall_clock_never_ranks": True,
            "miner_self_report_never_authoritative": True,
            "matched_budget": True,
            "required_entry_scripts": list(REQUIRED_ENTRY_SCRIPTS),
        },
        "gpu_verification": gpu_info,
        "tee_note": (
            "orthogonal; REAL-PROVIDER PASS not claimed; LOCAL-FIXTURE only if elevated "
            "crypto is exercised separately"
        ),
    }


def run_dual_family_official_compare(
    output_dir: Path | str,
    *,
    side_a_profile: FamilySynthProfile = DEFAULT_TRANSFORMER_PROFILE,
    side_b_profile: FamilySynthProfile = DEFAULT_MAMBA_PROFILE,
    pin: ProtocolPin | None = None,
    mode: CompareMode = "ArchCompare",
    device_class: DeviceClass = "fixture",
    package: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    """End-to-end dual-family official compare without NVIDIA.

    1. Package both unknown-style seed zips (optional if only synth metrics desired).
    2. Build multi-seed official records from synthetic challenge-owned metrics.
    3. Aggregate and :func:`compare_official`.
    4. Emit report dict (+ optional JSON under ``output_dir``).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    active_pin = pin if pin is not None else default_protocol_pin(device_class=device_class)
    gpu = gpu_verification_status()

    if package:
        packed = package_unknown_style_pair(
            out / "packages",
            side_a_family_id=side_a_profile.family_id,
            side_b_family_id=side_b_profile.family_id,
        )
    else:
        # Minimal stub packages for pure-metric unit paths that skip zip I/O:
        # still require families to be registered so fairness probes work.
        for fam in (side_a_profile.family_id, side_b_profile.family_id):
            if fam not in SEED_FAMILIES:
                raise KeyError(fam)
        packed = package_unknown_style_pair(
            out / "packages",
            side_a_family_id=side_a_profile.family_id,
            side_b_family_id=side_b_profile.family_id,
        )

    side_a = aggregate_side(side_a_profile, pin=active_pin, device=device_class)
    side_b = aggregate_side(side_b_profile, pin=active_pin, device=device_class)
    result = compare_official(side_a, side_b)
    report = build_compare_report(
        pin=active_pin,
        side_a=side_a,
        side_b=side_b,
        packed=packed,
        result=result,
        mode=mode,
        device_class=device_class,
        gpu=gpu,
    )
    if write_report:
        report_path = out / "prism_compare_report.v1.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {**report, "report_path": str(report_path)}
    return report


def main(argv: list[str] | None = None) -> int:
    """Operator CLI: offline dual-family official compare (CPU/fixture)."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Prism Official Comparison Protocol v1 dual-family CPU/fixture harness "
            f"({SIDE_A_FAMILY_ID} vs {SIDE_B_FAMILY_ID}). No GPU required; "
            "REAL-PROVIDER TEE PASS is never claimed."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/official-compare"),
        help="Directory for seed zips + prism_compare_report.v1.json",
    )
    parser.add_argument(
        "--mode",
        choices=("ArchCompare", "TrainCompare", "SystemCompare"),
        default="ArchCompare",
        help="Official compare mode (default ArchCompare).",
    )
    parser.add_argument(
        "--device-class",
        choices=("cpu", "cuda", "fixture"),
        default="fixture",
        help="Report device class. cuda is metadata only; no GPU train is launched.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full compare report JSON to stdout.",
    )
    args = parser.parse_args(argv)

    report = run_dual_family_official_compare(
        args.output_dir,
        mode=args.mode,  # type: ignore[arg-type]
        device_class=args.device_class,  # type: ignore[arg-type]
    )
    ranking = report["ranking"]
    gpu = report["gpu_verification"]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"protocol: {report['protocol_id']} hash={report['protocol_hash'][:16]}…")
        print(f"mode: {report['mode']} device_class={report['device_class']}")
        print(
            f"side A: {report['side_a']['label']} "
            f"heldout_delta={report['side_a']['mean_heldout_delta']} "
            f"bpb={report['side_a']['mean_bpb']}"
        )
        print(
            f"side B: {report['side_b']['label']} "
            f"heldout_delta={report['side_b']['mean_heldout_delta']} "
            f"bpb={report['side_b']['mean_bpb']}"
        )
        print(
            f"compare outcome: winner={ranking['winner']} "
            f"({ranking['outcome_label']['winner_label']}) "
            f"reason={ranking['reason']}"
        )
        print(
            f"gpu_verification: {gpu['status']} ({gpu['reason']}); "
            f"claim_gpu_pass={gpu['claim_gpu_pass']}"
        )
        print(f"tee_note: {report['tee_note']}")
        if report.get("report_path"):
            print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
