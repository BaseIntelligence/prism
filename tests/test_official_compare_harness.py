"""Dual-family Official Comparison CPU/fixture harness (VAL-COMP-009).

Proves two unknown-style seed packages (Transformer tiny-1m vs pure-torch Mamba)
under protocol fairness produce comparable official records and a clear A-vs-B
``compare_official`` outcome without NVIDIA. Synthetic challenge-owned metrics only;
no multi-hour train; REAL-PROVIDER TEE PASS never claimed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from prism_challenge.evaluator.official_compare_harness import (
    DEFAULT_MAMBA_PROFILE,
    DEFAULT_TRANSFORMER_PROFILE,
    REPORT_SCHEMA,
    SIDE_A_FAMILY_ID,
    SIDE_B_FAMILY_ID,
    FamilySynthProfile,
    SynthSeedMetrics,
    build_compare_report,
    default_protocol_pin,
    gpu_verification_status,
    package_unknown_style_pair,
    protocol_pin_hash,
    records_for_profile,
    run_dual_family_official_compare,
    synth_challenge_manifest,
)
from prism_challenge.evaluator.official_compare_harness import (
    main as harness_main,
)
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    PROTOCOL_ID,
    OfficialScoreRecord,
    compare_official,
    protocol_budget_constants,
)
from prism_challenge.seed_packaging import REQUIRED_ENTRY_SCRIPTS, SEED_FAMILIES


def test_gpu_verification_status_never_claims_gpu_pass() -> None:
    """VAL-COMP-009 / VAL-COMP-010: harness must not invent GPU verification PASS."""
    gpu = gpu_verification_status()
    assert gpu["claim_gpu_pass"] is False
    assert gpu["status"] in {"DEFERRED", "AVAILABLE"}
    # This mission host has no NVIDIA; status must be DEFERRED when absent.
    if not gpu["nvidia_smi"] or gpu["nvidia_device_count"] == 0:
        assert gpu["status"] == "DEFERRED"
        assert "not found" in gpu["reason"] or "absent" in gpu["reason"]


def test_package_unknown_style_pair_fair_outer_contract(tmp_path: Path) -> None:
    """Seed packaging already exists — dual packages share unknown-style outer shape."""
    assert SIDE_A_FAMILY_ID in SEED_FAMILIES
    assert SIDE_B_FAMILY_ID in SEED_FAMILIES
    packed = package_unknown_style_pair(tmp_path)
    assert set(packed) == {"a", "b"}
    a, b = packed["a"], packed["b"]
    assert a.family_id == SIDE_A_FAMILY_ID
    assert b.family_id == SIDE_B_FAMILY_ID
    assert a.zip_path.is_file() and b.zip_path.is_file()
    # Protocol fairness: same required entries, no secrets / weight blobs.
    for item in (a, b):
        names = set(item.entry_names)
        for required in REQUIRED_ENTRY_SCRIPTS:
            assert required in names
        with zipfile.ZipFile(item.zip_path) as zf:
            zip_names = set(zf.namelist())
        assert not any(n.endswith((".pem", ".key", ".env", ".pt", ".bin")) for n in zip_names)
    # Distinct payloads (different architectures) but identical entry requirements.
    assert a.content_sha256 != b.content_sha256
    assert set(REQUIRED_ENTRY_SCRIPTS).issubset(set(a.entry_names))
    assert set(REQUIRED_ENTRY_SCRIPTS).issubset(set(b.entry_names))


def test_synth_manifest_is_challenge_owned_not_miner_trust_root() -> None:
    pin = default_protocol_pin()
    metrics = SynthSeedMetrics(seed=1337, bpb=1.5, heldout_delta=0.7)
    manifest = synth_challenge_manifest(metrics, pin=pin, family_id=SIDE_A_FAMILY_ID)
    assert manifest["schema_version"] == "prism_run_manifest.v2"
    assert manifest["miner_reported_ignored"] is True
    assert manifest["protocol"]["unknown_style"] is True
    assert manifest["protocol"]["protocol_id"] == PROTOCOL_ID
    assert manifest["protocol"]["protocol_hash"] == protocol_pin_hash(pin)
    assert manifest["data"]["stopped_reason"] == "token_budget"
    # Fairness pin knobs are carried for both families identically.
    constants = protocol_budget_constants()
    assert manifest["protocol"]["token_budget"] == constants["token_budget"]
    assert manifest["protocol"]["tokenizer"] == constants["tokenizer"]
    assert manifest["protocol"]["param_cap"] == constants["param_cap"]


def test_dual_family_official_compare_clear_outcome_without_gpu(tmp_path: Path) -> None:
    """VAL-COMP-009: pytest dual compare without GPU; clear outcome A vs B."""
    report = run_dual_family_official_compare(
        tmp_path,
        side_a_profile=DEFAULT_TRANSFORMER_PROFILE,
        side_b_profile=DEFAULT_MAMBA_PROFILE,
        device_class="fixture",
    )
    assert report["schema"] == REPORT_SCHEMA
    assert report["protocol_id"] == PROTOCOL_ID
    assert report["mode"] == "ArchCompare"
    assert report["device_class"] == "fixture"
    assert report["gpu_verification"]["claim_gpu_pass"] is False
    assert "REAL-PROVIDER" in report["tee_note"]

    # Both sides comparable official records under matched pin.
    a = report["side_a"]
    b = report["side_b"]
    assert a["family_id"] == SIDE_A_FAMILY_ID
    assert b["family_id"] == SIDE_B_FAMILY_ID
    assert a["unknown_style"] is True and b["unknown_style"] is True
    assert a["mean_heldout_delta"] is not None
    assert b["mean_heldout_delta"] is not None
    assert a["mean_bpb"] is not None and b["mean_bpb"] is not None
    assert a["seed_count"] == len(OFFICIAL_DEFAULT_SEEDS)
    assert b["seed_count"] == len(OFFICIAL_DEFAULT_SEEDS)

    # Primary held-out: transformer fixture generalizes better → winner A.
    # Secondary bpb alone would prefer Mamba (lower train bpb) — official invert is visible.
    assert float(a["mean_heldout_delta"]) > float(b["mean_heldout_delta"])
    assert float(a["mean_bpb"]) > float(b["mean_bpb"])
    ranking = report["ranking"]
    assert ranking["winner"] == "a"
    assert ranking["reason"] == "primary_heldout"
    assert ranking["outcome_label"]["winner_side"] == "a"
    assert ranking["outcome_label"]["winner_label"] == a["label"]
    assert ranking["rule"] == "heldout_primary_then_bpb_secondary"

    # Report written for operators.
    report_path = Path(report["report_path"])
    assert report_path.is_file()
    disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert disk["ranking"]["winner"] == "a"
    assert max(report["seeds"]) == max(OFFICIAL_DEFAULT_SEEDS)


def test_matched_pin_and_same_seed_list_for_both_families(tmp_path: Path) -> None:
    pin = default_protocol_pin()
    assert pin.seeds == OFFICIAL_DEFAULT_SEEDS
    assert len(pin.seeds) >= 3
    report = run_dual_family_official_compare(tmp_path, pin=pin)
    assert report["seeds"] == list(OFFICIAL_DEFAULT_SEEDS)
    assert report["pin"]["token_budget"] == pin.token_budget
    assert report["pin"]["tokenizer"] == pin.tokenizer
    assert report["pin"]["param_cap"] == pin.param_cap
    assert report["validity"]["matched_budget"] is True
    assert report["validity"]["wall_clock_never_ranks"] is True
    assert report["validity"]["miner_self_report_never_authoritative"] is True


def test_primary_heldout_invert_visible_against_secondary_bpb(tmp_path: Path) -> None:
    """Clear A-vs-B: held-out primary beats secondary train bpb (protocol invert)."""
    report = run_dual_family_official_compare(tmp_path)
    # Reconstruct pure compare_official to show clear outcome path.
    pin = default_protocol_pin()
    a_records = records_for_profile(DEFAULT_TRANSFORMER_PROFILE, pin=pin)
    b_records = records_for_profile(DEFAULT_MAMBA_PROFILE, pin=pin)
    assert len(a_records) == 3 and len(b_records) == 3
    assert all(isinstance(r, OfficialScoreRecord) for r in a_records + b_records)

    # Aggregate surfaces from the filed report.
    mean_a_held = report["aggregate"]["a"]["mean_heldout_delta"]
    mean_b_held = report["aggregate"]["b"]["mean_heldout_delta"]
    mean_a_bpb = report["aggregate"]["a"]["mean_bpb"]
    mean_b_bpb = report["aggregate"]["b"]["mean_bpb"]
    assert mean_a_held > mean_b_held
    assert mean_a_bpb > mean_b_bpb  # Mamba stronger on secondary
    assert report["ranking"]["winner"] == "a"
    assert report["ranking"]["reason"] == "primary_heldout"

    # Determinism.
    again = run_dual_family_official_compare(tmp_path / "again")
    assert again["ranking"]["winner"] == report["ranking"]["winner"]
    assert again["ranking"]["reason"] == report["ranking"]["reason"]
    assert again["protocol_hash"] == report["protocol_hash"]


def test_flipped_primary_makes_b_win(tmp_path: Path) -> None:
    """If Mamba synth primary held-out is better, outcome flips to B (clear reverse)."""
    strong = FamilySynthProfile(
        family_id=SIDE_B_FAMILY_ID,
        label="mamba-tiny-1m",
        architecture_family="mamba",
        seeds=(
            SynthSeedMetrics(seed=1337, bpb=1.5, heldout_delta=1.20),
            SynthSeedMetrics(seed=2027, bpb=1.5, heldout_delta=1.10),
            SynthSeedMetrics(seed=4242, bpb=1.5, heldout_delta=1.15),
        ),
    )
    weak = FamilySynthProfile(
        family_id=SIDE_A_FAMILY_ID,
        label="transformer-tiny-1m",
        architecture_family="transformer",
        seeds=(
            SynthSeedMetrics(seed=1337, bpb=1.0, heldout_delta=0.10),
            SynthSeedMetrics(seed=2027, bpb=1.0, heldout_delta=0.12),
            SynthSeedMetrics(seed=4242, bpb=1.0, heldout_delta=0.08),
        ),
    )
    report = run_dual_family_official_compare(
        tmp_path,
        side_a_profile=weak,
        side_b_profile=strong,
    )
    assert report["ranking"]["winner"] == "b"
    assert report["ranking"]["reason"] == "primary_heldout"
    assert report["ranking"]["outcome_label"]["winner_label"] == "mamba-tiny-1m"


def test_wall_clock_imbalance_does_not_reorder_outcome(tmp_path: Path) -> None:
    slow_transformer = FamilySynthProfile(
        family_id=SIDE_A_FAMILY_ID,
        label="transformer-tiny-1m",
        architecture_family="transformer",
        seeds=tuple(
            SynthSeedMetrics(
                seed=s.seed,
                bpb=s.bpb,
                heldout_delta=s.heldout_delta,
                train_heldout_gap=s.train_heldout_gap,
                wall_clock_seconds=10_000.0,
            )
            for s in DEFAULT_TRANSFORMER_PROFILE.seeds
        ),
    )
    fast_mamba = FamilySynthProfile(
        family_id=SIDE_B_FAMILY_ID,
        label="mamba-tiny-1m",
        architecture_family="mamba",
        seeds=tuple(
            SynthSeedMetrics(
                seed=s.seed,
                bpb=s.bpb,
                heldout_delta=s.heldout_delta,
                train_heldout_gap=s.train_heldout_gap,
                wall_clock_seconds=1.0,
            )
            for s in DEFAULT_MAMBA_PROFILE.seeds
        ),
    )
    report = run_dual_family_official_compare(
        tmp_path,
        side_a_profile=slow_transformer,
        side_b_profile=fast_mamba,
    )
    # Still A by held-out; wall-clock imbalance cannot crown the faster GPU.
    assert report["ranking"]["winner"] == "a"
    assert report["ranking"]["reason"] == "primary_heldout"


def test_build_compare_report_schema_matches_docs_sketch(tmp_path: Path) -> None:
    packed = package_unknown_style_pair(tmp_path)
    pin = default_protocol_pin()
    a = OfficialScoreRecord(
        label="transformer-tiny-1m",
        bpb=1.8,
        heldout_delta=0.95,
        seed_count=3,
        bpb_std=0.05,
    )
    b = OfficialScoreRecord(
        label="mamba-tiny-1m",
        bpb=1.1,
        heldout_delta=0.25,
        seed_count=3,
        bpb_std=0.05,
    )
    result = compare_official(a, b)
    report = build_compare_report(
        pin=pin,
        side_a=a,
        side_b=b,
        packed=packed,
        result=result,
        mode="ArchCompare",
        device_class="cpu",
        gpu={"status": "DEFERRED", "reason": "unit", "claim_gpu_pass": False},
    )
    for key in (
        "schema",
        "protocol_id",
        "protocol_hash",
        "mode",
        "primary_form",
        "side_a",
        "side_b",
        "seeds",
        "aggregate",
        "ranking",
        "validity",
        "tee_note",
        "gpu_verification",
    ):
        assert key in report
    assert report["ranking"]["winner"] in {"a", "b", "tie"}


def test_harness_cli_offline_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = harness_main(["--output-dir", str(tmp_path), "--device-class", "fixture"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "compare outcome" in captured
    assert "winner=" in captured
    assert (tmp_path / "prism_compare_report.v1.json").is_file()
    assert "gpu_verification" in captured
