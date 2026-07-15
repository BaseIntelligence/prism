"""P10 logic probe suite unit fixtures (VAL-REASON-002..008).

CPU-only / pure-torch paths. No lm-eval, no Lium, no REAL TEE claim.
Proves all MUST probe metrics are computable for dual-family fixtures and
that multihop_binding is role/composition (distinct from MQAR).
"""

from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.benchmarks.logic_suite import (
    DEFAULT_TRIALS_PER_PROBE,
    GENERATOR_BY_PROBE,
    LOGIC_PROBE_KEYS,
    documented_logic_suite,
    fixture_forced_ce_from_accuracy,
    gen_boolean_parity_xor,
    gen_multihop_binding,
    gen_reverse_edit,
    generate_logic_suite,
    generate_probe_trials,
    logic_trial_seed,
    oracle_predictions,
    probe_forced_answer_ce,
    pure_torch_fixture_model,
    score_logic_from_predictions,
    score_probe_fixture,
    score_probe_with_logits,
    score_suite_fixture,
    tokenize_simple,
)
from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_REASONING_CHANCE_TABLE,
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCORECARD_ID,
    REASONING_SUITE_ID,
    assert_complete_view_document,
    validate_complete_view_document,
)
from prism_challenge.evaluator.complete_view_reasoning import (
    assert_probe_dual_channel,
    build_complete_view_with_reasoning,
    dual_family_reasoning_fixture,
    family_reasoning_fixture,
    fill_reasoning_panel,
    probe_metric_bundle,
)
from prism_challenge.evaluator.official_comparison import OfficialScoreRecord
from prism_challenge.evaluator.scorecard_suite import relative_to_chance


def _rec(*, label: str, heldout_delta: float = 0.5) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=1.2,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        valid=True,
        seed_count=3,
        bpb_std=0.01,
        heldout_std=0.02,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
    )


def _acc_profile(base: float, *, bump: dict[str, float] | None = None) -> dict[str, float]:
    """Per-probe accuracy profile clamped to [0,1]."""
    out = {k: float(base) for k in LOGIC_PROBE_KEYS}
    if bump:
        for k, v in bump.items():
            out[k] = max(0.0, min(1.0, float(v)))
    return out


# --- Suite identity / authorship -------------------------------------------------


def test_logic_suite_identity_no_lm_eval() -> None:
    doc = documented_logic_suite()
    assert doc["suite_id"] == REASONING_SUITE_ID == "logic_synthetic.v1"
    assert set(doc["probes"]) == set(LOGIC_PROBE_KEYS)
    assert doc["chance_table"] == dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE)
    assert doc["no_lm_eval_dependency"] is True
    assert "mqar" in doc["distinct_from"]
    assert "gsm8k" in doc["distinct_from"]["gsm8k_mmlu_lm_eval"].lower()
    # Generators are challenge-owned callables, not external kits.
    assert set(GENERATOR_BY_PROBE) == set(LOGIC_PROBE_KEYS)


def test_logic_trial_seed_deterministic() -> None:
    a = logic_trial_seed(probe="boolean_parity_xor", trial_i=3)
    b = logic_trial_seed(probe="boolean_parity_xor", trial_i=3)
    c = logic_trial_seed(probe="boolean_parity_xor", trial_i=4)
    assert a == b
    assert a != c


# --- VAL-REASON-002: boolean / parity / XOR --------------------------------------


def test_val_reason_002_boolean_parity_xor_scoreable_both_sides() -> None:
    trials = generate_probe_trials("boolean_parity_xor", n_trials=24)
    assert len(trials) == 24
    for t in trials:
        assert t.probe == "boolean_parity_xor"
        assert t.gold in t.candidates
        assert set(t.candidates) == {"0", "1"}
        assert "XOR" in t.prompt or "parity" in t.prompt

    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.70),
        b_acc=_acc_profile(0.55),
        n_trials=24,
    )
    sa = dual.a.probes["boolean_parity_xor"]
    sb = dual.b.probes["boolean_parity_xor"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.5)
    assert sb.chance == pytest.approx(0.5)
    assert sa.accuracy == pytest.approx(0.70, abs=1 / 24 + 1e-9)
    assert sb.accuracy == pytest.approx(0.55, abs=1 / 24 + 1e-9)
    assert sa.forced_ce is not None and sb.forced_ce is not None
    # Higher accuracy → lower forced CE (fixture mapping)
    assert sa.forced_ce < sb.forced_ce

    panel = dual.panel()
    entry = panel["probes"]["boolean_parity_xor"]
    assert entry["status"] == "filled"
    assert entry["acc"]["a"] == pytest.approx(sa.accuracy)
    assert entry["acc"]["b"] == pytest.approx(sb.accuracy)
    assert entry["forced_ce"]["a"] is not None
    assert entry["forced_ce"]["b"] is not None
    assert entry["rel_to_chance"]["a"] == pytest.approx(relative_to_chance(sa.accuracy, 0.5))


# --- VAL-REASON-003: arithmetic digit/mod ----------------------------------------


def test_val_reason_003_arith_digit_mod_scoreable_both_sides() -> None:
    trials = generate_probe_trials("arith_digit_mod", n_trials=16)
    for t in trials:
        assert t.gold in DIGIT_RANGE(t)
        assert t.gold in t.candidates
        assert t.prompt.startswith("arith:")
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.20, bump={"arith_digit_mod": 0.40}),
        b_acc=_acc_profile(0.20, bump={"arith_digit_mod": 0.25}),
        n_trials=20,
    )
    sa = dual.a.probes["arith_digit_mod"]
    sb = dual.b.probes["arith_digit_mod"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.1)
    bundle = probe_metric_bundle(dual.a, dual.b)
    assert "arith_digit_mod" in bundle
    assert bundle["arith_digit_mod"]["acc"]["a"] == pytest.approx(sa.accuracy)


def DIGIT_RANGE(t) -> set[str]:  # noqa: N802 - helper name kept close to test use
    return set(str(i) for i in range(10))


# --- VAL-REASON-004: transitive comparison ---------------------------------------


def test_val_reason_004_transitive_compare_scoreable_both_sides() -> None:
    trials = generate_probe_trials("transitive_compare", n_trials=12)
    for t in trials:
        assert t.gold in {">", "<", "=", "?"}
        assert "rel:" in t.prompt
        assert t.meta.get("hops") in (1, 2, 3, 4)
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.30, bump={"transitive_compare": 0.50}),
        b_acc=_acc_profile(0.30, bump={"transitive_compare": 0.40}),
    )
    sa = dual.a.probes["transitive_compare"]
    sb = dual.b.probes["transitive_compare"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.25)
    panel = dual.panel()
    assert panel["probes"]["transitive_compare"]["acc"]["a"] is not None
    assert panel["probes"]["transitive_compare"]["acc"]["b"] is not None


# --- VAL-REASON-005: multihop binding ≠ MQAR -------------------------------------


def test_val_reason_005_multihop_binding_distinct_from_mqar() -> None:
    trials = generate_probe_trials("multihop_binding", n_trials=16)
    assert trials
    for t in trials:
        assert t.meta.get("distinct_from_mqar") is True
        assert t.meta.get("binding_style") == "function_composition"
        assert "P(" in t.prompt  # composition formula
        # Not MQAR-style key=value associative grid text
        assert "mqar" not in t.prompt.lower()
        assert "needle_key" not in t.prompt
        assert t.gold in t.candidates

    # Explicit structural contrast to retrieval templates
    t0 = gen_multihop_binding(0)
    assert "bind:" in t0.prompt
    assert "start=a=" in t0.prompt

    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.20, bump={"multihop_binding": 0.35}),
        b_acc=_acc_profile(0.20, bump={"multihop_binding": 0.30}),
    )
    sa = dual.a.probes["multihop_binding"]
    sb = dual.b.probes["multihop_binding"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.125)
    panel = dual.panel()
    mh = panel["probes"]["multihop_binding"]
    assert mh["status"] == "filled"
    assert mh["acc"]["a"] is not None and mh["acc"]["b"] is not None
    # Panel still documents MQAR separation
    assert "mqar" in panel["distinct_from"]["P3_long_ctx_mqar"].lower()


# --- VAL-REASON-006: sort/order + reverse/edit -----------------------------------


def test_val_reason_006_sort_and_reverse_edit_reported_separately() -> None:
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(
            0.40,
            bump={"sort_order": 0.55, "reverse_edit": 0.45},
        ),
        b_acc=_acc_profile(
            0.40,
            bump={"sort_order": 0.50, "reverse_edit": 0.35},
        ),
    )
    sort_a = dual.a.probes["sort_order"]
    rev_a = dual.a.probes["reverse_edit"]
    sort_b = dual.b.probes["sort_order"]
    rev_b = dual.b.probes["reverse_edit"]
    for s in (sort_a, rev_a, sort_b, rev_b):
        assert_probe_dual_channel(s)
    assert sort_a.chance == pytest.approx(0.25)
    assert rev_a.chance == pytest.approx(0.25)

    # reverse/edit must be transform, not identity copy
    t_rev = gen_reverse_edit(0)
    assert t_rev.meta.get("distinct_from_copy") is True
    assert t_rev.gold != t_rev.meta.get("src")  # transformation applied

    panel = dual.panel()
    assert panel["probes"]["sort_order"]["acc"]["a"] is not None
    assert panel["probes"]["sort_order"]["acc"]["b"] is not None
    assert panel["probes"]["reverse_edit"]["acc"]["a"] is not None
    assert panel["probes"]["reverse_edit"]["acc"]["b"] is not None
    # Separate fields (not fused)
    assert "sort_order" in panel["probes"]
    assert "reverse_edit" in panel["probes"]
    assert panel["probes"]["sort_order"] is not panel["probes"]["reverse_edit"]


# --- VAL-REASON-007: count stream + dyck nesting ---------------------------------


def test_val_reason_007_count_stream_and_dyck_reported() -> None:
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.25, bump={"count_stream": 0.35, "dyck_nesting": 0.60}),
        b_acc=_acc_profile(0.25, bump={"count_stream": 0.30, "dyck_nesting": 0.55}),
    )
    ca = dual.a.probes["count_stream"]
    da = dual.a.probes["dyck_nesting"]
    cb = dual.b.probes["count_stream"]
    db = dual.b.probes["dyck_nesting"]
    for s in (ca, da, cb, db):
        assert_probe_dual_channel(s)
    assert ca.chance == pytest.approx(0.1)
    assert da.chance == pytest.approx(0.5)

    trials_c = generate_probe_trials("count_stream", n_trials=8)
    assert all("count:" in t.prompt for t in trials_c)
    trials_d = generate_probe_trials("dyck_nesting", n_trials=9)
    assert all("dyck:" in t.prompt for t in trials_d)

    panel = dual.panel()
    assert panel["probes"]["count_stream"]["forced_ce"]["a"] is not None
    assert panel["probes"]["dyck_nesting"]["acc"]["b"] is not None


# --- VAL-REASON-008: instruction-toy + contradiction -----------------------------


def test_val_reason_008_instruction_and_contradiction_scored() -> None:
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(
            0.30,
            bump={"instruction_toy": 0.40, "contradiction_detect": 0.65},
        ),
        b_acc=_acc_profile(
            0.30,
            bump={"instruction_toy": 0.35, "contradiction_detect": 0.55},
        ),
    )
    ia = dual.a.probes["instruction_toy"]
    ca = dual.a.probes["contradiction_detect"]
    ib = dual.b.probes["instruction_toy"]
    cb = dual.b.probes["contradiction_detect"]
    for s in (ia, ca, ib, cb):
        assert_probe_dual_channel(s)
    assert ia.chance == pytest.approx(0.2)
    assert ca.chance == pytest.approx(0.5)

    trials_i = generate_probe_trials("instruction_toy", n_trials=8)
    assert any(t.prompt.startswith("FMTA|") or t.prompt.startswith("instr:") for t in trials_i)
    trials_c = generate_probe_trials("contradiction_detect", n_trials=8)
    assert all(t.gold in ("consistent", "inconsistent") for t in trials_c)

    panel = dual.panel()
    assert panel["probes"]["instruction_toy"]["status"] == "filled"
    assert panel["probes"]["contradiction_detect"]["status"] == "filled"
    assert panel["probes"]["instruction_toy"]["acc"]["a"] is not None
    assert panel["probes"]["contradiction_detect"]["acc"]["b"] is not None


# --- Dual channel + panel assembly + pure-torch hooks ----------------------------


def test_all_must_probes_fixture_suite_and_panel_fill() -> None:
    scores = score_suite_fixture(n_trials=12, accuracy_by_probe=_acc_profile(0.5))
    assert set(scores) == set(LOGIC_PROBE_KEYS)
    for key, sc in scores.items():
        assert_probe_dual_channel(sc)
        assert sc.probe == key

    fam_a = family_reasoning_fixture(
        accuracy_by_probe=_acc_profile(0.6),
        n_trials=12,
        seeds=(1337, 2027, 4242),
    )
    fam_b = family_reasoning_fixture(
        accuracy_by_probe=_acc_profile(0.4),
        n_trials=12,
        suite_seed=99,
        seeds=(1337, 2027, 4242),
    )
    panel = fill_reasoning_panel(a=fam_a, b=fam_b)
    assert panel["status"] == "filled"
    assert panel["suite_id"] == REASONING_SUITE_ID
    assert panel["scoring"]["closed_choice_accuracy"] is True
    assert panel["scoring"]["forced_ce"] is True
    assert panel["scoring"]["chance_baselines"] is True
    for key in LOGIC_PROBE_KEYS:
        p = panel["probes"][key]
        assert p["status"] == "filled"
        assert p["acc"]["a"] is not None and p["acc"]["b"] is not None
        assert p["forced_ce"]["a"] is not None and p["forced_ce"]["b"] is not None
        assert p["rel_to_chance"]["a"] is not None

    # Full complete_view document with P10 probe metrics
    doc = build_complete_view_with_reasoning(
        _rec(label="transformer-tiny-1m", heldout_delta=0.4),
        _rec(label="mamba-tiny-1m", heldout_delta=0.55),
        family_a=fam_a,
        family_b=fam_b,
        score_class="fixture",
    )
    assert doc["schema"] == COMPLETE_VIEW_SCHEMA == "complete_view.v1.3"
    assert doc["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert doc["real_provider_tee"] == "BLOCKED"
    p10 = doc["panels"]["P10_reasoning_logic"]
    assert p10["status"] == "filled"
    for key in LOGIC_PROBE_KEYS:
        assert p10["probes"][key]["acc"]["a"] is not None
        assert p10["probes"][key]["acc"]["b"] is not None
    problems = validate_complete_view_document(doc)
    assert problems == []
    assert_complete_view_document(doc)


def test_oracle_vs_chance_relative_to_chance() -> None:
    trials = generate_probe_trials("boolean_parity_xor", n_trials=40)
    oracle = score_logic_from_predictions(trials, oracle_predictions(trials))
    assert oracle.accuracy == pytest.approx(1.0)
    assert oracle.relative == pytest.approx(1.0)
    # Chance ~0.5 binary: fixture forced ce mapping uses relative
    ce_hi = fixture_forced_ce_from_accuracy(1.0, chance=0.5)
    ce_lo = fixture_forced_ce_from_accuracy(0.5, chance=0.5)
    assert ce_hi < ce_lo


def test_pure_torch_cpu_scoring_path() -> None:
    """Prefer pure-torch CPU path for model-facing hooks (no trained_state load)."""
    bundle = pure_torch_fixture_model(vocab_size=256, seed=7)
    trials = generate_probe_trials("boolean_parity_xor", n_trials=8)
    # Forced CE path alone
    ce = probe_forced_answer_ce(bundle.nll_fn, trials[0].prompt, trials[0].gold)
    assert math.isfinite(ce) and ce > 0
    # Closed-choice logits path (may be near chance; must still produce dual metrics)
    score = score_probe_with_logits(
        trials,
        bundle.logits_fn,
        probe="boolean_parity_xor",
        nll_fn=bundle.nll_fn,
        device="cpu",
    )
    assert score.probe == "boolean_parity_xor"
    assert score.trials == 8
    assert 0.0 <= score.accuracy <= 1.0
    assert score.chance == pytest.approx(0.5)
    assert score.forced_ce is not None and math.isfinite(score.forced_ce)
    assert score.device == "cpu"
    # Tokenizer is deterministic single-byte map
    assert tokenize_simple("AB") == [ord("A"), ord("B")]


def test_suite_generate_covers_all_probes() -> None:
    suite = generate_logic_suite(n_trials=4)
    assert set(suite) == set(LOGIC_PROBE_KEYS)
    for _key, trials in suite.items():
        assert len(trials) == 4
        assert all(t.gold in t.candidates for t in trials)


def test_score_probe_fixture_default_trials() -> None:
    sc = score_probe_fixture("contradiction_detect")
    assert sc.trials == DEFAULT_TRIALS_PER_PROBE
    assert_probe_dual_channel(sc)


def test_boolean_generator_modes_include_xor_and_parity() -> None:
    modes = {gen_boolean_parity_xor(i).meta["mode"] for i in range(8)}
    assert "xor" in modes and "parity" in modes
