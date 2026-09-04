"""Invariant tests.

Every number this repo's README states is asserted here against the machine-
generated artifact it came from. The tests read the JSON; they do not recompute
it. So a stale README fails, a re-run that changes a conclusion fails, and a
claim that was never measured has nowhere to hide.

Each claim gets its own test rather than being bundled, so a failure names the
specific sentence in the README that is now wrong.

    pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ARTIFACTS = ROOT / "artifacts"


def load(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet; run python src/run_all.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def t1():
    return load("t1_shapes.json")


@pytest.fixture(scope="module")
def t2():
    return load("t2_finite_diff.json")


@pytest.fixture(scope="module")
def t3():
    return load("t3_accum.json")


@pytest.fixture(scope="module")
def t4():
    return load("t4_gradnorm.json")


@pytest.fixture(scope="module")
def t5():
    return load("t5_mfu.json")


@pytest.fixture(scope="module")
def t6():
    return load("t6_floats.json")


@pytest.fixture(scope="module")
def t6b():
    return load("t6b_bf16_train.json")


# ===========================================================================
# Task 1 - shapes
# ===========================================================================
def test_t1_every_parameter_tensor_has_an_axis_gloss(t1):
    for p in t1["parameters"]:
        assert len(p["axes"]) == len(p["shape"]), p["name"]
        assert all(a.strip() for a in p["axes"]), p["name"]


def test_t1_every_activation_has_an_axis_gloss(t1):
    for a in t1["activations"]:
        assert a["axes"], a["name"]
        if a["shape"]:
            assert len(a["axes"]) == len(a["shape"]), a["name"]


def test_t1_gradient_shapes_match_parameter_shapes(t1):
    assert t1["totals"]["all_grads_match_param_shape"]
    assert t1["totals"]["n_grad_tensors"] == t1["totals"]["n_param_tensors"]


def test_t1_adam_keeps_two_state_tensors_per_parameter(t1):
    assert t1["totals"]["all_optimizer_state_matches_param_shape"]
    assert (t1["totals"]["n_optimizer_state_tensors"]
            == 2 * t1["totals"]["n_param_tensors"])


def test_t1_output_head_is_tied_to_the_embedding(t1):
    assert t1["totals"]["lm_head_shares_storage_with_wte"]


def test_t1_no_two_axis_sizes_collide(t1):
    """The claim that a printed shape is never ambiguous."""
    sizes = t1["totals"]["distinct_axis_sizes"]
    assert len(sizes) == len(set(sizes))
    cfg = t1["config"]
    for a, b in (("B", "T"), ("T", "n_embd"), ("n_embd", "vocab_size"),
                 ("head_dim", "B"), ("head_dim", "T")):
        assert cfg[a] != cfg[b], f"{a} and {b} collide at {cfg[a]}"


def test_t1_mixed_precision_state_is_sixteen_bytes_per_param(t1):
    m = t1["memory_per_param_bytes"]
    assert (m["bf16_weight"] + m["bf16_grad"] + m["fp32_master_weight"]
            + m["adam_exp_avg_fp32"] + m["adam_exp_avg_sq_fp32"]) == m["total"] == 16
    assert m["this_model_state_bytes"] == 16 * t1["totals"]["n_params"]


# ===========================================================================
# Task 2 - gradient by hand
# ===========================================================================
def test_t2_fp64_agrees_to_at_least_eight_digits(t2):
    assert t2["headline"]["fp64_min_matching_digits"] >= 8.0


def test_t2_every_probe_individually_clears_eight_digits_in_fp64(t2):
    for r in t2["fp64"]["results"]:
        assert r["best_matching_digits"] >= 8.0, r["param"]


def test_t2_the_tied_embedding_probe_was_actually_checked(t2):
    """The probe that tests whether autograd summed both paths through wte."""
    tied = [r for r in t2["fp64"]["results"] if r["tied_to_lm_head"]]
    assert len(tied) == 1
    assert tied[0]["best_matching_digits"] >= 8.0


def test_t2_fp32_finite_differences_are_much_worse(t2):
    h = t2["headline"]
    assert h["fp64_beats_fp32"]
    assert h["fp32_min_matching_digits"] < 1.0
    assert h["fp64_min_matching_digits"] - h["fp32_min_matching_digits"] > 6.0


def test_t2_central_difference_beats_forward_difference(t2):
    for r in t2["fp64"]["results"]:
        best_c = max(s["matching_digits_central"] for s in r["sweep"])
        best_f = max(s["matching_digits_forward"] for s in r["sweep"])
        assert best_c > best_f, r["param"]


def test_t2_an_independent_implementation_agrees(t2):
    """gradcheck is PyTorch's own harness, so it checks mine rather than theirs."""
    gc = t2["gradcheck_corroboration"]
    assert gc["all_passed"]
    assert len(gc["results"]) >= 4
    assert all(r["passed"] for r in gc["results"])
    assert all(r["numel"] > 0 for r in gc["results"])


def test_t2_session_toy_example_reproduces(t2):
    ex = t2["instructor_toy_example"]
    assert ex["central_matches_exact"]
    assert ex["exact_derivative"] == ex["session_quoted_value"] == 64.0


# ===========================================================================
# Task 3 - broken gradient accumulation
# ===========================================================================
def test_t3_correct_reduction_reproduces_the_no_accumulation_reference(t3):
    for regime in ("ragged", "uniform"):
        err = t3["step0_gradient_comparison"][regime]["correct"][
            "rel_l2_error_vs_reference"]
        assert err < 1e-12, regime


def test_t3_broken_reduction_is_wrong_when_lengths_differ(t3):
    assert t3["headline"]["step0_broken_rel_l2_error_ragged"] > 0.05


def test_t3_control_the_bug_vanishes_at_equal_lengths(t3):
    """If this fails, the ragged gap is measuring something else."""
    assert t3["headline"]["step0_broken_rel_l2_error_uniform"] < 1e-12
    assert t3["headline"]["max_abs_uniform_gap"] == 0.0
    assert t3["headline"]["control_holds"]


def test_t3_broken_is_consistently_worse_on_held_out_loss(t3):
    h = t3["headline"]
    assert h["broken_is_worse_when_ragged"]
    assert h["ragged_mean_gap_last_half"] > 0
    assert h["ragged_gap_positive_fraction"] == 1.0


def test_t3_the_loss_gap_is_far_smaller_than_the_gradient_error(t3):
    """Why this bug survived to 2024: loud in the gradient, quiet in the loss."""
    grad_err = t3["headline"]["step0_broken_rel_l2_error_ragged"]
    loss_gap_rel = (t3["headline"]["ragged_mean_gap_last_half"]
                    / t3["headline"]["ragged_final_correct"])
    assert loss_gap_rel < grad_err / 5


def test_t3_error_grows_with_length_imbalance(t3):
    assert t3["headline"]["sweep_error_monotone_in_imbalance"]
    zero = [r for r in t3["skew_sweep"] if r["coefficient_of_variation"] == 0]
    assert zero and zero[0]["broken_rel_l2_error"] < 1e-12


# ===========================================================================
# Task 4 - grad norm
# ===========================================================================
def test_t4_norm_fired_while_the_loss_did_not(t4):
    """The central claim, free of the one-step ordering artefact."""
    d = t4["norm_vs_loss_detectability"]
    assert d["norm_would_have_fired"]
    assert not d["loss_would_have_fired"]
    assert d["z_grad"] > 4.0
    assert d["z_dloss"] < 4.0


def test_t4_the_loss_peaks_after_the_norm_spikes(t4):
    h = t4["headline"]
    assert h["steps_from_injection_start_to_peak_loss"] >= 2
    assert h["steps_from_injection_end_to_peak_loss"] >= 1


def test_t4_injection_corpus_is_entirely_in_vocabulary(t4):
    """The Sanskrit is a distribution shift, not an out-of-vocab artefact."""
    from common import load_corpus
    _, tok, _ = load_corpus()
    text = (ROOT / "data" / "sanskrit_hk.txt").read_text(encoding="utf-8")
    assert not (set(text) - set(tok.chars))


def test_t4_a_clean_run_shows_no_spikes_at_the_primary_threshold(t4):
    assert t4["headline"]["clean_run_has_no_lead_events_at_primary_threshold"]


def test_t4_in_healthy_training_a_big_gradient_predicts_lower_loss(t4):
    """Reported as found: the lead relationship is a property of anomalies."""
    assert t4["headline"]["xcorr_lag1_r"]["clean/noclip"] < 0
    assert (t4["headline"]["xcorr_lag1_r"]["inject/noclip"]
            > t4["headline"]["xcorr_lag1_r"]["clean/noclip"])


def test_t4_clipping_helps_but_saturates(t4):
    assert t4["headline"]["clipping_reduces_peak_excursion"]
    assert t4["headline"]["clipping_saturates"]
    assert t4["headline"]["tightest_clip_still_shows_excursion"]


def test_t4_threshold_sweep_is_reported_for_every_threshold(t4):
    zs = [r["z_threshold"] for r in t4["threshold_sensitivity"]]
    assert 4.0 in zs and len(zs) >= 5
    assert sum(r["primary"] for r in t4["threshold_sensitivity"]) == 1


# ===========================================================================
# Task 5 - MFU
# ===========================================================================
def test_t5_peak_is_measured_not_quoted(t5):
    assert t5["peak"]["peak_flops_per_s"] > 0
    assert len(t5["peak"]["square_gemm_sweep"]) >= 4


def test_t5_the_peak_is_actually_a_ceiling(t5):
    """No measured GEMM may beat the denominator MFU is divided by."""
    P = t5["peak"]["peak_flops_per_s"]
    for r in t5["peak"]["gemm_sweep"]:
        assert r["flops_per_s"] <= P + 1e-6, r["label"]
    for r in t5["shape_efficiency"]["per_layer"]:
        assert r["fraction_of_peak"] <= 1.0, r["layer"]
    assert t5["shape_efficiency"]["max_fraction_of_peak"] <= 1.0


def test_t5_peak_sweep_includes_the_models_own_shapes(t5):
    labels = {r["label"] for r in t5["peak"]["gemm_sweep"]}
    for layer in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", "lm_head"):
        assert layer in labels, layer


def test_t5_mfu_is_a_fraction(t5):
    assert 0.0 < t5["headline"]["measured_mfu_6N"] <= 1.0


def test_t5_both_flop_conventions_are_reported(t5):
    h = t5["headline"]
    assert h["measured_mfu_6N_plus_attention"] > h["measured_mfu_6N"]


def test_t5_mfu_improves_with_model_size(t5):
    assert t5["headline"]["mfu_improves_with_size"]
    for bs in sorted({r["batch_rows"] for r in t5["size_sweep"]}):
        rows = sorted([r for r in t5["size_sweep"] if r["batch_rows"] == bs],
                      key=lambda r: r["n_params"])
        assert rows[-1]["mfu_6N"] > rows[0]["mfu_6N"], bs


def test_t5_larger_batch_helps_at_every_size(t5):
    by = {}
    for r in t5["size_sweep"]:
        by.setdefault(r["n_params"], {})[r["batch_rows"]] = r["mfu_6N"]
    for n, d in by.items():
        assert d[max(d)] > d[min(d)], n


def test_t5_attribution_arithmetic_is_self_consistent(t5):
    """Deliberately weak, and labelled as such in the artifact.

    The residual is defined as the remainder, so this identity cannot fail. It
    catches a coding slip in the budget, nothing more.
    """
    assert t5["attribution"]["bookkeeping_identity_holds"]
    for k, v in t5["attribution"]["budget"].items():
        assert v["mfu_lost_pp"] >= -1e-9, k


def test_t5_attribution_declares_which_terms_are_measured(t5):
    a = t5["attribution"]
    assert "remainder_term" in a and "independently_measured_terms" in a
    assert a["remainder_term"] not in a["independently_measured_terms"]
    assert "not evidence" in a["caveat"]


def test_t5_the_sessions_v4_mfu_reproduces_from_the_formula(t5):
    ref = t5["instructor_v4_reference"]
    assert ref["reproduces_quoted_value"]
    assert abs(ref["achieved_tflops_per_s"] - 648.0) < 1.0


# ===========================================================================
# Task 6 - float formats
# ===========================================================================
def test_t6_hand_derived_bits_match_what_pytorch_stores(t6):
    assert t6["all_hand_encodings_match_pytorch"]
    for name, c in t6["cross_check_against_pytorch"].items():
        assert c["bits_match"], name
        assert c["values_match"], name


def test_t6_no_binary_format_stores_0_1_exactly(t6):
    for name, e in t6["encodings"].items():
        assert e["relative_error"] > 0, name
        assert e["stored_value"] != 0.1, name
    assert t6["why_0_1_is_hard"]["repeating_block"] == "0011"


def test_t6_bit_widths_are_correct(t6):
    expect = {"fp32": 32, "fp16": 16, "bf16": 16, "fp8_e4m3": 8}
    for name, n in expect.items():
        assert len(t6["encodings"][name]["bits"]) == n, name


def test_t6_e4m3_encoding_of_0_1(t6):
    e = t6["encodings"]["fp8_e4m3"]
    assert e["bits"] == "00011101"
    assert e["stored_value"] == 0.1015625
    assert abs(e["relative_error_percent"] - 1.5625) < 1e-9


def test_t6_derived_format_limits_match_torch_finfo(t6):
    """The range/precision numbers were derived, so check them against torch."""
    import torch
    pairs = [("fp32", torch.float32), ("fp16", torch.float16),
             ("bf16", torch.bfloat16),
             ("fp8_e4m3", getattr(torch, "float8_e4m3fn", None))]
    for name, dt in pairs:
        if dt is None:
            continue
        p = t6["encodings"][name]["properties"]
        fi = torch.finfo(dt)
        assert p["largest_finite"] == fi.max, name
        assert p["smallest_normal"] == fi.tiny, name


def test_t6_hand_encoder_matches_pytorch_on_random_values():
    """A randomised cross-check, not just the four values in the README.

    Spot-checking 0.1 passes even when the encoder is wrong in the tail. This
    found two real bugs: values in (448, 480] encoded to 01111111, which is
    E4M3FN's reserved NaN code rather than a number, and the overflow flag never
    fired for them. Both are exactly where a hand-written encoder goes wrong,
    and neither is reachable from the headline value.
    """
    import random
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from t6_floats import FORMATS, encode, torch_bits

    from t6_floats import detect_torch_overflow_mode

    # Overflow is a cast convention, not a format property, and PyTorch changed
    # it: 2.8 emits the reserved NaN code, >= 2.13 saturates to 448. So the
    # installed behaviour is detected and the encoder is asked for the matching
    # convention. Everything in range is compared exactly either way.
    mode = detect_torch_overflow_mode().get("mode", "saturate")
    if mode == "unknown":
        pytest.skip("unrecognised E4M3 overflow convention in this PyTorch")

    rng = random.Random(0)
    values = [0.1, 1.0, 0.5, -0.1, 1e-8, 1e-40, 6.1e-5, 0.001953125, 0.015625,
              448.0, 449.0, 460.0, 487.96, -491.0, 500.0, 65504.0, 65600.0,
              1e30, -1e30]
    values += [rng.uniform(-10, 10) for _ in range(150)]
    values += [10 ** rng.uniform(-9, 5) * rng.choice([1, -1]) for _ in range(150)]

    for name, (eb, mb, hi) in FORMATS.items():
        for v in values:
            hand = encode(v, eb, mb, hi, overflow_mode=mode)
            ref = torch_bits(v, name)
            assert hand["bits"] == ref["bits"], (name, v, hand["bits"], ref["bits"])
            if hand["stored_value"] == hand["stored_value"]:   # not NaN
                assert hand["stored_value"] == ref["stored_value"], (name, v)


def test_t6_overflow_convention_is_recorded_not_assumed(t6):
    """The repo must say which convention it observed, not silently pick one."""
    conv = t6["torch_overflow_convention"]
    assert conv["available"]
    assert conv["mode"] in ("saturate", "nan")
    assert conv["torch_version"]
    both = t6["overflow_is_a_cast_mode_not_a_format_property"]
    assert both["saturate"]["bits"] == "01111110"
    assert both["nan"]["bits"] == "01111111"
    assert both["saturate"]["stored_value"] == 448.0


def test_t6_e4m3_never_emits_the_reserved_nan_code():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from t6_floats import encode

    for v in (447.0, 448.0, 449.0, 460.0, 479.0, 480.0, 1e6):
        bits = encode(v, 4, 3, has_inf=False)["bits"]
        assert bits[1:] != "1111111", f"{v} encoded to the NaN code"


def test_t6_overflow_saturates_without_inf_and_goes_to_inf_with_it():
    import math
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from t6_floats import encode

    e4 = encode(1e6, 4, 3, has_inf=False, overflow_mode="saturate")
    assert e4["overflow"] and e4["saturated_to_largest_finite"]
    assert e4["stored_value"] == 448.0

    nan_mode = encode(1e6, 4, 3, has_inf=False, overflow_mode="nan")
    assert nan_mode["overflow"] and not nan_mode["saturated_to_largest_finite"]
    assert nan_mode["stored_value"] != nan_mode["stored_value"]   # NaN

    f16 = encode(1e30, 5, 10, has_inf=True)
    assert f16["overflow"] and not f16["saturated_to_largest_finite"]
    assert math.isinf(f16["stored_value"])


def test_t6_e4m3_max_is_448_not_480(t6):
    """E4M3FN reserves the all-ones mantissa in the top binade for NaN."""
    assert t6["encodings"]["fp8_e4m3"]["properties"]["largest_finite"] == 448.0


def test_t6_bf16_keeps_fp32s_exponent_and_loses_mantissa(t6):
    b = t6["encodings"]["bf16"]["properties"]
    f = t6["encodings"]["fp32"]["properties"]
    assert b["exponent_bits"] == f["exponent_bits"] == 8
    assert b["mantissa_bits"] < f["mantissa_bits"]
    assert b["smallest_normal"] == f["smallest_normal"]


def test_t6_fp16_cannot_reach_where_bf16_can(t6):
    b = t6["encodings"]["bf16"]["properties"]
    h = t6["encodings"]["fp16"]["properties"]
    assert h["smallest_normal"] > b["smallest_normal"] * 1e30
    assert t6["small_value_survival"]["fp16"]["1e-08"]["flushes_to_zero"]
    assert not t6["small_value_survival"]["bf16"]["1e-08"]["flushes_to_zero"]


def test_t6_fp16_is_more_precise_than_bf16_at_0_1(t6):
    """The trade is range for resolution, in both directions."""
    assert (t6["encodings"]["fp16"]["relative_error"]
            < t6["encodings"]["bf16"]["relative_error"])


def test_t6_bf16_loses_no_gradients_but_raw_fp8_loses_many(t6):
    h = t6["headline"]
    assert h["bf16_loses_no_gradients"]
    assert h["gradient_fraction_flushed_fp8"] > 0.3


def test_t6_scaling_rescues_fp8(t6):
    h = t6["headline"]
    assert h["scaling_rescues_fp8"]
    assert h["gradient_fraction_flushed_fp8_with_scaling"] < 0.05


# ===========================================================================
# Task 6b - the recommendation, trained
# ===========================================================================
def test_t6b_bf16_with_master_copy_tracks_fp32(t6b):
    assert t6b["deviation_from_fp32"]["bf16_autocast"]["max_abs_gap_vs_fp32"] < 0.01


def test_t6b_pure_bf16_drifts(t6b):
    assert t6b["deviation_from_fp32"]["pure_bf16"]["max_abs_gap_vs_fp32"] > 0.01


def test_t6b_the_master_copy_is_what_makes_the_difference(t6b):
    h = t6b["headline"]
    assert h["master_copy_helps"]
    assert h["penalty_ratio"] > 10


def test_t6b_all_arms_scored_on_the_same_ruler(t6b):
    assert "always fp32" in t6b["setup"]["evaluation"]
    assert len({len(a["eval_loss"]) for a in t6b["arms"].values()}) == 1


def test_t6b_pure_bf16_arm_really_is_bf16(t6b):
    assert "bfloat16" in t6b["arms"]["pure_bf16"]["param_dtype"]
    assert "float32" in t6b["arms"]["fp32"]["param_dtype"]
    assert "float32" in t6b["arms"]["bf16_autocast"]["param_dtype"]


# ===========================================================================
# Task 1 additions
# ===========================================================================
def test_t1_every_block_was_traced_not_just_the_first(t1):
    assert t1["totals"]["n_blocks_traced"] == t1["config"]["n_layer"]
    assert t1["totals"]["all_blocks_have_identical_activation_shapes"]


def test_t1_per_layer_breakdown_sums_to_the_total(t1):
    by = t1["parameters_by_layer"]
    assert sum(v["n_params"] for v in by.values()) == t1["totals"]["n_params"]
    assert sum(v["n_tensors"] for v in by.values()) == t1["totals"]["n_param_tensors"]


def test_t1_all_parameters_are_trainable(t1):
    for k, v in t1["parameters_by_layer"].items():
        assert all(t["trainable"] for t in v["tensors"]), k


# ===========================================================================
# Task 4 additions
# ===========================================================================
def test_t4_a_clean_run_lead_event_with_lag_above_one_is_documented(t4):
    """The brief asked for one step. This is it, with no injection involved."""
    sec = t4["lead_events_at_secondary_threshold"]
    assert "clean/noclip" in sec
    e = sec["clean/noclip"]
    assert e["lag_steps"] > 1
    assert e["z_grad"] > e["z_dloss_same_step"]
    assert e["z_dloss_at_lag"] > e["z_dloss_same_step"]


def test_t4_loss_is_bit_identical_to_clean_when_the_norm_fires(t4):
    assert t4["headline"]["loss_bit_identical_to_clean_at_the_spike_step"]
    assert t4["norm_vs_loss_detectability"]["loss_at_this_step_vs_clean"] == 0.0


def test_t4_lag_is_reported_from_both_ends_of_the_injection(t4):
    la = t4["injection_response"]["lag_accounting"]
    assert la["steps_from_injection_END_to_peak"] < la["steps_from_injection_START_to_peak"]
    assert t4["headline"]["steps_from_injection_end_to_peak_loss"] >= 1


# ===========================================================================
# cross-task
# ===========================================================================
def test_all_artifacts_record_their_environment():
    for f in sorted(ARTIFACTS.glob("t*.json")):
        d = json.loads(f.read_text())
        assert "_env" in d, f.name
        for k in ("torch", "python", "device_name", "corpus_sha256"):
            assert d["_env"].get(k) is not None, (f.name, k)


def test_all_artifacts_agree_on_the_corpus():
    hashes = {json.loads(f.read_text())["_env"]["corpus_sha256"]
              for f in ARTIFACTS.glob("t*.json")}
    assert len(hashes) == 1


def test_the_precision_story_is_consistent_across_tasks_2_and_6():
    """Task 2's fp32 collapse and Task 6's format argument are the same point."""
    t2, t6 = load("t2_finite_diff.json"), load("t6_floats.json")
    assert t2["headline"]["fp32_min_matching_digits"] < 1.0
    assert t6["encodings"]["fp32"]["properties"]["decimal_digits_of_precision"] < 8
