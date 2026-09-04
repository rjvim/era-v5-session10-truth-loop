"""README consistency tests.

`test_invariants.py` checks that the experiments say what they are supposed to
say. This file checks something different and easier to get wrong: that the
numbers typed into README.md are the numbers the artifacts actually contain.

Every entry below is a literal string that must appear in the README, paired
with the artifact field it was copied from. Re-running an experiment and getting
a different value now breaks the build instead of silently leaving a stale
figure in the prose.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text()


def art(name: str) -> dict:
    p = ARTIFACTS / name
    if not p.exists():
        pytest.skip(f"{name} not generated; run python src/run_all.py")
    return json.loads(p.read_text())


def claim(readme: str, text: str) -> None:
    assert text in readme, f"README does not contain {text!r}"


def claim_approx(readme: str, label: str, value: float, tol: float,
                 unit: str = "") -> None:
    """For wall-clock numbers, which do not reproduce to the digit.

    Finds the number the README prints next to `label` and checks it is within
    `tol` of what the artifact currently says. Exact-string matching would make
    Task 5 fail on every rerun, and loosening it to nothing would let a wrong
    number through, so the tolerance is the measured run-to-run spread.
    """
    i = readme.find(label)
    assert i >= 0, f"README does not mention {label!r}"
    window = readme[i:i + 400]
    nums = []
    for m in re.finditer(r"([-+]?\d[\d,]*(?:\.\d+)?)" + re.escape(unit), window):
        try:
            nums.append(float(m.group(1).replace(",", "").rstrip(".")))
        except ValueError:
            continue
    assert any(abs(n - value) <= tol for n in nums), (
        f"near {label!r} the README shows {nums[:6]}, "
        f"artifact says {value:.4f} (tolerance {tol})"
    )


# ---------------------------------------------------------------- Task 1
def test_readme_parameter_count(readme):
    n = art("t1_shapes.json")["totals"]["n_params"]
    claim(readme, f"{n:,}")


def test_readme_tensor_counts(readme):
    t = art("t1_shapes.json")["totals"]
    claim(readme, f"{t['n_param_tensors']} parameter tensors")
    claim(readme, f"{t['n_traced_activations']} traced activations")
    claim(readme, f"{t['n_optimizer_state_tensors']} Adam state tensors")


def test_readme_memory_figures(readme):
    m = art("t1_shapes.json")["memory_per_param_bytes"]
    claim(readme, f"**{m['total']} bytes**")
    claim(readme, f"{m['this_model_state_MiB']:.2f} MiB")
    claim(readme, f"**{m['extrapolated_checkpoint_TiB_at_120B']:.2f} TiB**")


def test_readme_per_layer_table(readme):
    by = art("t1_shapes.json")["parameters_by_layer"]
    for k, val in by.items():
        claim(readme, f"| {val['n_tensors']} | {val['n_params']:,} |")


# ---------------------------------------------------------------- Task 6b
def test_readme_bf16_training_table(readme):
    h = art("t6b_bf16_train.json")["headline"]
    claim(readme, f"{h['fp32_final']:.6f}")
    claim(readme, f"{h['bf16_autocast_final']:.6f}")
    claim(readme, f"{h['pure_bf16_final']:.6f}")
    claim(readme, f"**{h['bf16_autocast_max_gap']:.6f}**")
    claim(readme, f"**{h['pure_bf16_max_gap']:.6f}**")
    claim(readme, f"**{h['penalty_ratio']:.0f}× more**")


def test_readme_mentions_the_colab_path(readme):
    assert "Regenerating Task 5 on a GPU" in readme
    assert "src/t5_mfu.py measure" in readme


# ---------------------------------------------------------------- Task 2
def test_readme_finite_difference_digits(readme):
    h = art("t2_finite_diff.json")["headline"]
    claim(readme, f"**{h['fp64_min_matching_digits']:.2f}** significant digits")
    claim(readme, f"**{h['fp32_min_matching_digits']:.2f}** significant digits")
    claim(readme, f"{h['fp64_median_matching_digits']:.2f}")
    claim(readme, f"{h['fp32_median_matching_digits']:.2f}")


def test_readme_tied_probe_digits(readme):
    t2 = art("t2_finite_diff.json")
    tied = [r for r in t2["fp64"]["results"] if r["tied_to_lm_head"]][0]
    claim(readme, f"{tied['best_matching_digits']:.2f} digits")


def test_readme_layernorm_fp32_example(readme):
    t2 = art("t2_finite_diff.json")
    r = [x for x in t2["fp32"]["results"]
         if x["param"] == "blocks.1.ln_1.weight"][0]
    assert "blocks.1.ln_1.weight" in readme
    for v in (r["autograd_grad"], r["best_central_difference"]):
        mantissa = f"{abs(v):.4e}".replace("e-0", "e-")
        assert mantissa in readme, f"{v} not quoted in README as {mantissa}"


# ---------------------------------------------------------------- Task 3
def test_readme_accum_step0_errors(readme):
    h = art("t3_accum.json")["headline"]
    claim(readme, f"**{h['step0_broken_rel_l2_error_ragged']:.3f}**")


def test_readme_accum_micro_batch_lengths(readme):
    counts = art("t3_accum.json")["step0_gradient_comparison"]["ragged"][
        "valid_tokens_per_micro_batch"]
    claim(readme, ", ".join(str(c) for c in counts))


def test_readme_accum_final_losses(readme):
    h = art("t3_accum.json")["headline"]
    claim(readme, f"{h['ragged_final_correct']:.6f}")
    claim(readme, f"{h['ragged_final_broken']:.6f}")
    claim(readme, f"**+{h['ragged_final_gap']:.6f}**")
    claim(readme, f"**+{h['ragged_mean_gap_last_half']:.4f}**")


# ---------------------------------------------------------------- Task 4
def test_readme_detectability(readme):
    d = art("t4_gradnorm.json")["norm_vs_loss_detectability"]
    claim(readme, f"**{d['z_grad']:.2f}**")
    claim(readme, f"**{d['z_dloss']:.2f}**")
    claim(readme, f"**{d['detectability_ratio']:.1f}×**")


def test_readme_injection_response(readme):
    t4 = art("t4_gradnorm.json")
    h, ir = t4["headline"], t4["injection_response"]
    claim(readme, f"{ir['grad_norm_before_injection']:.3f}")
    claim(readme, f"{ir['grad_norm_at_injection']:.3f}")
    claim(readme, f"**+{h['peak_excess_loss_noclip']:.3f}**")


def test_readme_lag_is_quoted_from_both_ends(readme):
    la = art("t4_gradnorm.json")["injection_response"]["lag_accounting"]
    claim(readme, f"steps {la['injection_window'][0]}–{la['injection_window'][1]}")
    claim(readme, f"step {la['peak_loss_step']}")
    claim(readme, f"| {la['steps_from_injection_START_to_peak']} |")
    claim(readme, f"| **{la['steps_from_injection_END_to_peak']}** |")


def test_readme_documents_the_clean_run_lead_event(readme):
    e = art("t4_gradnorm.json")["lead_events_at_secondary_threshold"]["clean/noclip"]
    claim(readme, f"step {e['step']}")
    claim(readme, f"lag **{e['lag_steps']}**")
    claim(readme, f"{e['grad_norm_multiple_of_trailing_median']:.2f}×")
    claim(readme, f"z = {e['z_grad']:.2f}")
    claim(readme, f"z = {e['z_dloss_same_step']:.2f}")
    claim(readme, f"z = {e['z_dloss_at_lag']:.2f}")


def test_readme_bit_identical_claim_is_true(readme):
    d = art("t4_gradnorm.json")["norm_vs_loss_detectability"]
    assert d["loss_at_this_step_vs_clean"] == 0.0
    claim(readme, "bit-identical to the clean arm")


def test_readme_cross_correlation(readme):
    x = art("t4_gradnorm.json")["headline"]["xcorr_lag1_r"]
    claim(readme, f"**{x['clean/noclip']:.3f}**")
    claim(readme, f"+{x['inject/noclip']:.3f}")
    claim(readme, f"+{x['inject/clip1.0']:.3f}")


def test_readme_clip_sweep_row(readme):
    ms = art("t4_gradnorm.json")["mitigation_sweep"]
    for r in ms:
        claim(readme, f"{r['peak_excess_loss_vs_clean']:.3f}")


def test_readme_clean_run_lag_is_the_documented_one(readme):
    """The README now names a specific clean-run step rather than a lag list."""
    lags = art("t4_gradnorm.json")["headline"]["clean_run_lead_event_lags_at_z3"]
    e = art("t4_gradnorm.json")["lead_events_at_secondary_threshold"]["clean/noclip"]
    assert e["lag_steps"] in lags
    claim(readme, f"lag **{e['lag_steps']}**")


# ---------------------------------------------------------------- Task 5
#
# Task 5 is wall-clock. These tests use the measured run-to-run spread as the
# tolerance instead of matching strings, and separately assert that the README
# actually discloses the spread rather than quoting a bare number.


def test_readme_declares_task5_is_not_reproducible(readme):
    assert "does not reproduce to the digit" in readme


def test_readme_task5_section_is_generated_and_current(readme):
    """Task 5's prose is generated from its artifact, so it cannot go stale.

    Re-running the benchmark on other hardware rewrites the section; this asserts
    nobody has since hand-edited it or re-run Task 5 without re-rendering.
    """
    sys.path.insert(0, str(ROOT / "src"))
    import render_readme

    assert render_readme.BEGIN in readme and render_readme.END in readme
    i = readme.index(render_readme.BEGIN)
    j = readme.index(render_readme.END) + len(render_readme.END)
    assert readme[i:j] == render_readme.render(art("t5_mfu.json")), (
        "README Task 5 section is stale; run: python src/render_readme.py"
    )


def test_readme_mfu_within_measured_spread(readme):
    h = art("t5_mfu.json")["headline"]
    tol = max(2.0, h["mfu_6N_spread_pp"])
    claim_approx(readme, "**MFU (6N)**", h["measured_mfu_6N"] * 100, tol, "%")
    claim_approx(readme, "across 4 single-pass trials",
                 h["mfu_6N_mean_over_trials"] * 100, tol, "%")
    claim_approx(readme, "MFU (6N + attention)",
                 h["measured_mfu_6N_plus_attention"] * 100, tol, "%")


def test_readme_quotes_the_spread_itself(readme):
    h = art("t5_mfu.json")["headline"]
    claim_approx(readme, "single-pass trials", h["mfu_6N_std_over_trials"] * 100, 0.5)
    claim_approx(readme, "(spread ", h["mfu_6N_spread_pp"], 0.5, " pp")


def test_readme_peak_and_throughput(readme):
    h = art("t5_mfu.json")["headline"]
    claim_approx(readme, "benchmarked in-process", h["sustained_peak_gflops"], 15.0,
                 " GFLOP/s")
    claim_approx(readme, "throughput", h["tokens_per_second"], 800, " tokens/s")


def test_readme_attribution_budget(readme):
    b = art("t5_mfu.json")["attribution"]["budget"]
    claim_approx(readme, "matmul shapes too small",
                 b["1_small_gemm_shapes"]["mfu_lost_pp"], 5.0, " pp")
    claim_approx(readme, "time not in matmuls",
                 b["2_time_not_spent_in_matmuls"]["mfu_lost_pp"], 2.0, " pp")
    claim_approx(readme, "elementwise residual",
                 b["3_residual"]["mfu_lost_pp"], 8.0, " pp")
    claim_approx(readme, "benchmarked in isolation reach only",
                 b["1_small_gemm_shapes"]["fraction_of_peak_reachable"] * 100, 8.0, "%")


def test_readme_size_sweep_table_within_tolerance(readme):
    """Each sweep row's MFU must appear near its parameter count."""
    tol = max(3.0, art("t5_mfu.json")["headline"]["mfu_6N_spread_pp"])
    for r in art("t5_mfu.json")["size_sweep"]:
        label = f"| {r['n_embd']} | {r['n_layer']} | {r['batch_rows']} |"
        claim_approx(readme, label, r["mfu_6N"] * 100, tol, "%")
        claim(readme, f"{r['n_params']/1e6:.2f}M")


def test_readme_per_layer_shape_efficiency_range(readme):
    fr = [r["fraction_of_peak"] * 100
          for r in art("t5_mfu.json")["shape_efficiency"]["per_layer"]]
    claim_approx(readme, "Per-layer shape efficiency runs", min(fr), 8.0, "%")
    claim_approx(readme, "Per-layer shape efficiency runs", max(fr), 8.0, "%")


def test_readme_phase_split(readme):
    ph = art("t5_mfu.json")["baseline"]["phase_fraction"]
    claim_approx(readme, "A step splits", ph["forward"] * 100, 6.0, "%")
    claim_approx(readme, "A step splits", ph["backward"] * 100, 6.0, "%")
    claim_approx(readme, "A step splits", ph["optimizer"] * 100, 3.0, "%")


def test_readme_v4_reconstruction(readme):
    """Not wall-clock: this is arithmetic and must match exactly."""
    ref = art("t5_mfu.json")["instructor_v4_reference"]
    claim(readme, f"{ref['achieved_tflops_per_s']:.0f} TFLOP/s")
    claim(readme, f"{ref['mfu']*100:.2f}%")


# ---------------------------------------------------------------- Task 6
def test_readme_bit_patterns(readme):
    enc = art("t6_floats.json")["encodings"]
    for name, e in enc.items():
        claim(readme, f"`{e['exponent_bits']}`")
        claim(readme, f"`{e['mantissa_bits']}`")
        claim(readme, f"{e['stored_value']:.18f}")


def test_readme_format_errors(readme):
    enc = art("t6_floats.json")["encodings"]
    claim(readme, f"**{enc['fp8_e4m3']['relative_error_percent']:.4f}%**")
    claim(readme, f"{enc['bf16']['relative_error_percent']:.4f}%")
    claim(readme, f"{enc['fp16']['relative_error_percent']:.4f}%")


def test_readme_format_limits(readme):
    props = {k: v["properties"] for k, v in art("t6_floats.json")["encodings"].items()}
    claim(readme, f"**{props['fp8_e4m3']['largest_finite']:.0f}**")
    claim(readme, f"{props['fp16']['largest_finite']:,.0f}")
    for k in ("fp32", "fp16", "bf16", "fp8_e4m3"):
        claim(readme, f"{props[k]['decimal_digits_of_precision']:.2f}")


def test_readme_gradient_cast_table(readme):
    g = art("t6_floats.json")["gradient_experiment"]
    claim(readme, f"{g['per_format']['fp16']['fraction_flushed_to_zero']*100:.4f}%")
    claim(readme, f"**{g['per_format']['bf16']['fraction_flushed_to_zero']*100:.4f}%**")
    claim(readme, f"**{g['per_format']['fp8_e4m3']['fraction_flushed_to_zero']*100:.2f}%**")
    claim(readme,
          f"**{g['with_global_scaling']['fp8_e4m3']['fraction_flushed_to_zero']*100:.2f}%**")
    claim(readme, f"{g['gradient_stats']['dynamic_range_orders_of_magnitude']:.1f} orders")


# ---------------------------------------------------------------- meta
def test_readme_test_count_is_accurate(readme):
    """The README states how many tests exist; make that self-checking."""
    total = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        total += len(re.findall(r"^def test_", f.read_text(), re.M))
    claim(readme, f"{total} test functions")


def test_readme_has_no_accidental_block_markers(readme):
    """A wrapped line starting with '+', '-', '>' or '#' is not prose to Markdown.

    GitHub rendered "+ 4 + 4 (Adam m, v) = 16 bytes" as a bullet, because the
    line wrap happened to put a plus at column zero. Only visible in the
    rendered page, never in the source.
    """
    in_fence = False
    for i, line in enumerate(readme.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        # A marker only starts a block when followed by a space ("+0.0000" is
        # prose; "+ 4" is a bullet), except ">" which always quotes.
        starts_block = line[:2] in ("+ ", "- ", "* ", "# ") or line.startswith(">")
        if starts_block:
            prev = readme.splitlines()[i - 2] if i >= 2 else ""
            assert prev.strip() == "" or prev[:2] in ("+ ", "- ", "* ", "# ") \
                or prev.startswith((">", "|", " ")), (
                f"line {i} starts a Markdown block mid-paragraph: {line[:60]!r}"
            )


def test_readme_limitations_do_not_restate_live_generated_numbers(readme):
    """Hand-copied *current* figures go stale the moment Task 5 is re-run.

    Historical numbers in the bug narratives (the layer that once reported
    105.4% of peak) are fixed facts and are fine. What must not appear is a
    value that regenerates, because nothing would update it here.
    """
    lim = readme[readme.index("## Limitations"):readme.index("## Sources")]
    h = art("t5_mfu.json")["headline"]
    live = {
        "measured MFU": f"{h['measured_mfu_6N'] * 100:.1f}%",
        "trial mean": f"{h['mfu_6N_mean_over_trials'] * 100:.1f}%",
        "spread": f"{h['mfu_6N_spread_pp']:.1f} pp",
        "std": f"{h['mfu_6N_std_over_trials'] * 100:.1f} pp",
    }
    for label, text in live.items():
        assert text not in lim, (
            f"Limitations restates the live {label} ({text}); it regenerates, so "
            f"reference the Task 5 section instead of copying the figure"
        )


def test_readme_has_a_limitations_section(readme):
    assert "## Limitations" in readme
    assert "Not settled by this work" in readme
