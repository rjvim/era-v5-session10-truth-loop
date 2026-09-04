"""Task 5 - compute your own MFU, report it honestly, and say what is costing
you the distance to 40%.

MFU = (FLOPs the model mathematically requires per second)
      / (FLOPs the hardware can actually sustain)

Two decisions make the number honest rather than flattering:

1. The denominator is measured, not quoted. A spec-sheet peak is a marketing
   number for one dtype under one sparsity assumption; dividing by it inflates
   or deflates MFU depending on which footnote you picked. Here the denominator
   is the best sustained rate this machine achieves on a large square matmul in
   the same dtype the model trains in, benchmarked in this process. The
   theoretical figure is recorded alongside it when a GPU is present, so both
   are visible.

2. The numerator uses the analytic FLOP count, not a profiler's op count.
   6*N*tokens is the session's formula. It undercounts, because attention's
   score and value matmuls do not involve parameters at all, so the PaLM-style
   term 12*L*T*C per token is reported as well. Quoting whichever is larger
   would be dishonest in the other direction, so both are given and the
   headline uses the strict 6*N form the brief asked for.

The gap to 40% is then attributed with measurements rather than adjectives:

  * phase timing - how much of a step is forward, backward, optimizer, data
  * shape efficiency - the same GEMM shapes the model actually issues,
    benchmarked in isolation, against the machine's best large-GEMM rate. This
    isolates "the matmuls are too small to reach peak" from "there is other work
    happening".
  * a size sweep - MFU against model width and batch size, which shows whether
    the gap is a property of this configuration or of the code.

This script is device-agnostic. Run it on a GPU and every number regenerates.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common import (
    FIGURES,
    read_artifact,
    GPTConfig,
    build_model,
    load_corpus,
    set_seed,
    uniform_batch,
    write_artifact,
)

TARGET_MFU = 0.40


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_dtype(dev) -> torch.dtype:
    """The dtype a real run would use on this device.

    Factored out of the stage functions so the CUDA branch is reachable from a
    test on a machine with no CUDA device.
    """
    return torch.float32 if dev.type == "cpu" else torch.bfloat16


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# denominator: what this machine can actually sustain
# --------------------------------------------------------------------------
def benchmark_gemm(m: int, k: int, n: int, dev, dtype, iters: int = 10,
                   warmup: int = 5, repeats: int = 7) -> float:
    """Sustained FLOP/s for an m x k @ k x n matmul.

    Best-of-N rather than an average. Interference from anything else on the
    machine can only ever make a timing slower, never faster, so the maximum is
    the least contaminated estimate of what the hardware can do. Averaging here
    made the per-layer numbers swing by 28 points between identical runs.
    """
    a = torch.randn(m, k, device=dev, dtype=dtype)
    b = torch.randn(k, n, device=dev, dtype=dtype)
    for _ in range(warmup):
        a @ b
    sync(dev)
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        sync(dev)
        best = max(best, 2.0 * m * k * n * iters / (time.perf_counter() - t0))
    return best


def empirical_peak(dev, dtype, extra_shapes: list | None = None,
                   sizes: list | None = None, **bench_kw) -> dict:
    """Fastest rate this machine actually reaches, in the training dtype.

    Large square matmuls alone are NOT a safe denominator. A short wide GEMM
    that fits in cache can beat the biggest square one: on this machine
    mlp.c_proj (1024x512 @ 512x128) ran at 155.5 GFLOP/s against a 2048-square
    peak of 147.6, which made one layer report 105.4% "of peak" and meant the
    denominator was not an upper bound at all. MFU divided by a non-ceiling is
    not MFU.

    So the model's own shapes are benchmarked as part of the peak sweep, and the
    denominator is the maximum over everything measured. It is then a genuine
    observed ceiling, and shape efficiency cannot exceed 100%.
    """
    # `sizes` and `bench_kw` are overridable so tests can exercise this code path
    # cheaply. They must never be overridden when producing artifacts: the
    # default sweep IS the measurement. A test that called this with production
    # sizes in bfloat16 ran for many minutes on hardware without a native bf16
    # CPU GEMM, which is how this parameter came to exist.
    if sizes is None:
        sizes = [256, 512, 1024, 2048] if dev.type == "cpu" else [1024, 2048, 4096, 8192]
    rows = [{"shape": [s, s, s], "label": f"square {s}",
             "flops_per_s": benchmark_gemm(s, s, s, dev, dtype, **bench_kw)}
            for s in sizes]
    for label, m, k, n in (extra_shapes or []):
        rows.append({"shape": [m, k, n], "label": label,
                     "flops_per_s": benchmark_gemm(m, k, n, dev, dtype, **bench_kw)})
    best = max(rows, key=lambda r: r["flops_per_s"])
    return {
        "gemm_sweep": rows,
        "square_gemm_sweep": [r for r in rows if r["label"].startswith("square")],
        "peak_flops_per_s": best["flops_per_s"],
        "peak_at_shape": best["shape"],
        "peak_at_label": best["label"],
        "peak_at_size": best["shape"][0],
        "method": "maximum sustained rate over large square matmuls AND the "
                  "model's own GEMM shapes, in the training dtype; the model's "
                  "shapes are included because a cache-resident non-square GEMM "
                  "can beat the biggest square one, which would leave the "
                  "denominator below an observed rate",
    }


def model_gemm_shapes(cfg: GPTConfig, B: int, T: int) -> list:
    """Every linear layer's (M, K, N), as the forward pass issues them."""
    M, C = B * T, cfg.n_embd
    return [
        ("attn.c_attn", M, C, 3 * C),
        ("attn.c_proj", M, C, C),
        ("mlp.c_fc", M, C, 4 * C),
        ("mlp.c_proj", M, 4 * C, C),
        ("lm_head", M, C, cfg.vocab_size),
    ]


def theoretical_peak(dev) -> dict | None:
    if dev.type != "cuda":
        return None
    name = torch.cuda.get_device_name(0)
    # Vendor bf16/fp16 dense tensor-core figures. Present for comparison only;
    # the headline MFU uses the measured denominator.
    table = {
        "A100": 312e12, "H100": 989e12, "L4": 121e12, "T4": 65e12,
        "V100": 125e12, "L40S": 362e12, "A10G": 125e12, "RTX 4090": 165e12,
    }
    # Longest key first. Substring matching in dict order made "L4" win against
    # an L40S, which would have scored it at 121 TFLOP/s instead of 362 - a 3x
    # error in the denominator, in the direction that flatters the result.
    for key in sorted(table, key=len, reverse=True):
        if key in name:
            return {"device": name, "vendor_peak_flops_per_s": table[key],
                    "matched_key": key}
    return {"device": name, "vendor_peak_flops_per_s": None,
            "note": "device not in the lookup table; measured peak used instead"}


# --------------------------------------------------------------------------
# numerator: FLOPs the model requires
# --------------------------------------------------------------------------
def flops_per_token(cfg: GPTConfig, n_params: int, n_params_nonembd: int,
                    T: int) -> dict:
    six_n = 6 * n_params
    six_n_nonembd = 6 * n_params_nonembd
    attn = 12 * cfg.n_layer * T * cfg.n_embd   # scores + value matmuls, fwd+bwd
    return {
        "six_n_all_params": six_n,
        "six_n_non_embedding": six_n_nonembd,
        "attention_term_12_L_T_C": attn,
        "six_n_plus_attention": six_n + attn,
        "attention_share_of_total": attn / (six_n + attn),
    }


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------
def time_step(cfg: GPTConfig, B: int, T: int, ids, dev, dtype,
              iters: int = 12, warmup: int = 4, repeats: int = 3) -> dict:
    """Time a full training step, phase by phase.

    Best-of-N for the same reason the GEMM benchmark is: contention on a shared
    machine can only ever make a pass slower. The fastest complete pass is the
    least contaminated estimate, and `repeatability()` reports how much the
    remaining noise moves the headline.
    """
    set_seed(1337)
    model = build_model(cfg).to(dev, dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(0)
    batches = [uniform_batch(ids, B, T, gen) for _ in range(iters + warmup)]
    batches = [(x.to(dev), y.to(dev)) for x, y in batches]

    best_total, best_phase = float("inf"), None
    for _ in range(repeats):
        phase = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        for i, (x, y) in enumerate(batches):
            rec = i >= warmup
            sync(dev); t = time.perf_counter()
            x2, y2 = x.contiguous(), y.contiguous()
            sync(dev)
            if rec: phase["data"] += time.perf_counter() - t

            t = time.perf_counter()
            _, loss = model(x2, y2)
            sync(dev)
            if rec: phase["forward"] += time.perf_counter() - t

            t = time.perf_counter()
            model.zero_grad(set_to_none=True)
            loss.backward()
            sync(dev)
            if rec: phase["backward"] += time.perf_counter() - t

            t = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sync(dev)
            if rec: phase["optimizer"] += time.perf_counter() - t

        total = sum(phase.values())
        if total < best_total:
            best_total, best_phase = total, phase

    return {
        "iters_timed": iters,
        "timing_repeats": repeats,
        "seconds_per_step": best_total / iters,
        "tokens_per_step": B * T,
        "tokens_per_second": iters * B * T / best_total,
        "phase_seconds_per_step": {k: v / iters for k, v in best_phase.items()},
        "phase_fraction": {k: v / best_total for k, v in best_phase.items()},
        "n_params": model.n_params(),
        "n_params_non_embedding": model.n_params(non_embedding=True),
    }


def shape_efficiency(cfg: GPTConfig, B: int, T: int, peak: dict) -> dict:
    """How close the model's own GEMM shapes get to the machine's best rate.

    The per-layer rates are READ from the peak sweep rather than measured again.
    Re-benchmarking would reintroduce the bug this guards against: two separate
    timings of the same shape differ by a few percent, so a layer could exceed
    100% of a peak that was, in the same breath, defined as the maximum
    including that layer. Reusing the measurements makes the ratio exact.
    """
    shapes = model_gemm_shapes(cfg, B, T)
    by_label = {r["label"]: r for r in peak["gemm_sweep"]}
    P = peak["peak_flops_per_s"]
    rows = []
    for name, m, k, n in shapes:
        measured = by_label[name]
        assert measured["shape"] == [m, k, n], (name, measured["shape"], [m, k, n])
        rows.append(
            {
                "layer": name, "M": m, "K": k, "N": n,
                "flops_per_s": measured["flops_per_s"],
                "fraction_of_peak": measured["flops_per_s"] / P,
                "flops_per_call": 2 * m * k * n,
            }
        )
    tot_f = sum(r["flops_per_call"] for r in rows)
    weighted = sum(r["flops_per_call"] / r["flops_per_s"] for r in rows)
    return {
        "per_layer": rows,
        "flop_weighted_fraction_of_peak": (tot_f / weighted) / P,
        "max_fraction_of_peak": max(r["fraction_of_peak"] for r in rows),
        "note": "forward-pass linear layers only, rates taken from the same "
                "benchmark sweep that defined the peak, so no layer can exceed it",
    }


def mfu_for(cfg: GPTConfig, B: int, T: int, ids, dev, dtype, peak: float,
            **kw) -> dict:
    t = time_step(cfg, B, T, ids, dev, dtype, **kw)
    fpt = flops_per_token(cfg, t["n_params"], t["n_params_non_embedding"], T)
    achieved_6n = fpt["six_n_all_params"] * t["tokens_per_second"]
    achieved_full = fpt["six_n_plus_attention"] * t["tokens_per_second"]
    return {
        **t,
        "flops_per_token": fpt,
        "achieved_flops_per_s_6N": achieved_6n,
        "achieved_flops_per_s_6N_plus_attention": achieved_full,
        "peak_flops_per_s": peak,
        "mfu_6N": achieved_6n / peak,
        "mfu_6N_plus_attention": achieved_full / peak,
    }


# --------------------------------------------------------------------------
def instructor_reference() -> dict:
    """The session's own V4 number, recomputed from the formula.

    The transcript reads "6848 teraflops" achieved against 7912 peak for 8.2%
    MFU, which cannot all be true at once - 6848/7912 is 87%. Recomputing from
    the stated model size and throughput gives 648 TFLOP/s, and 648/7912 = 8.2%,
    matching the MFU he quoted. The 6848 is a transcription slip for 648, and
    the formula reproduces his figure exactly.
    """
    n, tps, peak = 9e9, 12_000, 7912e12
    achieved = 6 * n * tps
    return {
        "params": n,
        "tokens_per_second": tps,
        "cluster_peak_flops_per_s": peak,
        "achieved_flops_per_s": achieved,
        "achieved_tflops_per_s": achieved / 1e12,
        "mfu": achieved / peak,
        "session_quoted_mfu": 0.082,
        "reproduces_quoted_value": abs(achieved / peak - 0.082) < 0.001,
    }


def repeatability(cfg: GPTConfig, B: int, T: int, ids, dev, dtype, peak: float,
                  trials: int = 4) -> dict:
    """How much does the MFU number move between identical runs?

    Everything else in this repo is deterministic given a seed. This is not:
    it is a wall-clock measurement on a machine with other things happening on
    it. Quoting a single MFU to four significant figures would imply a
    precision the method does not have, so the spread is measured and reported
    with the number.
    """
    vals = [mfu_for(cfg, B, T, ids, dev, dtype, peak, repeats=1)["mfu_6N"]
            for _ in range(trials)]
    mean = float(np.mean(vals))
    return {
        "trials": trials,
        "mfu_6N_values": vals,
        "mean": mean,
        "std": float(np.std(vals, ddof=1)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "coefficient_of_variation": float(np.std(vals, ddof=1) / mean),
        "spread_pp": float((np.max(vals) - np.min(vals)) * 100),
    }


def stage_measure() -> None:
    dev = device()
    dtype = training_dtype(dev)
    _, tok, ids = load_corpus()
    print(f"device={dev} dtype={dtype} threads={torch.get_num_threads()}", flush=True)
    base_cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64)
    B, T = 16, 64
    peak = empirical_peak(dev, dtype, model_gemm_shapes(base_cfg, B, T))
    P = peak["peak_flops_per_s"]
    print(f"   sustained peak {P/1e9:.2f} GFLOP/s on {peak['peak_at_label']}",
          flush=True)
    baseline = mfu_for(base_cfg, B, T, ids, dev, dtype, P)
    print(f"   {baseline['tokens_per_second']:.0f} tok/s, "
          f"MFU(6N) {baseline['mfu_6N']*100:.2f}%", flush=True)
    rep = repeatability(base_cfg, B, T, ids, dev, dtype, P)
    print(f"   MFU over {rep['trials']} single-pass runs: {rep['mean']*100:.2f}% "
          f"+/- {rep['std']*100:.2f} (spread {rep['spread_pp']:.2f} pp)", flush=True)
    shapes = shape_efficiency(base_cfg, B, T, peak)
    print(f"   FLOP-weighted shape efficiency "
          f"{shapes['flop_weighted_fraction_of_peak']*100:.1f}% of peak", flush=True)
    write_artifact("_t5_measure.json", {
        "peak": peak, "baseline": {**baseline, "B": B, "T": T},
        "repeatability": rep, "shape_efficiency": shapes,
        "theoretical_peak": theoretical_peak(dev),
        "environment": {
            "device": str(dev),
            "device_name": (torch.cuda.get_device_name(0)
                            if dev.type == "cuda" else platform.processor()
                            or platform.machine()),
            "dtype": str(dtype),
            "torch_threads": torch.get_num_threads(),
        },
    })


def stage_sweep() -> None:
    dev = device()
    dtype = training_dtype(dev)
    _, tok, ids = load_corpus()
    P = read_artifact("_t5_measure.json")["peak"]["peak_flops_per_s"]
    T = 64
    sweep = []
    for n_embd, n_layer in ((128, 4), (256, 4), (384, 6), (512, 6)):
        for bs in (8, 32):
            cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=T,
                            n_embd=n_embd, n_layer=n_layer,
                            n_head=max(4, n_embd // 64))
            r = mfu_for(cfg, bs, T, ids, dev, dtype, P, iters=8, repeats=3)
            sweep.append({
                "n_embd": n_embd, "n_layer": n_layer, "batch_rows": bs,
                "n_params": r["n_params"],
                "tokens_per_second": r["tokens_per_second"],
                "mfu_6N": r["mfu_6N"],
                "mfu_6N_plus_attention": r["mfu_6N_plus_attention"],
                "optimizer_fraction": r["phase_fraction"]["optimizer"],
            })
            print(f"   n_embd={n_embd} L={n_layer} B={bs}: MFU {r['mfu_6N']*100:.2f}% "
                  f"({r['n_params']/1e6:.2f}M params)", flush=True)
    write_artifact("_t5_sweep.json", {"size_sweep": sweep})


def main() -> None:
    m = read_artifact("_t5_measure.json")
    peak, baseline = m["peak"], m["baseline"]
    rep, shapes = m["repeatability"], m["shape_efficiency"]
    P = peak["peak_flops_per_s"]
    sweep = read_artifact("_t5_sweep.json")["size_sweep"]
    best = max(sweep, key=lambda r: r["mfu_6N"])

    ph = baseline["phase_fraction"]
    non_matmul_overhead = ph["optimizer"] + ph["data"]
    shape_eff = shapes["flop_weighted_fraction_of_peak"]
    ceiling_after_shape = shape_eff
    ceiling_after_overhead = ceiling_after_shape * (1 - non_matmul_overhead)
    residual = ceiling_after_overhead - baseline["mfu_6N"]

    attribution = {
        "measured_mfu_6N": baseline["mfu_6N"],
        "target": TARGET_MFU,
        "distance_to_target_pp": (TARGET_MFU - baseline["mfu_6N"]) * 100,
        "budget": {
            "1_small_gemm_shapes": {
                "explanation": "the model's real matmul shapes reach only this "
                               "fraction of the machine's large-GEMM rate",
                "fraction_of_peak_reachable": shape_eff,
                "mfu_lost_pp": (1 - shape_eff) * 100,
            },
            "2_time_not_spent_in_matmuls": {
                "explanation": "optimizer step and batch preparation, measured",
                "fraction_of_step": non_matmul_overhead,
                "mfu_lost_pp": (ceiling_after_shape - ceiling_after_overhead) * 100,
            },
            "3_residual": {
                "explanation": "elementwise work counted in neither the 6N FLOP "
                               "model nor the GEMM benchmark: LayerNorm, GELU, "
                               "softmax, the causal mask, residual adds, plus "
                               "kernel launch and memory traffic",
                "mfu_lost_pp": residual * 100,
            },
        },
        "bookkeeping_identity_holds": abs(
            (1 - (1 - shape_eff) - (ceiling_after_shape - ceiling_after_overhead)
             - residual) - baseline["mfu_6N"]
        ) < 1e-9,
        "caveat": "the residual is DEFINED as what is left after the first two "
                  "terms, so 'bookkeeping_identity_holds' is an arithmetic check "
                  "on this script, not evidence that the decomposition is right. "
                  "The first two terms are independently measured (a GEMM "
                  "benchmark at the model's real shapes, and phase timing); the "
                  "third is a remainder and is only as trustworthy as the claim "
                  "that those two are exhaustive.",
        "independently_measured_terms": ["1_small_gemm_shapes",
                                         "2_time_not_spent_in_matmuls"],
        "remainder_term": "3_residual",
    }

    payload = {
        "environment": {
            **m["environment"],
            "note": "this repo's headline numbers were produced on CPU; the script "
                    "is device-agnostic and regenerates every figure on a GPU",
            "timing_caveat": "every number in this artifact is a wall-clock "
                             "measurement and is the only part of the repo that is "
                             "not reproducible to the digit; see 'repeatability'",
        },
        "peak": peak,
        "theoretical_peak": m["theoretical_peak"],
        "baseline": baseline,
        "repeatability": rep,
        "shape_efficiency": shapes,
        "size_sweep": sweep,
        "best_configuration": best,
        "attribution": attribution,
        "instructor_v4_reference": instructor_reference(),
        "headline": {
            "measured_mfu_6N": baseline["mfu_6N"],
            "mfu_6N_mean_over_trials": rep["mean"],
            "mfu_6N_std_over_trials": rep["std"],
            "mfu_6N_spread_pp": rep["spread_pp"],
            "measured_mfu_6N_plus_attention": baseline["mfu_6N_plus_attention"],
            "tokens_per_second": baseline["tokens_per_second"],
            "sustained_peak_gflops": P / 1e9,
            "best_sweep_mfu_6N": best["mfu_6N"],
            "best_sweep_config": {k: best[k] for k in
                                  ("n_embd", "n_layer", "batch_rows", "n_params")},
            "mfu_improves_with_size": best["mfu_6N"] > baseline["mfu_6N"],
            "flop_weighted_shape_efficiency": shape_eff,
            "optimizer_fraction_of_step": ph["optimizer"],
            "distance_to_40pct_pp": attribution["distance_to_target_pp"],
        },
    }
    write_artifact("t5_mfu.json", payload)
    plot(payload)


def plot(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    sw = payload["size_sweep"]
    for bs in sorted({r["batch_rows"] for r in sw}):
        rows = [r for r in sw if r["batch_rows"] == bs]
        rows.sort(key=lambda r: r["n_params"])
        ax.plot([r["n_params"] / 1e6 for r in rows],
                [r["mfu_6N"] * 100 for r in rows], "o-", lw=2, label=f"B={bs} rows")
    ax.axhline(payload["attribution"]["target"] * 100, color="r", ls="--", lw=1,
               label="40% target")
    ax.set_xscale("log")
    ax.set_title("MFU vs model size")
    ax.set_xlabel("parameters (millions, log)"); ax.set_ylabel("MFU % (6N)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    ph = payload["baseline"]["phase_fraction"]
    ks = list(ph.keys())
    ax.bar(ks, [ph[k] * 100 for k in ks], color="tab:green", alpha=.8)
    for i, k in enumerate(ks):
        ax.text(i, ph[k] * 100, f"{ph[k]*100:.1f}%", ha="center", va="bottom",
                fontsize=8)
    ax.set_title("Where a step's time goes")
    ax.set_ylabel("% of step time"); ax.grid(alpha=.3, axis="y")

    ax = axes[2]
    se = payload["shape_efficiency"]["per_layer"]
    ax.barh([r["layer"] for r in se], [r["fraction_of_peak"] * 100 for r in se],
            color="tab:orange", alpha=.85)
    ax.axvline(payload["shape_efficiency"]["flop_weighted_fraction_of_peak"] * 100,
               color="k", ls="--", lw=1, label="FLOP-weighted")
    ax.set_title("Each layer's GEMM shape, as % of the machine's large-GEMM rate")
    ax.set_xlabel("% of sustained peak"); ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="x")

    fig.tight_layout()
    out = FIGURES / "t5_mfu.png"
    fig.savefig(out, dpi=130)
    print(f"[figure] {out}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "measure"):
        stage_measure()
    if stage in ("all", "sweep"):
        stage_sweep()
    if stage in ("all", "combine"):
        main()
