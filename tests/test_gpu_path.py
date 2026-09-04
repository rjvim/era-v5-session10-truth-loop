"""Exercising the GPU path from a machine with no GPU.

Task 5 is meant to be re-run on Colab. That means code which has never executed
here will execute there, on someone else's time, and a crash costs a round trip.
There is no CUDA device in the environment this repo was built in, so the GPU
path cannot be *run* — but almost none of the risk is in the GPU itself. It is
in the branches that only a GPU reaches. Three layers, in order of how much of
the real risk they cover:

1. **The dtype branch, executed for real.** A GPU run switches the training
   dtype to bfloat16. PyTorch supports bf16 on CPU, so the entire Task 5
   pipeline can be run in bf16 here — the GEMM benchmark, the step timing, the
   model, the FLOP accounting. This is not a simulation; it is the same code on
   the same dtype, and it catches anything that breaks when float32 is not
   assumed.

2. **The CUDA-only calls, with the device mocked.** `device()`,
   `training_dtype()`, `sync()` and `theoretical_peak()` all branch on
   `torch.cuda.is_available()`. Patching it reaches every one of those lines,
   including each entry in the vendor lookup table, which is otherwise dead code
   on any single machine.

3. **The generated prose, on synthetic GPU artifacts.** `render_readme.render()`
   has branches that only fire when the device is CUDA and when the vendor peak
   is known, plus a 40%-crossing sentence with three outcomes. Feeding it
   fabricated GPU numbers reaches all of them. This layer already caught a real
   bug: sorting the sweep by parameter count alone produced "clearing 40%
   between 10.7M and 10.7M parameters", because two rows share each count.

What none of this covers, stated plainly: kernel-level behaviour, CUDA OOM at
the 8192 benchmark size, and whether bf16 tensor cores change the shape
efficiency conclusion. Those need a real device.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import render_readme  # noqa: E402
import t5_mfu  # noqa: E402
from common import GPTConfig, load_corpus  # noqa: E402


# ===========================================================================
# Layer 1 - the bf16 branch, actually executed
# ===========================================================================
# ---------------------------------------------------------------------------
# bf16-on-CPU availability probe
#
# Layer 1 works by running the GPU dtype (bfloat16) for real on the CPU. That is
# only sensible where the CPU has a usable bf16 matmul. On x86 with oneDNN it
# runs at roughly fp32 speed. On Apple Silicon under torch 2.8 there is no
# native bf16 CPU GEMM and the fallback is ~300x slower, which turned these
# tests into a five-minute hang that looked exactly like a crash.
#
# So the machine is measured before the fact and the bf16 tests are skipped with
# a reason rather than left to grind. Nothing else is lost: layers 2 and 3, which
# carry the actual GPU-branch coverage, do not need bf16 at all.
BF16_MIN_GFLOPS = 5.0


def _bf16_cpu_gflops() -> float:
    a = torch.randn(128, 128, dtype=torch.bfloat16)
    b = torch.randn(128, 128, dtype=torch.bfloat16)
    a @ b                                    # warm up
    t0 = time.perf_counter()
    for _ in range(3):
        a @ b
    dt = time.perf_counter() - t0
    return 2 * 128 ** 3 * 3 / dt / 1e9


_BF16_RATE = _bf16_cpu_gflops()
requires_fast_bf16_cpu = pytest.mark.skipif(
    _BF16_RATE < BF16_MIN_GFLOPS,
    reason=(f"bf16 CPU matmul runs at only {_BF16_RATE:.2f} GFLOP/s "
            f"(< {BF16_MIN_GFLOPS}); this machine has no native bf16 CPU GEMM, so "
            f"running the pipeline in the GPU dtype here would take minutes and "
            f"prove nothing the mocked-device tests do not already cover"),
)


def tiny_peak(cfg, dev, dtype=torch.bfloat16):
    """The peak sweep at test scale.

    The production defaults (up to 2048-square, 10 iterations, 7 repeats) are
    the measurement, not a warm-up, and on hardware with no native bf16 CPU GEMM
    they take minutes. Tests exercise the same code path at a size that finishes
    instantly.
    """
    return t5_mfu.empirical_peak(
        dev, dtype, t5_mfu.model_gemm_shapes(cfg, 4, 32),
        sizes=[64, 128], iters=2, warmup=1, repeats=1,
    )


@pytest.fixture(scope="module")
def corpus():
    _, tok, ids = load_corpus()
    return tok, ids


@requires_fast_bf16_cpu
def test_gemm_benchmark_runs_in_bf16(corpus):
    dev = torch.device("cpu")
    f = t5_mfu.benchmark_gemm(128, 128, 128, dev, torch.bfloat16, iters=2,
                              warmup=1, repeats=2)
    assert f > 0 and torch.isfinite(torch.tensor(f))


@requires_fast_bf16_cpu
def test_full_mfu_pipeline_runs_in_bf16(corpus):
    """The dtype a GPU run would use, end to end."""
    tok, ids = corpus
    dev = torch.device("cpu")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_embd=32,
                    n_head=2)
    peak = t5_mfu.benchmark_gemm(128, 128, 128, dev, torch.bfloat16, iters=2,
                                 warmup=1, repeats=1)
    r = t5_mfu.mfu_for(cfg, 2, 16, ids, dev, torch.bfloat16, peak,
                       iters=2, warmup=1, repeats=1)
    assert 0.0 < r["mfu_6N"]
    assert r["tokens_per_second"] > 0
    assert abs(sum(r["phase_fraction"].values()) - 1.0) < 1e-9


@requires_fast_bf16_cpu
def test_shape_efficiency_runs_in_bf16(corpus):
    tok, _ = corpus
    dev = torch.device("cpu")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_embd=64,
                    n_head=4)
    peak = tiny_peak(cfg, dev)
    se = t5_mfu.shape_efficiency(cfg, 4, 32, peak)
    assert len(se["per_layer"]) == 5
    assert all(r["flops_per_s"] > 0 for r in se["per_layer"])


@requires_fast_bf16_cpu
def test_no_layer_can_exceed_the_peak_in_bf16(corpus):
    """The peak must be a ceiling, not just a reference rate.

    A cache-resident non-square GEMM can beat the largest square one, which had
    one layer reporting 105.4% "of peak" and meant MFU was being divided by
    something that was not an upper bound.
    """
    tok, _ = corpus
    dev = torch.device("cpu")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_embd=64,
                    n_head=4)
    peak = tiny_peak(cfg, dev)
    se = t5_mfu.shape_efficiency(cfg, 4, 32, peak)
    assert se["max_fraction_of_peak"] <= 1.0
    assert all(r["fraction_of_peak"] <= 1.0 for r in se["per_layer"])


# ===========================================================================
# Layer 2 - the CUDA-only branches, with the device mocked
# ===========================================================================
@pytest.fixture
def fake_cuda(monkeypatch):
    def _apply(name: str = "Tesla T4"):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: name)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return _apply


def test_device_selection_picks_cuda(fake_cuda):
    fake_cuda()
    assert t5_mfu.device().type == "cuda"


def test_training_dtype_switches_to_bf16_on_gpu(fake_cuda):
    assert t5_mfu.training_dtype(torch.device("cpu")) is torch.float32
    fake_cuda()
    assert t5_mfu.training_dtype(t5_mfu.device()) is torch.bfloat16


def test_sync_is_called_on_cuda_and_skipped_on_cpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(1))
    t5_mfu.sync(torch.device("cpu"))
    assert calls == []
    t5_mfu.sync(torch.device("cuda"))
    assert calls == [1]


@pytest.mark.parametrize(
    "name", ["A100", "H100", "L4", "T4", "V100", "L40S", "A10G", "RTX 4090"]
)
def test_every_vendor_peak_table_entry_resolves(fake_cuda, name):
    """Otherwise dead code on any one machine."""
    fake_cuda(f"NVIDIA {name}")
    tp = t5_mfu.theoretical_peak(torch.device("cuda"))
    assert tp["matched_key"] == name
    assert tp["vendor_peak_flops_per_s"] > 1e13


def test_unknown_gpu_degrades_gracefully(fake_cuda):
    fake_cuda("NVIDIA Some Unreleased Thing")
    tp = t5_mfu.theoretical_peak(torch.device("cuda"))
    assert tp["vendor_peak_flops_per_s"] is None
    assert "measured peak used instead" in tp["note"]


def test_theoretical_peak_is_none_on_cpu():
    assert t5_mfu.theoretical_peak(torch.device("cpu")) is None


def test_gpu_benchmark_sizes_are_larger(fake_cuda, monkeypatch):
    """The peak sweep must use GPU-scale matrices, not the CPU sizes."""
    seen = []

    def spy(m, k, n, dev, dtype, **kw):
        seen.append(m)
        return 1.0

    monkeypatch.setattr(t5_mfu, "benchmark_gemm", spy)
    t5_mfu.empirical_peak(torch.device("cuda"), torch.bfloat16)
    assert max(seen) == 8192
    seen.clear()
    t5_mfu.empirical_peak(torch.device("cpu"), torch.float32)
    assert max(seen) == 2048


# ===========================================================================
# The contract between producer and consumer
# ===========================================================================
def test_the_real_artifact_has_every_field_the_renderer_reads():
    """A synthetic fixture can be kinder than the pipeline. This checks it isn't.

    `render_readme` reads `environment["device_name"]` through a `.get()` with a
    fallback, so a missing key degrades silently rather than raising - and the
    GPU tests below build their own artifact, so they cannot catch it. They
    didn't: the pipeline was not writing `device_name` at all, and a Colab run
    would have printed "Measured on cuda" instead of naming the card.
    """
    real = json.loads((ROOT / "artifacts" / "t5_mfu.json").read_text())
    for key in ("device", "device_name", "dtype", "torch_threads"):
        assert key in real["environment"], key
        assert real["environment"][key] not in (None, ""), key
    for key in ("peak_flops_per_s", "peak_at_size"):
        assert key in real["peak"], key
    assert "theoretical_peak" in real
    assert render_readme.render(real)


def test_the_synthetic_gpu_artifact_matches_the_real_schema():
    """Keep the fixture honest: same keys as the pipeline actually emits."""
    real = json.loads((ROOT / "artifacts" / "t5_mfu.json").read_text())
    fake = gpu_artifact()
    assert set(fake["environment"]) >= set(real["environment"])
    assert set(fake["headline"]) == set(real["headline"])
    assert set(fake["peak"]) == set(real["peak"])


# ===========================================================================
# Layer 3 - the generated prose, on synthetic GPU artifacts
# ===========================================================================
def gpu_artifact(mfu_scale: float = 1.0, device_name: str = "Tesla T4",
                 vendor: float | None = 65e12) -> dict:
    a = copy.deepcopy(json.loads((ROOT / "artifacts" / "t5_mfu.json").read_text()))
    a["environment"].update({"device": "cuda", "dtype": "torch.bfloat16",
                             "device_name": device_name, "torch_threads": 2})
    a["theoretical_peak"] = {"device": device_name,
                             "vendor_peak_flops_per_s": vendor}
    a["headline"]["sustained_peak_gflops"] = 41_000.0
    a["peak"]["peak_at_size"] = 4096
    for r in a["size_sweep"]:
        r["mfu_6N"] = min(r["mfu_6N"] * mfu_scale, 0.99)
    a["headline"]["best_sweep_mfu_6N"] = max(r["mfu_6N"] for r in a["size_sweep"])
    return a


def test_render_names_the_gpu_and_switches_units():
    out = render_readme.render(gpu_artifact())
    assert "Tesla T4, bfloat16" in out
    assert "41.0 TFLOP/s" in out          # TFLOP/s, not GFLOP/s
    assert "GFLOP/s" not in out


def test_render_includes_the_vendor_comparison_when_known():
    out = render_readme.render(gpu_artifact())
    assert "vendor figure for this device is 65 TFLOP/s" in out
    assert "the harder denominator" in out


def test_render_omits_vendor_comparison_for_an_unknown_gpu():
    out = render_readme.render(gpu_artifact(vendor=None))
    assert "vendor figure" not in out


def test_render_keeps_gflops_on_cpu():
    a = json.loads((ROOT / "artifacts" / "t5_mfu.json").read_text())
    out = render_readme.render(a)
    assert "GFLOP/s" in out and "TFLOP/s at" not in out


@pytest.mark.parametrize(
    "scale,expected",
    [
        (0.3, "without any tested configuration reaching 40%"),
        (1.0, "first clearing 40% at"),
        (3.0, "above 40% at every size tested"),
    ],
)
def test_all_three_branches_of_the_40pct_sentence(scale, expected):
    assert expected in render_readme.render(gpu_artifact(scale))


def test_40pct_sentence_names_a_configuration_not_a_degenerate_range():
    """Regression: sorting by parameter count alone gave '10.7M and 10.7M'.

    Two sweep rows share every parameter count (B=8 and B=32), so the crossing
    must be described by configuration, not by an interval of parameter counts.
    """
    out = render_readme.render(gpu_artifact(1.0))
    assert "between" not in out.split("noise floor")[1].split(".")[0]
    assert "n_embd" in out.split("noise floor")[1].split(".")[0]


def test_generated_section_is_valid_markdown_tables():
    out = render_readme.render(gpu_artifact())
    for block in out.split("\n\n"):
        rows = [r for r in block.splitlines() if r.startswith("|")]
        if len(rows) < 2:
            continue
        widths = {r.count("|") for r in rows}
        assert len(widths) == 1, f"ragged table:\n{block}"
