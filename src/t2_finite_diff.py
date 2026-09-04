"""Task 2 - verify a gradient by hand.

Nudge a weight, measure how the loss actually changed, compare to what
backward() reported.

The session's worked example was: weight 3 -> loss 16, weight 3.01 -> loss
16.064, therefore dL/dw = 0.064 / 0.01 = 6.4 per 0.001... i.e. 64. That is a
forward difference with a large step. We reproduce that form, then do the job
properly:

* central difference (L(w+h) - L(w-h)) / 2h, whose error is O(h^2) rather than
  O(h), so it can actually reach the "several decimals" the brief asks for;
* an h-sweep, so the agreement is shown to be a real limit and not one lucky
  step size;
* fp64 and fp32 side by side. In fp32 the check plateaus around 3-4 digits no
  matter how small h gets, because L(w+h) and L(w-h) are two nearly-equal
  numbers being subtracted. That plateau is not a bug in backward(); it is
  catastrophic cancellation, and it is the same precision story as Task 6.

We deliberately include wte.weight, which is tied to lm_head.weight. Perturbing
one scalar there changes both the input embedding and the output logits. If
autograd only accumulated one of the two paths the check would fail, so this
entry is the one that actually tests something.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from common import (
    GPTConfig,
    build_model,
    load_corpus,
    set_seed,
    uniform_batch,
    write_artifact,
)

B, T = 4, 32
H_SWEEP = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# (parameter name, flat index) - chosen to cover embedding, attention, MLP,
# LayerNorm and the tied output head.
PROBES = [
    ("wte.weight", 731),                      # tied: also the lm_head row
    ("wpe.weight", 100),
    ("blocks.0.attn.c_attn.weight", 5000),
    ("blocks.0.attn.c_proj.bias", 7),
    ("blocks.2.mlp.c_fc.weight", 20000),
    ("blocks.3.mlp.c_proj.weight", 31000),
    ("blocks.1.ln_1.weight", 42),
    ("ln_f.weight", 11),
]


def loss_at(model, x, y, param, idx, delta):
    """Loss with one scalar of one parameter shifted by delta. No grad graph."""
    with torch.no_grad():
        original = param.view(-1)[idx].item()
        param.view(-1)[idx] = original + delta
        _, loss = model(x, y)
        param.view(-1)[idx] = original
    return loss.item()


def matching_digits(a: float, b: float) -> float:
    if a == b:
        return math.inf
    denom = max(abs(a), abs(b))
    if denom == 0:
        return math.inf
    return -math.log10(abs(a - b) / denom)


def run(dtype) -> dict:
    set_seed(1337)
    _, tok, ids = load_corpus()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=T)
    model = build_model(cfg, dtype=dtype)
    model.eval()  # no dropout in this model, but be explicit: the loss must be
                  # a deterministic function of the weights or the check is void
    gen = torch.Generator().manual_seed(0)
    x, y = uniform_batch(ids, B, T, gen)

    _, loss = model(x, y)
    base_loss = loss.item()
    model.zero_grad(set_to_none=True)
    loss.backward()

    named = dict(model.named_parameters())
    results = []
    for pname, idx in PROBES:
        p = named[pname]
        analytic = p.grad.view(-1)[idx].item()
        w0 = p.view(-1)[idx].item()

        sweep = []
        for h in H_SWEEP:
            lp = loss_at(model, x, y, p, idx, +h)
            lm = loss_at(model, x, y, p, idx, -h)
            central = (lp - lm) / (2 * h)
            forward = (lp - base_loss) / h
            sweep.append(
                {
                    "h": h,
                    "loss_plus": lp,
                    "loss_minus": lm,
                    "central_difference": central,
                    "forward_difference": forward,
                    "abs_err_central": abs(central - analytic),
                    "rel_err_central": abs(central - analytic) / max(abs(analytic), 1e-300),
                    "matching_digits_central": matching_digits(central, analytic),
                    "matching_digits_forward": matching_digits(forward, analytic),
                }
            )
        best = max(sweep, key=lambda s: s["matching_digits_central"])
        results.append(
            {
                "param": pname,
                "flat_index": idx,
                "weight_value": w0,
                "autograd_grad": analytic,
                "best_h": best["h"],
                "best_central_difference": best["central_difference"],
                "best_matching_digits": best["matching_digits_central"],
                "tied_to_lm_head": pname == "wte.weight",
                "sweep": sweep,
            }
        )
    return {
        "dtype": str(dtype),
        "base_loss": base_loss,
        "results": results,
        "min_matching_digits": min(r["best_matching_digits"] for r in results),
        "median_matching_digits": sorted(r["best_matching_digits"] for r in results)[
            len(results) // 2
        ],
    }


def gradcheck_corroboration() -> dict:
    """A second opinion from an implementation I did not write.

    Everything above is my finite-difference harness checking PyTorch. If that
    harness has a bug - a sign error, a stale weight left perturbed, the wrong
    flat index - it could agree with autograd for the wrong reason, and nothing
    in this file would notice. `torch.autograd.gradcheck` is PyTorch's own
    reference implementation of the same idea, written by other people, so
    running it on the same model is an independent check on the harness rather
    than on the engine.

    A smaller model is used because gradcheck evaluates the loss twice per
    scalar, so cost scales with the size of the tensor being checked.
    """
    from torch.func import functional_call

    set_seed(1337)
    _, tok, ids = load_corpus()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2,
                    n_embd=32, n_head=2)
    model = build_model(cfg, dtype=torch.float64)
    model.eval()
    gen = torch.Generator().manual_seed(0)
    x, y = uniform_batch(ids, 2, 16, gen)
    params = dict(model.named_parameters())

    results = []
    for name in ("blocks.0.mlp.c_fc.bias", "blocks.1.ln_1.weight",
                 "blocks.0.attn.c_proj.bias", "ln_f.bias"):
        w0 = params[name].detach().clone().requires_grad_(True)

        def f(w, _name=name):
            merged = {k: (w if k == _name else v) for k, v in params.items()}
            return functional_call(model, merged, (x, y))[1]

        ok = torch.autograd.gradcheck(f, (w0,), eps=1e-6, atol=1e-8, rtol=1e-5)
        results.append({"param": name, "numel": int(w0.numel()), "passed": bool(ok)})

    return {
        "note": "torch.autograd.gradcheck on a 2-layer fp64 model; an independent "
                "implementation checking this file's own harness",
        "config": {"n_layer": 2, "n_embd": 32, "n_head": 2, "B": 2, "T": 16},
        "eps": 1e-6, "atol": 1e-8, "rtol": 1e-5,
        "results": results,
        "all_passed": all(r["passed"] for r in results),
    }


def instructor_example() -> dict:
    """Reproduce the session's own toy example so the arithmetic is on record.

    L(w) = (24/3 * w - 20)^2 with w = 3 gives y = 24, L = 16.
    """
    def L(w):
        return (8.0 * w - 20.0) ** 2

    w = 3.0
    fwd = (L(w + 0.01) - L(w)) / 0.01
    ctr = (L(w + 0.01) - L(w - 0.01)) / 0.02
    exact = 2 * (8 * w - 20) * 8
    return {
        "note": "session's blackboard example: y = 8w, target 20, squared loss",
        "w": w,
        "loss_at_w": L(w),
        "loss_at_w_plus_0.01": L(w + 0.01),
        "forward_difference": fwd,
        "central_difference": ctr,
        "exact_derivative": exact,
        "session_quoted_value": 64.0,
        "central_matches_exact": abs(ctr - exact) < 1e-9,
    }


def main() -> None:
    fp64 = run(torch.float64)
    fp32 = run(torch.float32)
    print("corroborating with torch.autograd.gradcheck ...", flush=True)
    gc = gradcheck_corroboration()

    payload = {
        "setup": {
            "B": B,
            "T": T,
            "h_sweep": H_SWEEP,
            "probes": [{"param": p, "flat_index": i} for p, i in PROBES],
            "method": "central difference (L(w+h)-L(w-h))/2h, error O(h^2)",
        },
        "instructor_toy_example": instructor_example(),
        "gradcheck_corroboration": gc,
        "fp64": fp64,
        "fp32": fp32,
        "headline": {
            "fp64_min_matching_digits": fp64["min_matching_digits"],
            "fp64_median_matching_digits": fp64["median_matching_digits"],
            "fp32_min_matching_digits": fp32["min_matching_digits"],
            "fp32_median_matching_digits": fp32["median_matching_digits"],
            "fp64_beats_fp32": fp64["min_matching_digits"] > fp32["min_matching_digits"],
            "gradcheck_all_passed": gc["all_passed"],
            "gradcheck_tensors_checked": len(gc["results"]),
        },
    }
    write_artifact("t2_finite_diff.json", payload)

    for label, res in (("fp64", fp64), ("fp32", fp32)):
        print(f"\n=== {label}  (base loss {res['base_loss']:.12f}) ===")
        print(f"{'parameter':<30}{'autograd':>16}{'central diff':>16}"
              f"{'best h':>10}{'digits':>9}")
        for r in res["results"]:
            print(f"{r['param']:<30}{r['autograd_grad']:>16.9f}"
                  f"{r['best_central_difference']:>16.9f}"
                  f"{r['best_h']:>10.0e}{r['best_matching_digits']:>9.2f}")
        print(f"worst agreement: {res['min_matching_digits']:.2f} significant digits")
    print(f"\ntorch.autograd.gradcheck on {len(gc['results'])} tensors of a "
          f"2-layer fp64 model: all passed = {gc['all_passed']}")


if __name__ == "__main__":
    main()
