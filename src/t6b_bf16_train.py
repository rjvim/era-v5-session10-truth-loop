"""Task 6, part 2 - the recommendation, actually trained.

Task 6 measures what each format does to one step's gradients. That is a
snapshot, and a snapshot cannot answer the question the brief really asks:
would you *train* in it. Rounding error that looks harmless on one step can
accumulate over thousands, and the whole reason mixed precision keeps an fp32
master copy is that repeatedly adding small updates into a low-precision
accumulator loses them.

So the recommendation is run rather than asserted. Three arms, identical seed,
identical data stream, evaluated identically in fp32:

  fp32              - the reference. What we are trying to match.
  bf16 + fp32 master - the recommended recipe: compute under autocast in bf16,
                       weights and optimizer state stay fp32.
  pure bf16          - the thing you are not supposed to do: model, gradients
                       and optimizer state all bf16, no master copy.

If the recommendation is right, arm 2 tracks arm 1 closely and arm 3 does not.
If all three land on top of each other, the recommendation is untested folklore
at this scale and the README should say so.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common import (
    FIGURES,
    GPTConfig,
    build_model,
    grad_global_norm,
    load_corpus,
    n_valid,
    set_seed,
    uniform_batch,
    write_artifact,
)

B, T = 16, 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 400
EVAL_EVERY = 10
LR = 3e-4


@torch.no_grad()
def evaluate(model, eval_batches) -> float:
    """Always in fp32, so the three arms are scored on the same ruler."""
    was = next(model.parameters()).dtype
    model.to(torch.float32)
    tot, ntok = 0.0, 0
    for x, y in eval_batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, s = model(x, y, reduction="sum")
        tot += float(s)
        ntok += n_valid(y)
    model.to(was)
    return tot / ntok


def train(mode: str, ids, vocab_size, eval_batches) -> dict:
    set_seed(1337)
    model = build_model(GPTConfig(vocab_size=vocab_size, block_size=T)).to(DEVICE)
    if mode == "pure_bf16":
        model = model.to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(2024)

    steps, losses, gnorms = [], [], []
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            steps.append(step)
            losses.append(evaluate(model, eval_batches))
        if step == STEPS:
            break
        x, y = uniform_batch(ids, B, T, gen)
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad(set_to_none=True)
        if mode == "bf16_autocast":
            # device_type must match where the model actually lives; hardcoding
            # "cpu" here silently disabled autocast on a GPU.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.float().backward()
        else:
            _, loss = model(x, y)
            loss.backward()
        gnorms.append(grad_global_norm(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return {
        "mode": mode,
        "param_dtype": str(next(model.parameters()).dtype),
        "eval_steps": steps,
        "eval_loss": losses,
        "final_eval_loss": losses[-1],
        "grad_norm_median": float(np.median(gnorms)),
    }


def main() -> None:
    _, tok, ids = load_corpus()
    split = int(0.9 * len(ids))
    train_ids, held = ids[:split], ids[split:]
    eg = torch.Generator().manual_seed(1234)
    eval_batches = [uniform_batch(held, B, T, eg) for _ in range(3)]

    arms = {}
    for mode in ("fp32", "bf16_autocast", "pure_bf16"):
        print(f"training {mode} ...", flush=True)
        arms[mode] = train(mode, train_ids, tok.vocab_size, eval_batches)
        print(f"   final held-out loss {arms[mode]['final_eval_loss']:.6f}", flush=True)

    ref = np.array(arms["fp32"]["eval_loss"])
    dev = {}
    for mode in ("bf16_autocast", "pure_bf16"):
        d = np.array(arms[mode]["eval_loss"]) - ref
        dev[mode] = {
            "final_gap_vs_fp32": float(d[-1]),
            "max_abs_gap_vs_fp32": float(np.abs(d).max()),
            "mean_abs_gap_second_half": float(np.abs(d[len(d) // 2:]).mean()),
        }

    payload = {
        "setup": {
            "B": B, "T": T, "steps": STEPS, "lr": LR, "clip_grad_norm": 1.0,
            "device": str(DEVICE),
            "evaluation": "always fp32, so all three arms are scored identically",
            "arms": {
                "fp32": "reference",
                "bf16_autocast": "bf16 compute under autocast, fp32 master weights "
                                 "and fp32 Adam state - the recommended recipe",
                "pure_bf16": "weights, gradients and optimizer state all bf16, "
                             "no master copy",
            },
        },
        "arms": arms,
        "deviation_from_fp32": dev,
        "headline": {
            "fp32_final": arms["fp32"]["final_eval_loss"],
            "bf16_autocast_final": arms["bf16_autocast"]["final_eval_loss"],
            "pure_bf16_final": arms["pure_bf16"]["final_eval_loss"],
            "bf16_autocast_max_gap": dev["bf16_autocast"]["max_abs_gap_vs_fp32"],
            "pure_bf16_max_gap": dev["pure_bf16"]["max_abs_gap_vs_fp32"],
            "master_copy_helps": (dev["bf16_autocast"]["max_abs_gap_vs_fp32"]
                                  < dev["pure_bf16"]["max_abs_gap_vs_fp32"]),
            "penalty_ratio": (dev["pure_bf16"]["max_abs_gap_vs_fp32"]
                              / max(dev["bf16_autocast"]["max_abs_gap_vs_fp32"], 1e-12)),
        },
    }
    write_artifact("t6b_bf16_train.json", payload)
    plot(payload)


def plot(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    a = payload["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    for mode, style in (("fp32", "-"), ("bf16_autocast", "--"), ("pure_bf16", ":")):
        ax.plot(a[mode]["eval_steps"], a[mode]["eval_loss"], style, lw=2, label=mode)
    ax.set_title("Held-out loss, evaluated in fp32 for all three arms")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("token-weighted loss")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    ax = axes[1]
    ref = np.array(a["fp32"]["eval_loss"])
    for mode, style in (("bf16_autocast", "--"), ("pure_bf16", ":")):
        ax.plot(a[mode]["eval_steps"], np.array(a[mode]["eval_loss"]) - ref,
                style, lw=2, label=mode)
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("Deviation from the fp32 reference")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("loss difference")
    ax.legend(fontsize=9); ax.grid(alpha=.3)

    fig.tight_layout()
    out = FIGURES / "t6b_bf16_train.png"
    fig.savefig(out, dpi=130)
    print(f"[figure] {out}")


if __name__ == "__main__":
    main()
