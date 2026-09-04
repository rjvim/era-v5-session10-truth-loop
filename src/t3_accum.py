"""Task 3 - break gradient accumulation on purpose.

The bug the session described: when you accumulate over micro-batches you must
weight each micro-batch by the number of tokens it actually scored, not treat
every micro-batch as equal.

Publicly documented by Unsloth in October 2024 and fixed in Hugging Face's
Trainer around the same time. It is not purely historical, though: the same
distortion arises across data-parallel ranks, and as of 2026 it is reported as
still live in several widely used post-training libraries.

    correct : sum_i (sum of per-token losses in i) / sum_i n_valid_i
    broken  : mean_i ( mean of per-token losses in i )

These are identical when every micro-batch scores the same number of tokens,
which is exactly why nobody caught it: pretraining on packed fixed-length
sequences never triggers it. It bites when n_valid differs per micro-batch -
padding, or post-training where the prompt tokens are masked out of the loss.

Four arms are run, all from the same seed and the same data stream:

  ragged/correct  - token-weighted, unequal micro-batch lengths
  ragged/broken   - mean of means, unequal micro-batch lengths
  uniform/correct - token-weighted, every micro-batch the same length
  uniform/broken  - mean of means, every micro-batch the same length

The uniform pair is the control. If the uniform curves do not sit on top of
each other, the ragged gap is not evidence of anything and the experiment is
measuring some other difference between the two code paths.

Separately, before any training, both reductions are compared at step 0 against
a *reference* gradient computed in one shot over the concatenated batch with no
accumulation at all. That reference is the ground truth: it is what you would
get with enough memory to skip accumulation entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common import (
    FIGURES,
    IGNORE_INDEX,
    GPTConfig,
    build_model,
    load_corpus,
    n_valid,
    set_seed,
    write_artifact,
)

T = 64
MICRO_B = 4          # rows per micro-batch
ACCUM = 4            # micro-batches per optimizer step
STEPS = 400
EVAL_EVERY = 10
LR = 3e-4


# --------------------------------------------------------------------------
# data: micro-batches whose valid-token counts differ a lot
# --------------------------------------------------------------------------
SIGMA = 1.3   # length skew; 0 would make every row the same length


def skewed_lengths(rng: np.random.Generator, n: int, sigma: float = SIGMA) -> list[int]:
    """Lengths skewed short with a long tail, the shape of real SFT data."""
    raw = rng.lognormal(mean=2.6, sigma=sigma, size=n)
    return [int(np.clip(round(v), 2, T)) for v in raw]


def make_micro_batches(ids, rng, torch_gen, ragged: bool, sigma: float = SIGMA):
    """One optimizer step's worth of micro-batches, on identical underlying text.

    ragged=False uses the same text but scores every position, so the two
    regimes differ only in the masking, not in the tokens seen.
    """
    batches = []
    for _ in range(ACCUM):
        lengths = skewed_lengths(rng, MICRO_B, sigma) if ragged else [T] * MICRO_B
        ix = torch.randint(len(ids) - T - 1, (MICRO_B,), generator=torch_gen)
        x = torch.zeros(MICRO_B, T, dtype=torch.long)
        y = torch.full((MICRO_B, T), IGNORE_INDEX, dtype=torch.long)
        for r, (i, L) in enumerate(zip(ix, lengths)):
            x[r] = ids[i : i + T]
            y[r, :L] = ids[i + 1 : i + 1 + L]
        batches.append((x, y))
    return batches


# --------------------------------------------------------------------------
# the two reductions
# --------------------------------------------------------------------------
def accumulate(model, batches, mode: str) -> tuple[float, int]:
    """Run one optimizer step's worth of accumulation. Leaves grads populated."""
    model.zero_grad(set_to_none=True)
    total_valid = sum(n_valid(y) for _, y in batches)
    reported = 0.0
    for x, y in batches:
        if mode == "correct":
            # sum of per-token losses, divided once by the step's total tokens
            _, loss_sum = model(x, y, reduction="sum")
            scaled = loss_sum / total_valid
        elif mode == "broken":
            # mean over this micro-batch, then averaged across micro-batches
            _, loss_mean = model(x, y, reduction="mean")
            scaled = loss_mean / len(batches)
        else:
            raise ValueError(mode)
        scaled.backward()
        reported += float(scaled.detach())
    return reported, total_valid


def reference_grad(model, batches):
    """Ground truth: one forward over everything concatenated, no accumulation."""
    model.zero_grad(set_to_none=True)
    x = torch.cat([b[0] for b in batches], dim=0)
    y = torch.cat([b[1] for b in batches], dim=0)
    _, loss = model(x, y, reduction="mean")   # mean over all valid tokens at once
    loss.backward()
    g = torch.cat([p.grad.detach().reshape(-1).clone() for p in model.parameters()])
    return g, float(loss.detach())


def flat_grad(model):
    return torch.cat([p.grad.detach().reshape(-1).clone() for p in model.parameters()])


# --------------------------------------------------------------------------
# step-0 gradient comparison
# --------------------------------------------------------------------------
def step0_comparison(ids) -> dict:
    out = {}
    for regime in ("ragged", "uniform"):
        set_seed(1337)
        _, tok, _ = load_corpus()
        model = build_model(GPTConfig(vocab_size=tok.vocab_size, block_size=T),
                            dtype=torch.float64)
        rng = np.random.default_rng(7)
        tg = torch.Generator().manual_seed(7)
        batches = make_micro_batches(ids, rng, tg, ragged=(regime == "ragged"))
        counts = [n_valid(y) for _, y in batches]

        ref, ref_loss = reference_grad(model, batches)
        entry = {
            "valid_tokens_per_micro_batch": counts,
            "max_over_min_ratio": max(counts) / min(counts),
            "reference_loss": ref_loss,
        }
        for mode in ("correct", "broken"):
            accumulate(model, batches, mode)
            g = flat_grad(model)
            diff = g - ref
            entry[mode] = {
                "rel_l2_error_vs_reference": float(diff.norm() / ref.norm()),
                "cosine_similarity_vs_reference": float(
                    torch.dot(g, ref) / (g.norm() * ref.norm())
                ),
                "max_abs_elementwise_error": float(diff.abs().max()),
                "grad_norm": float(g.norm()),
                "reference_grad_norm": float(ref.norm()),
            }
        out[regime] = entry
    return out


def skew_sweep(ids) -> list[dict]:
    """How big is the bug as a function of how unequal the micro-batches are?

    sigma=0 makes every row the same length, which reproduces the uniform
    control from inside the ragged code path.
    """
    _, tok, _ = load_corpus()
    rows = []
    for sigma in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        set_seed(1337)
        model = build_model(GPTConfig(vocab_size=tok.vocab_size, block_size=T),
                            dtype=torch.float64)
        rng = np.random.default_rng(7)
        tg = torch.Generator().manual_seed(7)
        batches = make_micro_batches(ids, rng, tg, ragged=True, sigma=sigma)
        counts = [n_valid(y) for _, y in batches]
        ref, _ = reference_grad(model, batches)
        accumulate(model, batches, "broken")
        g = flat_grad(model)
        rows.append(
            {
                "sigma": sigma,
                "valid_tokens_per_micro_batch": counts,
                "max_over_min_ratio": max(counts) / min(counts),
                "coefficient_of_variation": float(np.std(counts) / np.mean(counts)),
                "broken_rel_l2_error": float((g - ref).norm() / ref.norm()),
                "broken_cosine": float(torch.dot(g, ref) / (g.norm() * ref.norm())),
            }
        )
    return rows


# --------------------------------------------------------------------------
# training arms
# --------------------------------------------------------------------------
def build_eval_set(ids, ragged: bool):
    rng = np.random.default_rng(99)
    tg = torch.Generator().manual_seed(99)
    return [make_micro_batches(ids, rng, tg, ragged)[0] for _ in range(16)]


@torch.no_grad()
def evaluate(model, eval_batches) -> float:
    """Token-weighted held-out loss. Deliberately not the training reduction."""
    tot_loss, tot_tok = 0.0, 0
    for x, y in eval_batches:
        _, s = model(x, y, reduction="sum")
        tot_loss += float(s)
        tot_tok += n_valid(y)
    return tot_loss / tot_tok


def train_arm(ids, vocab_size, ragged: bool, mode: str, eval_batches) -> dict:
    set_seed(1337)
    model = build_model(GPTConfig(vocab_size=vocab_size, block_size=T))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    rng = np.random.default_rng(2024)
    tg = torch.Generator().manual_seed(2024)

    steps, evals, train_losses, tokens = [], [], [], []
    seen = 0
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            steps.append(step)
            evals.append(evaluate(model, eval_batches))
            tokens.append(seen)
        if step == STEPS:
            break
        batches = make_micro_batches(ids, rng, tg, ragged)
        reported, nv = accumulate(model, batches, mode)
        seen += nv
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_losses.append(reported)
    return {
        "regime": "ragged" if ragged else "uniform",
        "mode": mode,
        "eval_steps": steps,
        "eval_loss": evals,
        "tokens_seen_at_eval": tokens,
        "final_eval_loss": evals[-1],
        "reported_train_loss": train_losses,
    }


def main() -> None:
    _, tok, ids = load_corpus()

    print("step-0 gradient comparison against a no-accumulation reference...")
    step0 = step0_comparison(ids)
    for regime, e in step0.items():
        print(f"  {regime}: valid tokens per micro-batch {e['valid_tokens_per_micro_batch']}")
        for mode in ("correct", "broken"):
            print(f"    {mode:<8} rel L2 err {e[mode]['rel_l2_error_vs_reference']:.3e}"
                  f"   cos {e[mode]['cosine_similarity_vs_reference']:.9f}")

    print("skew sweep ...")
    sweep = skew_sweep(ids)
    for r in sweep:
        print(f"  sigma={r['sigma']:.1f}  CV={r['coefficient_of_variation']:.3f}"
              f"  rel L2 err={r['broken_rel_l2_error']:.3e}")

    arms = {}
    for ragged in (True, False):
        eval_batches = build_eval_set(ids, ragged)
        for mode in ("correct", "broken"):
            key = f"{'ragged' if ragged else 'uniform'}/{mode}"
            print(f"training {key} ...")
            arms[key] = train_arm(ids, tok.vocab_size, ragged, mode, eval_batches)
            print(f"   final held-out loss {arms[key]['final_eval_loss']:.6f}")

    ragged_gap = [b - a for a, b in zip(arms["ragged/correct"]["eval_loss"],
                                        arms["ragged/broken"]["eval_loss"])]
    uniform_gap = [b - a for a, b in zip(arms["uniform/correct"]["eval_loss"],
                                         arms["uniform/broken"]["eval_loss"])]

    payload = {
        "setup": {
            "T": T, "micro_batch_rows": MICRO_B, "grad_accum_steps": ACCUM,
            "steps": STEPS, "lr": LR, "eval_every": EVAL_EVERY,
            "clip_grad_norm": 1.0,
            "correct_reduction": "sum(per-token losses) / sum(n_valid) over the step",
            "broken_reduction": "mean_i(mean of per-token losses in micro-batch i)",
            "eval_metric": "token-weighted held-out loss on a fixed batch set",
        },
        "step0_gradient_comparison": step0,
        "skew_sweep": sweep,
        "arms": arms,
        "gap_ragged_broken_minus_correct": ragged_gap,
        "gap_uniform_broken_minus_correct": uniform_gap,
        "headline": {
            "ragged_final_correct": arms["ragged/correct"]["final_eval_loss"],
            "ragged_final_broken": arms["ragged/broken"]["final_eval_loss"],
            "ragged_final_gap": ragged_gap[-1],
            "uniform_final_correct": arms["uniform/correct"]["final_eval_loss"],
            "uniform_final_broken": arms["uniform/broken"]["final_eval_loss"],
            "uniform_final_gap": uniform_gap[-1],
            "max_abs_uniform_gap": max(abs(g) for g in uniform_gap),
            "ragged_mean_gap_last_half": float(np.mean(ragged_gap[len(ragged_gap)//2:])),
            "uniform_mean_gap_last_half": float(np.mean(uniform_gap[len(uniform_gap)//2:])),
            "ragged_gap_positive_fraction": float(
                np.mean([g > 0 for g in ragged_gap[1:]])
            ),
            "control_holds": max(abs(g) for g in uniform_gap) < abs(ragged_gap[-1]) / 10,
            "broken_is_worse_when_ragged": ragged_gap[-1] > 0,
            "step0_broken_rel_l2_error_ragged":
                step0["ragged"]["broken"]["rel_l2_error_vs_reference"],
            "step0_broken_rel_l2_error_uniform":
                step0["uniform"]["broken"]["rel_l2_error_vs_reference"],
            "sweep_error_monotone_in_imbalance": all(
                b["broken_rel_l2_error"] >= a["broken_rel_l2_error"] - 1e-12
                for a, b in zip(
                    sorted(sweep, key=lambda r: r["coefficient_of_variation"]),
                    sorted(sweep, key=lambda r: r["coefficient_of_variation"])[1:],
                )
            ),
            "sweep_note": "the error is monotone in the realised imbalance "
                          "(coefficient of variation of valid tokens), not in sigma: "
                          "past sigma~1.2 the lognormal tail is clipped at T so the "
                          "realised imbalance stops growing",
        },
    }
    write_artifact("t3_accum.json", payload)
    plot(payload)


def plot(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = payload["arms"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))

    ax = axes[0]
    ax.plot(a["ragged/correct"]["eval_steps"], a["ragged/correct"]["eval_loss"],
            label="correct (token-weighted)", lw=2)
    ax.plot(a["ragged/broken"]["eval_steps"], a["ragged/broken"]["eval_loss"],
            label="broken (mean of means)", lw=2, ls="--")
    ax.set_title("Unequal micro-batch lengths")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("held-out loss (token-weighted)")
    ax.legend(); ax.grid(alpha=.3)
    ins = ax.inset_axes([0.42, 0.42, 0.55, 0.45])
    k = len(a["ragged/correct"]["eval_steps"]) // 2
    ins.plot(a["ragged/correct"]["eval_steps"][k:], a["ragged/correct"]["eval_loss"][k:], lw=1.6)
    ins.plot(a["ragged/broken"]["eval_steps"][k:], a["ragged/broken"]["eval_loss"][k:],
             lw=1.6, ls="--")
    ins.set_title("second half, zoomed", fontsize=8)
    ins.tick_params(labelsize=7); ins.grid(alpha=.3)

    ax = axes[1]
    ax.plot(a["uniform/correct"]["eval_steps"], a["uniform/correct"]["eval_loss"],
            label="correct", lw=2)
    ax.plot(a["uniform/broken"]["eval_steps"], a["uniform/broken"]["eval_loss"],
            label="broken", lw=2, ls="--")
    ax.set_title("Control: equal micro-batch lengths")
    ax.set_xlabel("optimizer step"); ax.legend(); ax.grid(alpha=.3)

    ax = axes[2]
    ax.plot(a["ragged/correct"]["eval_steps"],
            payload["gap_ragged_broken_minus_correct"], label="ragged", lw=2)
    ax.plot(a["uniform/correct"]["eval_steps"],
            payload["gap_uniform_broken_minus_correct"], label="uniform (control)",
            lw=2, ls="--")
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("Gap: broken minus correct")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("loss difference")
    ax.legend(); ax.grid(alpha=.3)

    ax = axes[3]
    sw = payload["skew_sweep"]
    ax.plot([r["coefficient_of_variation"] for r in sw],
            [r["broken_rel_l2_error"] for r in sw], "o-", lw=2)
    ax.set_title("Step-0 gradient error vs length imbalance")
    ax.set_xlabel("coefficient of variation of valid tokens per micro-batch")
    ax.set_ylabel("relative L2 error of broken gradient")
    ax.grid(alpha=.3)

    fig.tight_layout()
    out = FIGURES / "t3_accum.png"
    fig.savefig(out, dpi=130)
    print(f"[figure] {out}")


if __name__ == "__main__":
    main()
