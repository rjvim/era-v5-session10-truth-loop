"""Task 1 - print every tensor shape in one training step and say what each
dimension means.

"Every tensor" is taken literally: parameters, activations, gradients and
optimizer state. Each axis carries its own gloss rather than one sentence per
tensor, because the point of the exercise is to be able to say what axis 2 of
the attention score tensor is without pausing.

Two deliberate choices:

* Axis meanings are *declared* symbolically (B, T, C, 3C, ...) and only then
  resolved to numbers, with an assertion that the declaration matches the real
  tensor. Inferring meaning from the number itself is how you end up labelling
  the input axis of a weight matrix as "sequence position" because block_size
  happened to equal n_embd.
* The config is chosen so B=8, T=64, V=65, head_dim=32, C=128, 3C=384 and
  4C=512 are all distinct, so no printed shape is ambiguous to a reader either.
"""

from __future__ import annotations

import re
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

B, T = 8, 64

SYMBOL = {
    "B": "B - independent sequences in this micro-batch",
    "T": "T - sequence positions (context length used by this batch)",
    "T_max": "block_size - the largest position the model was built for",
    "Tk": "T (key positions) - what each query position is allowed to attend to",
    "C": "C = n_embd - residual-stream channels",
    "3C": "3*n_embd - q, k and v projections fused into a single matmul",
    "4C": "4*n_embd - MLP expansion width",
    "V": "V = vocab_size - one slot per token id",
    "H": "H = n_head - attention heads, computed in parallel with no interaction",
    "Dh": "head_dim = n_embd/n_head - channels owned by a single head",
    "BT": "B*T - rows flattened so cross_entropy sees one prediction per row",
}

# nn.Linear stores weight as (out_features, in_features).
PARAM_AXES = [
    (r"^wte\.weight$", ["V", "C"]),
    (r"^wpe\.weight$", ["T_max", "C"]),
    (r"ln_\d\.(weight|bias)$", ["C"]),
    (r"^ln_f\.(weight|bias)$", ["C"]),
    (r"attn\.c_attn\.weight$", ["3C", "C"]),
    (r"attn\.c_attn\.bias$", ["3C"]),
    (r"attn\.c_proj\.weight$", ["C", "C"]),
    (r"attn\.c_proj\.bias$", ["C"]),
    (r"mlp\.c_fc\.weight$", ["4C", "C"]),
    (r"mlp\.c_fc\.bias$", ["4C"]),
    (r"mlp\.c_proj\.weight$", ["C", "4C"]),
    (r"mlp\.c_proj\.bias$", ["C"]),
    (r"^lm_head\.weight$", ["V", "C"]),
]

ACT_AXES = {
    "input_ids": ["B", "T"],
    "positions": ["T"],
    "token_embeddings": ["B", "T", "C"],
    "position_embeddings": ["T", "C"],
    "residual_stream": ["B", "T", "C"],
    "logits": ["B", "T", "V"],
    "logits_flat": ["BT", "V"],
    "targets_flat": ["BT"],
    "loss": [],
    "attn.input": ["B", "T", "C"],
    "attn.qkv_fused": ["B", "T", "3C"],
    "attn.q_heads": ["B", "H", "T", "Dh"],
    "attn.k_heads": ["B", "H", "T", "Dh"],
    "attn.v_heads": ["B", "H", "T", "Dh"],
    "attn.scores": ["B", "H", "T", "Tk"],
    "attn.per_head_output": ["B", "H", "T", "Dh"],
    "attn.heads_merged": ["B", "T", "C"],
    "attn.output": ["B", "T", "C"],
    "mlp.input": ["B", "T", "C"],
    "mlp.hidden": ["B", "T", "4C"],
    "mlp.activated": ["B", "T", "4C"],
    "mlp.output": ["B", "T", "C"],
}


def symbols_for_param(name: str) -> list[str]:
    for pat, axes in PARAM_AXES:
        if re.search(pat, name):
            return axes
    raise KeyError(f"no declared axis meaning for parameter {name!r}")


def resolve(symbols: list[str], shape: tuple, cfg: GPTConfig) -> list[str]:
    sizes = {
        "B": B, "T": T, "T_max": cfg.block_size, "Tk": T,
        "C": cfg.n_embd, "3C": 3 * cfg.n_embd, "4C": 4 * cfg.n_embd,
        "V": cfg.vocab_size, "H": cfg.n_head,
        "Dh": cfg.n_embd // cfg.n_head, "BT": B * T,
    }
    assert len(symbols) == len(shape), f"declared {symbols} for real shape {shape}"
    out = []
    for sym, n in zip(symbols, shape):
        assert sizes[sym] == n, f"declared {sym}={sizes[sym]} but the axis is {n}"
        out.append(f"{n}: {SYMBOL[sym]}")
    return out


def main() -> None:
    set_seed(1337)
    _, tok, ids = load_corpus()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=T)
    model = build_model(cfg)
    gen = torch.Generator().manual_seed(0)
    x, y = uniform_batch(ids, B, T, gen)

    # ---- 1. parameters ---------------------------------------------------
    params = [
        {
            "name": name,
            "shape": list(p.shape),
            "numel": p.numel(),
            "dtype": str(p.dtype),
            "requires_grad": p.requires_grad,
            "axes": resolve(symbols_for_param(name), tuple(p.shape), cfg),
        }
        for name, p in model.named_parameters()
    ]

    # Weight tying: wte.weight and lm_head.weight are the same storage, so
    # named_parameters() yields it once. Recorded rather than left implicit.
    tied = model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()

    # ---- 2. activations from one traced forward --------------------------
    logits, loss = model(x, y, trace=True)
    acts = dict(model.trace)
    acts.update(model.blocks[0].traced_shapes())
    # Every block is traced, not just the first. They turn out to be identical,
    # which is a claim worth checking rather than assuming: a shape that drifted
    # between layers would be a real bug and this is where it would show.
    per_block = {i: blk.traced_shapes() for i, blk in enumerate(model.blocks)}
    blocks_identical = all(per_block[i] == per_block[0] for i in per_block)
    activations = []
    for k, v in acts.items():
        shape = tuple(v) if v else ()
        activations.append(
            {
                "name": k,
                "shape": list(shape),
                "axes": resolve(ACT_AXES[k], shape, cfg)
                or ["scalar: the loss is 0-dimensional, one number for the whole batch"],
            }
        )

    # ---- 3. gradients ----------------------------------------------------
    loss.backward()
    grads = [
        {
            "name": n,
            "grad_shape": list(p.grad.shape),
            "matches_param_shape": list(p.grad.shape) == list(p.shape),
        }
        for n, p in model.named_parameters()
        if p.grad is not None
    ]

    # ---- 4. optimizer state ----------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    state = [
        {
            "name": n,
            "exp_avg_shape": list(opt.state[p]["exp_avg"].shape),
            "exp_avg_sq_shape": list(opt.state[p]["exp_avg_sq"].shape),
        }
        for n, p in model.named_parameters()
    ]

    # ---- 5. per-layer parameter breakdown --------------------------------
    by_layer = {}
    for name, p in model.named_parameters():
        layer = name.split(".")[1] if name.startswith("blocks.") else "top-level"
        key = f"block {layer}" if name.startswith("blocks.") else layer
        e = by_layer.setdefault(key, {"n_params": 0, "n_tensors": 0, "tensors": []})
        e["n_params"] += p.numel()
        e["n_tensors"] += 1
        e["tensors"].append({"name": name, "shape": list(p.shape),
                             "numel": p.numel(), "trainable": p.requires_grad})

    n_params = model.n_params()
    payload = {
        "config": {
            "B": B, "T": T,
            "vocab_size": cfg.vocab_size,
            "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
            "head_dim": cfg.n_embd // cfg.n_head,
            "block_size": cfg.block_size,
            "tie_weights": cfg.tie_weights,
        },
        "totals": {
            "n_params": n_params,
            "n_param_tensors": len(params),
            "n_grad_tensors": len(grads),
            "n_optimizer_state_tensors": 2 * len(state),
            "n_traced_activations": len(activations),
            "all_blocks_have_identical_activation_shapes": blocks_identical,
            "n_blocks_traced": len(per_block),
            "lm_head_shares_storage_with_wte": tied,
            "all_grads_match_param_shape": all(g["matches_param_shape"] for g in grads),
            "all_optimizer_state_matches_param_shape": all(
                s["exp_avg_shape"] == p["shape"] and s["exp_avg_sq_shape"] == p["shape"]
                for s, p in zip(state, params)
            ),
            "distinct_axis_sizes": sorted(
                {B, T, cfg.vocab_size, cfg.n_embd, 3 * cfg.n_embd, 4 * cfg.n_embd,
                 cfg.n_head, cfg.n_embd // cfg.n_head}
            ),
        },
        "memory_per_param_bytes": {
            "note": "mixed-precision recipe from the session: bf16 weight + bf16 grad "
                    "+ fp32 master copy + fp32 Adam m and v = 16 bytes/param",
            "bf16_weight": 2,
            "bf16_grad": 2,
            "fp32_master_weight": 4,
            "adam_exp_avg_fp32": 4,
            "adam_exp_avg_sq_fp32": 4,
            "total": 16,
            "this_model_state_bytes": 16 * n_params,
            "this_model_state_MiB": round(16 * n_params / 2**20, 3),
            "extrapolated_checkpoint_TiB_at_120B": round(16 * 120e9 / 2**40, 3),
        },
        "parameters_by_layer": by_layer,
        "activations_per_block": {str(k): v for k, v in per_block.items()},
        "parameters": params,
        "activations": activations,
        "gradients": grads,
        "optimizer_state": state,
    }
    write_artifact("t1_shapes.json", payload)

    print(f"\n=== PARAMETERS ({len(params)} tensors, {n_params:,} scalars) ===")
    for p in params:
        print(f"{p['name']:<32} {str(tuple(p['shape'])):<14} {p['numel']:>8,}")
        for a in p["axes"]:
            print(f"      {a}")
    print(f"\n=== ACTIVATIONS in one forward (B={B}, T={T}) ===")
    for a in activations:
        print(f"{a['name']:<24} {str(tuple(a['shape'])) if a['shape'] else '()'}")
        for g in a["axes"]:
            print(f"      {g}")
    print("\n=== PARAMETERS BY LAYER ===")
    for k, v in by_layer.items():
        print(f"{k:<16}{v['n_tensors']:>3} tensors{v['n_params']:>10,} params  "
              f"all trainable: {all(t['trainable'] for t in v['tensors'])}")
    print(f"\nall {len(per_block)} blocks have identical activation shapes -> "
          f"{blocks_identical}")
    print(f"weight tying: lm_head.weight IS wte.weight -> {tied}")
    print(f"grad shape == param shape for all {len(grads)} tensors -> "
          f"{payload['totals']['all_grads_match_param_shape']}")
    print(f"Adam keeps 2 state tensors per parameter tensor -> "
          f"{payload['totals']['n_optimizer_state_tensors']} state tensors")


if __name__ == "__main__":
    main()
