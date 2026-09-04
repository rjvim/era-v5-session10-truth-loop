"""Shared model, data and utilities for ERA V5 Session 10.

A deliberately small nanoGPT-style decoder so that every experiment in this
repo can be re-run end to end on a single CPU core in a few minutes.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
DATA = ROOT / "data" / "tinyshakespeare.txt"

IGNORE_INDEX = -100  # PyTorch's default "do not score this position" label


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# tokenizer / data
# --------------------------------------------------------------------------
class CharTokenizer:
    """Character-level tokenizer, the nanoGPT default.

    Deliberately trivial: this repo is about the training step, not about
    tokenization, and a tiny vocab keeps the logits tensor small enough to
    print in full.
    """

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_corpus() -> tuple[str, CharTokenizer, torch.Tensor]:
    text = DATA.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return text, tok, ids


def corpus_sha256() -> str:
    return hashlib.sha256(DATA.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("bias", mask, persistent=False)
        self.trace: dict[str, tuple] = {}

    def forward(self, x, trace: bool = False):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        yh = att @ v                                    # (B, H, T, Dh)
        y = yh.transpose(1, 2).contiguous().view(B, T, C)  # heads concatenated
        out = self.c_proj(y)
        if trace:
            self.trace = {
                "attn.input": tuple(x.shape),
                "attn.qkv_fused": tuple(qkv.shape),
                "attn.q_heads": tuple(q.shape),
                "attn.k_heads": tuple(k.shape),
                "attn.v_heads": tuple(v.shape),
                "attn.scores": tuple(att.shape),
                "attn.per_head_output": tuple(yh.shape),
                "attn.heads_merged": tuple(y.shape),
                "attn.output": tuple(out.shape),
            }
        return out


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.trace: dict[str, tuple] = {}

    def forward(self, x, trace: bool = False):
        h = self.c_fc(x)
        a = F.gelu(h)
        out = self.c_proj(a)
        if trace:
            self.trace = {
                "mlp.input": tuple(x.shape),
                "mlp.hidden": tuple(h.shape),
                "mlp.activated": tuple(a.shape),
                "mlp.output": tuple(out.shape),
            }
        return out


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, trace: bool = False):
        x = x + self.attn(self.ln_1(x), trace=trace)
        x = x + self.mlp(self.ln_2(x), trace=trace)
        return x

    def traced_shapes(self) -> dict:
        return {**self.attn.trace, **self.mlp.trace}


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        self.trace: dict[str, tuple] = {}

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, reduction: str = "mean", trace: bool = False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        for blk in self.blocks:
            x = blk(x, trace=trace)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction=reduction,
            )
        if trace:
            self.trace = {
                "input_ids": tuple(idx.shape),
                "positions": tuple(pos.shape),
                "token_embeddings": tuple(tok_emb.shape),
                "position_embeddings": tuple(pos_emb.shape),
                "residual_stream": tuple(x.shape),
                "logits": tuple(logits.shape),
                "logits_flat": (logits.view(-1, logits.size(-1)).shape[0],
                                logits.size(-1)),
                "targets_flat": (targets.reshape(-1).shape[0],) if targets is not None else None,
                "loss": tuple(loss.shape) if loss is not None else None,
            }
        return logits, loss


def build_model(cfg: GPTConfig | None = None, dtype=torch.float32, seed: int = 1337) -> GPT:
    set_seed(seed)
    cfg = cfg or GPTConfig()
    m = GPT(cfg).to(dtype)
    return m


# --------------------------------------------------------------------------
# batches
# --------------------------------------------------------------------------
def uniform_batch(ids: torch.Tensor, B: int, T: int, gen: torch.Generator):
    """Every row is full length T. Valid-token count is identical per row."""
    ix = torch.randint(len(ids) - T - 1, (B,), generator=gen)
    x = torch.stack([ids[i : i + T] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + T] for i in ix])
    return x, y


def n_valid(targets: torch.Tensor) -> int:
    return int((targets != IGNORE_INDEX).sum().item())


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().double().pow(2).sum())
    return math.sqrt(total)


def env_stamp() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_threads": torch.get_num_threads(),
        "git_commit": commit,
        "corpus_sha256": corpus_sha256(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_artifact(name: str, payload: dict) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["_env"] = env_stamp()
    path = ARTIFACTS / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"[artifact] {path.relative_to(ROOT)}")
    return path


def read_artifact(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())
