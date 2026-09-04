"""Task 4 - log the grad norm at every step, then find one step where it moved
before the loss did.

Three arms, same seed, same data stream:

  clean/noclip    - ordinary Shakespeare, no gradient clipping
  inject/noclip   - a two-step window of out-of-distribution text at step 250
  inject/clip1.0  - identical, with clip_grad_norm_(..., 1.0)

The injected text is Sanskrit in Harvard-Kyoto transliteration. That choice is
not decoration. HK is pure ASCII, and every character in the injection corpus
is already in the Shakespeare character vocabulary (asserted in the tests), so
the tokenizer cannot tell the two apart at all - which is the session's own
example of Sanskrit leaking into a Hindi run because a script-based filter sees
Devanagari in both cases. Here the model sees familiar characters in a wholly
unfamiliar distribution, and the only thing that notices is the gradient.

WHAT IS MEASURED, DECIDED BEFORE LOOKING AT ANY DATA
----------------------------------------------------
At step k the model state is theta_k. We record, in this order:

  eval_loss[k]  - token-weighted loss on a fixed clean held-out set, at theta_k
  grad_norm[k]  - global L2 norm of the gradient from batch k, before clipping
  then the optimizer produces theta_{k+1}

so grad_norm[k] can only influence eval_loss at k+1 or later. A one-step lead is
therefore causal by construction and proves nothing on its own. The claim only
has content if the observed lag exceeds one step - if the norm is visibly
anomalous while the loss still looks fine for several steps afterwards. Both the
lag histogram and the cross-correlation argmax are reported so this can be
checked rather than asserted.

Anomaly definitions, fixed in advance:

  * trailing window W = 50 steps, excluding the current step
  * robust z = (x - median) / (1.4826 * MAD), threshold Z = 4.0
  * the grad-norm series is used raw (it is roughly stationary)
  * the loss series is used as its step-over-step change, because eval loss is a
    smoothly declining trend and a spike detector run on the level would fire on
    ordinary drift. Damage shows up as the loss going UP, i.e. a positive change.
  * "the norm moved before the loss did" at step k iff
        z_grad[k] > Z  and  z_dloss[k] <= Z  and  z_dloss[k+m] > Z
    for some m in 1..10; the lag is the smallest such m.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common import (
    FIGURES,
    read_artifact,
    ROOT,
    CharTokenizer,
    GPTConfig,
    build_model,
    grad_global_norm,
    load_corpus,
    n_valid,
    set_seed,
    uniform_batch,
    write_artifact,
)

B, T = 8, 64
STEPS = 400
LR = 3e-4
INJECT_AT = (200, 201, 202, 203)
WINDOW = 50
Z_THRESH = 4.0
MAX_LAG = 10

SANSKRIT = ROOT / "data" / "sanskrit_hk.txt"


# --------------------------------------------------------------------------
def robust_z(series: np.ndarray, window: int) -> np.ndarray:
    """Trailing robust z-score. Positions without a full window are NaN."""
    z = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        w = series[i - window : i]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        scale = 1.4826 * mad
        if scale <= 0:
            continue
        z[i] = (series[i] - med) / scale
    return z


def find_lead_events(z_grad, z_dloss, z=Z_THRESH, max_lag=MAX_LAG) -> list[dict]:
    events = []
    for k in range(len(z_grad) - 1):
        if not (np.isfinite(z_grad[k]) and z_grad[k] > z):
            continue
        if not (np.isfinite(z_dloss[k]) and z_dloss[k] <= z):
            continue
        for m in range(1, max_lag + 1):
            if k + m >= len(z_dloss):
                break
            if np.isfinite(z_dloss[k + m]) and z_dloss[k + m] > z:
                events.append(
                    {
                        "step": int(k),
                        "lag": int(m),
                        "z_grad_at_step": float(z_grad[k]),
                        "z_dloss_at_step": float(z_dloss[k]),
                        "z_dloss_at_step_plus_lag": float(z_dloss[k + m]),
                    }
                )
                break
    return events


def cross_correlation(z_grad, z_dloss, max_lag=MAX_LAG) -> list[dict]:
    """Correlation between the norm series and the loss-change series at each lag."""
    out = []
    for lag in range(0, max_lag + 1):
        a = z_grad[: len(z_grad) - lag] if lag else z_grad
        b = z_dloss[lag:]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 30:
            continue
        r = float(np.corrcoef(a[mask], b[mask])[0, 1])
        out.append({"lag": lag, "pearson_r": r, "n": int(mask.sum())})
    return out


# --------------------------------------------------------------------------
def sanskrit_batches(tok: CharTokenizer, n: int, gen: torch.Generator):
    text = SANSKRIT.read_text(encoding="utf-8")
    missing = sorted(set(text) - set(tok.chars))
    assert not missing, f"injection corpus has out-of-vocab characters: {missing}"
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    return [uniform_batch(ids, B, T, gen) for _ in range(n)], len(text)


@torch.no_grad()
def evaluate(model, batches) -> float:
    tot, ntok = 0.0, 0
    for x, y in batches:
        _, s = model(x, y, reduction="sum")
        tot += float(s)
        ntok += n_valid(y)
    return tot / ntok


def run_arm(train_ids, eval_batches, vocab_size, inject: bool, clip: float | None,
            tok) -> dict:
    set_seed(1337)
    model = build_model(GPTConfig(vocab_size=vocab_size, block_size=T))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))
    tg = torch.Generator().manual_seed(4242)
    inj_gen = torch.Generator().manual_seed(11)
    inj_batches, _ = sanskrit_batches(tok, len(INJECT_AT), inj_gen)

    rec = {k: [] for k in ("step", "eval_loss", "train_loss", "grad_norm",
                           "grad_norm_postclip", "injected")}
    for step in range(STEPS):
        ev = evaluate(model, eval_batches)          # loss at theta_k
        is_inj = inject and step in INJECT_AT
        if is_inj:
            x, y = inj_batches[INJECT_AT.index(step)]
        else:
            x, y = uniform_batch(train_ids, B, T, tg)

        model.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        gn = grad_global_norm(model)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        gn_post = grad_global_norm(model)
        opt.step()                                   # -> theta_{k+1}

        rec["step"].append(step)
        rec["eval_loss"].append(ev)
        rec["train_loss"].append(float(loss.detach()))
        rec["grad_norm"].append(gn)
        rec["grad_norm_postclip"].append(gn_post)
        rec["injected"].append(bool(is_inj))

    ev_loss = np.array(rec["eval_loss"])
    gnorm = np.array(rec["grad_norm"])
    d_loss = np.concatenate([[np.nan], np.diff(ev_loss)])

    z_grad = robust_z(gnorm, WINDOW)
    z_dloss = robust_z(d_loss, WINDOW)
    events = find_lead_events(z_grad, z_dloss)
    xcorr = cross_correlation(z_grad, z_dloss)

    return {
        "inject": inject,
        "clip": clip,
        "log": rec,
        "z_grad": [None if not np.isfinite(v) else float(v) for v in z_grad],
        "z_dloss": [None if not np.isfinite(v) else float(v) for v in z_dloss],
        "lead_events": events,
        "n_lead_events": len(events),
        "lags": sorted({e["lag"] for e in events}),
        "cross_correlation": xcorr,
        "xcorr_argmax_lag": (max(xcorr, key=lambda r: r["pearson_r"])["lag"]
                             if xcorr else None),
        "final_eval_loss": float(ev_loss[-1]),
        "max_grad_norm": float(gnorm.max()),
        "max_grad_norm_step": int(gnorm.argmax()),
        "median_grad_norm": float(np.median(gnorm)),
    }


def threshold_sweep(arms: dict) -> list[dict]:
    """Sensitivity of the event count to the spike threshold.

    Reported for every threshold rather than picking the one that produces the
    nicest number. Z = 4.0 is the pre-registered primary.
    """
    rows = []
    for z in (3.0, 3.5, 4.0, 4.5, 5.0):
        row = {"z_threshold": z, "primary": z == Z_THRESH}
        for name, a in arms.items():
            zg = np.array([np.nan if v is None else v for v in a["z_grad"]])
            zd = np.array([np.nan if v is None else v for v in a["z_dloss"]])
            ev = find_lead_events(zg, zd, z=z)
            row[name] = {"n_events": len(ev), "lags": sorted({e["lag"] for e in ev})}
        rows.append(row)
    return rows


def excursion(inj_arm: dict, clean_arm: dict, k0: int, horizon: int = 40) -> dict:
    """How far, and how long, the injected run's held-out loss sits above clean."""
    d = [inj_arm["log"]["eval_loss"][k0 + i] - clean_arm["log"]["eval_loss"][k0 + i]
         for i in range(horizon)]
    peak_i = int(np.argmax(d))
    back = next((i for i in range(peak_i, horizon) if d[i] < 0.1 * d[peak_i]), None)
    return {
        "peak_excess_loss_vs_clean": float(d[peak_i]),
        "steps_from_injection_to_peak": peak_i,
        "steps_to_recover_to_10pct_of_peak": back,
        "excess_loss_by_step": {str(k0 + i): float(v) for i, v in enumerate(d)},
    }


def describe_event(arm: dict, e: dict) -> dict:
    """Full context around one lead event, so the claim can be audited by hand."""
    k, m = e["step"], e["lag"]
    lo, hi = k - 3, k + m + 3
    log = arm["log"]
    return {
        "step": k,
        "lag_steps": m,
        "grad_norm_at_step": log["grad_norm"][k],
        "median_grad_norm_in_trailing_window": float(
            np.median(log["grad_norm"][k - WINDOW : k])
        ),
        "grad_norm_multiple_of_trailing_median": log["grad_norm"][k]
        / float(np.median(log["grad_norm"][k - WINDOW : k])),
        "z_grad": e["z_grad_at_step"],
        "z_dloss_same_step": e["z_dloss_at_step"],
        "z_dloss_at_lag": e["z_dloss_at_step_plus_lag"],
        "eval_loss_window": {
            str(i): log["eval_loss"][i] for i in range(max(lo, 0), min(hi, STEPS))
        },
        "grad_norm_window": {
            str(i): log["grad_norm"][i] for i in range(max(lo, 0), min(hi, STEPS))
        },
        "injected_window": {
            str(i): log["injected"][i] for i in range(max(lo, 0), min(hi, STEPS))
        },
    }


def mitigation_sweep(train_ids, eval_batches, vocab_size, tok, steps: int = 280) -> list[dict]:
    """Does clipping actually contain the injection?

    Adam rescales each parameter's update by its own running second moment, so a
    large gradient does not automatically produce a large step. Clipping shrinks
    the gradient's magnitude but leaves its direction untouched. Whether that is
    enough is an empirical question, so it is measured across clip thresholds
    rather than assumed.
    """
    global STEPS
    saved = STEPS
    STEPS = steps
    try:
        clean = run_arm(train_ids, eval_batches, vocab_size, False, None, tok)
        rows = []
        for clip in (None, 2.0, 1.0, 0.5, 0.25):
            a = run_arm(train_ids, eval_batches, vocab_size, True, clip, tok)
            exc = excursion(a, clean, INJECT_AT[0], horizon=steps - INJECT_AT[0] - 1)
            rows.append(
                {
                    "clip": clip,
                    "fraction_of_steps_clipped": (
                        None if clip is None else
                        float(np.mean([g > clip for g in a["log"]["grad_norm"]]))
                    ),
                    "grad_norm_at_injection_preclip": a["log"]["grad_norm"][INJECT_AT[0]],
                    "grad_norm_at_injection_postclip":
                        a["log"]["grad_norm_postclip"][INJECT_AT[0]],
                    "peak_excess_loss_vs_clean": exc["peak_excess_loss_vs_clean"],
                    "steps_from_injection_to_peak": exc["steps_from_injection_to_peak"],
                }
            )
        return rows
    finally:
        STEPS = saved


def _setup():
    _, tok, ids = load_corpus()
    split = int(0.9 * len(ids))
    train_ids, held = ids[:split], ids[split:]
    eg = torch.Generator().manual_seed(1234)
    eval_batches = [uniform_batch(held, B, T, eg) for _ in range(2)]
    return tok, train_ids, eval_batches


def stage_main() -> None:
    tok, train_ids, eval_batches = _setup()
    arms = {}
    for name, inject, clip in (
        ("clean/noclip", False, None),
        ("inject/noclip", True, None),
        ("inject/clip1.0", True, 1.0),
    ):
        print(f"running {name} ...", flush=True)
        arms[name] = run_arm(train_ids, eval_batches, tok.vocab_size, inject, clip, tok)
        a = arms[name]
        print(f"   final eval loss {a['final_eval_loss']:.4f}  "
              f"median grad norm {a['median_grad_norm']:.3f}  "
              f"lead events {a['n_lead_events']} lags {a['lags']}", flush=True)
    write_artifact("_t4_arms.json", {"arms": arms})


def stage_mitigation() -> None:
    tok, train_ids, eval_batches = _setup()
    rows = mitigation_sweep(train_ids, eval_batches, tok.vocab_size, tok)
    for r in rows:
        print(f"   clip={str(r['clip']):>5}  peak excess loss "
              f"{r['peak_excess_loss_vs_clean']:.4f} at "
              f"+{r['steps_from_injection_to_peak']}", flush=True)
    write_artifact("_t4_mitigation.json", {"mitigation_sweep": rows})


def main() -> None:
    # Reads the two stage caches; no training happens here.
    arms = read_artifact("_t4_arms.json")["arms"]
    highlighted = {}
    for name, a in arms.items():
        if a["lead_events"]:
            best = max(a["lead_events"], key=lambda e: e["z_grad_at_step"])
            highlighted[name] = describe_event(a, best)

    # The brief asks for "one step where it moved before the loss did", and the
    # clean run produces none at the primary threshold. It does at Z=3.0, and
    # those are the more interesting ones because no injection is involved, so
    # they get the same full step-level context rather than only a count.
    secondary = {}
    for name, a in arms.items():
        zg = np.array([np.nan if v is None else v for v in a["z_grad"]])
        zd = np.array([np.nan if v is None else v for v in a["z_dloss"]])
        evs = [e for e in find_lead_events(zg, zd, z=3.0) if e["lag"] > 1]
        if evs:
            best = max(evs, key=lambda e: e["z_grad_at_step"])
            secondary[name] = {**describe_event(a, best), "z_threshold": 3.0}

    inj, clipped, clean = (arms["inject/noclip"], arms["inject/clip1.0"],
                           arms["clean/noclip"])
    k0 = INJECT_AT[0]
    exc_noclip = excursion(inj, clean, k0)
    exc_clip = excursion(clipped, clean, k0)
    k_last = INJECT_AT[-1]
    peak_step = k0 + exc_noclip["steps_from_injection_to_peak"]

    injection_response = {
        "injected_at_steps": list(INJECT_AT),
        "grad_norm_before_injection": inj["log"]["grad_norm"][k0 - 1],
        "grad_norm_at_injection": inj["log"]["grad_norm"][k0],
        "grad_norm_spike_multiple": inj["log"]["grad_norm"][k0]
        / inj["log"]["grad_norm"][k0 - 1],
        "grad_norm_z_at_injection": inj["z_grad"][k0],
        "eval_loss_at_injection": inj["log"]["eval_loss"][k0],
        "excursion_noclip": exc_noclip,
        "lag_accounting": {
            "injection_window": [k0, k_last],
            "peak_loss_step": peak_step,
            "steps_from_injection_START_to_peak": peak_step - k0,
            "steps_from_injection_END_to_peak": peak_step - k_last,
            "note": "both are reported because quoting only the first would "
                    "flatter the result: the window is four steps wide, so the "
                    "distance from the last bad batch is the smaller number",
        },
        "excursion_clip1.0": exc_clip,
        "final_eval_loss_noclip_minus_clean":
            inj["final_eval_loss"] - clean["final_eval_loss"],
        "final_eval_loss_clip_minus_clean":
            clipped["final_eval_loss"] - clean["final_eval_loss"],
    }

    print("threshold sensitivity ...")
    tsweep = threshold_sweep(arms)
    msweep = read_artifact("_t4_mitigation.json")["mitigation_sweep"]

    # The cleanest statement of the claim, free of the 1-step ordering artefact:
    # at the injection step, how anomalous is the norm and how anomalous is the
    # loss? If a monitor watching the loss would have seen nothing while a
    # monitor watching the norm would have fired, the norm moved first.
    detectability = {
        "step": k0,
        "loss_at_this_step_vs_clean": (inj["log"]["eval_loss"][k0]
                                       - clean["log"]["eval_loss"][k0]),
        "loss_is_bit_identical_to_clean_arm":
            inj["log"]["eval_loss"][k0] == clean["log"]["eval_loss"][k0],
        "grad_norm_previous_step": inj["log"]["grad_norm"][k0 - 1],
        "grad_norm_this_step": inj["log"]["grad_norm"][k0],
        "z_grad": inj["z_grad"][k0],
        "z_dloss": inj["z_dloss"][k0],
        "z_threshold": Z_THRESH,
        "norm_would_have_fired": inj["z_grad"][k0] > Z_THRESH,
        "loss_would_have_fired": inj["z_dloss"][k0] > Z_THRESH,
        "detectability_ratio": inj["z_grad"][k0] / abs(inj["z_dloss"][k0]),
        "steps_until_loss_change_first_exceeds_threshold": next(
            (i for i in range(0, 40)
             if inj["z_dloss"][k0 + i] is not None
             and inj["z_dloss"][k0 + i] > Z_THRESH),
            None,
        ),
    }

    payload = {
        "setup": {
            "B": B, "T": T, "steps": STEPS, "lr": LR,
            "inject_at": list(INJECT_AT),
            "injection_corpus": "Sanskrit, Harvard-Kyoto transliteration (pure ASCII, "
                                "every character already in the Shakespeare vocab)",
            "trailing_window": WINDOW, "z_threshold": Z_THRESH, "max_lag": MAX_LAG,
            "ordering": "eval_loss[k] and grad_norm[k] are both measured at theta_k, "
                        "before the optimizer step that produces theta_{k+1}",
            "caveat": "a lag of 1 is causal by construction and is not evidence; only "
                      "lags >= 2 show the norm moving while the loss still looks fine",
        },
        "arms": arms,
        "highlighted_lead_events": highlighted,
        "lead_events_at_secondary_threshold": secondary,
        "injection_response": injection_response,
        "norm_vs_loss_detectability": detectability,
        "threshold_sensitivity": tsweep,
        "mitigation_sweep": msweep,
        "headline": {
            "lead_events_per_arm": {k: a["n_lead_events"] for k, a in arms.items()},
            "observed_lags": {k: a["lags"] for k, a in arms.items()},
            "max_observed_lag": max(
                [e["lag"] for a in arms.values() for e in a["lead_events"]] or [0]
            ),
            "any_lag_greater_than_one": any(
                e["lag"] > 1 for a in arms.values() for e in a["lead_events"]
            ),
            "clean_run_has_no_lead_events_at_primary_threshold":
                arms["clean/noclip"]["n_lead_events"] == 0,
            "xcorr_lag1_r": {k: next(r["pearson_r"] for r in a["cross_correlation"]
                                     if r["lag"] == 1) for k, a in arms.items()},
            "injection_grad_norm_spike_multiple":
                injection_response["grad_norm_spike_multiple"],
            "peak_excess_loss_noclip": exc_noclip["peak_excess_loss_vs_clean"],
            "steps_from_injection_start_to_peak_loss":
                exc_noclip["steps_from_injection_to_peak"],
            "steps_from_injection_end_to_peak_loss": peak_step - k_last,
            "loss_bit_identical_to_clean_at_the_spike_step":
                inj["log"]["eval_loss"][k0] == clean["log"]["eval_loss"][k0],
            "secondary_threshold_lead_events_with_lag_above_one":
                {k: v["lag_steps"] for k, v in secondary.items()},
            "clipping_reduces_peak_excursion":
                exc_clip["peak_excess_loss_vs_clean"]
                < exc_noclip["peak_excess_loss_vs_clean"],
            "z_grad_at_injection": detectability["z_grad"],
            "z_dloss_at_injection": detectability["z_dloss"],
            "norm_fired_and_loss_did_not_at_injection_step":
                detectability["norm_would_have_fired"]
                and not detectability["loss_would_have_fired"],
            "clean_run_lead_event_lags_at_z3": next(
                r["clean/noclip"]["lags"] for r in tsweep if r["z_threshold"] == 3.0
            ),
            "clipping_saturates": abs(
                msweep[-1]["peak_excess_loss_vs_clean"]
                - msweep[-2]["peak_excess_loss_vs_clean"]
            ) < 1e-3,
            "tightest_clip_still_shows_excursion":
                msweep[-1]["peak_excess_loss_vs_clean"] > 0,
        },
    }
    write_artifact("t4_gradnorm.json", payload)
    plot(payload)


def plot(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = payload["arms"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(21, 9))
    k0 = payload["setup"]["inject_at"][0]

    ax = axes[0][0]
    for name, style in (("clean/noclip", "-"), ("inject/noclip", "-"),
                        ("inject/clip1.0", "--")):
        ax.plot(arms[name]["log"]["step"], arms[name]["log"]["grad_norm"],
                style, lw=1.0, label=name, alpha=.85)
    ax.axvline(k0, color="k", lw=.8, ls=":")
    ax.set_yscale("log"); ax.set_title("Gradient norm every step (pre-clip)")
    ax.set_xlabel("step"); ax.set_ylabel("global grad L2 norm")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[0][1]
    for name, style in (("clean/noclip", "-"), ("inject/noclip", "-"),
                        ("inject/clip1.0", "--")):
        ax.plot(arms[name]["log"]["step"], arms[name]["log"]["eval_loss"],
                style, lw=1.2, label=name, alpha=.85)
    ax.axvline(k0, color="k", lw=.8, ls=":")
    ax.set_title("Held-out loss every step")
    ax.set_xlabel("step"); ax.set_ylabel("token-weighted eval loss")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1][0]
    a = arms["inject/noclip"]
    lo, hi = k0 - 25, k0 + 45
    ax.plot(a["log"]["step"][lo:hi], a["log"]["grad_norm"][lo:hi],
            color="tab:red", lw=1.4, label="grad norm")
    ax.axvline(k0, color="k", lw=.8, ls=":")
    ax.set_yscale("log"); ax.set_ylabel("grad norm", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(a["log"]["step"][lo:hi], a["log"]["eval_loss"][lo:hi],
             color="tab:blue", lw=1.6, label="eval loss")
    ax2.set_ylabel("eval loss", color="tab:blue")
    ax.set_title(f"Zoom on the injection at step {k0} (no clipping)")
    ax.set_xlabel("step"); ax.grid(alpha=.3)

    ax = axes[1][1]
    for name in arms:
        xc = arms[name]["cross_correlation"]
        ax.plot([r["lag"] for r in xc], [r["pearson_r"] for r in xc],
                "o-", lw=1.4, label=name, ms=4)
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("Correlation: grad-norm z vs loss-change z, by lag")
    ax.set_xlabel("lag (steps the loss trails the norm)")
    ax.set_ylabel("Pearson r"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[0][2]
    a = arms["inject/noclip"]
    lo, hi = k0 - 25, k0 + 45
    zg = [np.nan if v is None else v for v in a["z_grad"]][lo:hi]
    zd = [np.nan if v is None else v for v in a["z_dloss"]][lo:hi]
    st = a["log"]["step"][lo:hi]
    ax.plot(st, zg, color="tab:red", lw=1.4, label="grad norm, robust z")
    ax.plot(st, zd, color="tab:blue", lw=1.4, label="loss change, robust z")
    ax.axhline(payload["setup"]["z_threshold"], color="k", ls="--", lw=.9,
               label=f"threshold Z={payload['setup']['z_threshold']}")
    ax.axvline(k0, color="k", lw=.8, ls=":")
    ax.set_title("Same event, both series z-scored")
    ax.set_xlabel("step"); ax.set_ylabel("robust z"); ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[1][2]
    ms = payload["mitigation_sweep"]
    labels = ["none" if r["clip"] is None else str(r["clip"]) for r in ms]
    vals = [r["peak_excess_loss_vs_clean"] for r in ms]
    ax.bar(labels, vals, color="tab:purple", alpha=.8)
    ax.set_title("Peak held-out loss excess after injection, by clip threshold")
    ax.set_xlabel("clip_grad_norm threshold"); ax.set_ylabel("peak excess loss")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=.3, axis="y")

    fig.tight_layout()
    out = FIGURES / "t4_gradnorm.png"
    fig.savefig(out, dpi=130)
    print(f"[figure] {out}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "main"):
        stage_main()
    if stage in ("all", "mitigation"):
        stage_mitigation()
    if stage in ("all", "combine"):
        main()
