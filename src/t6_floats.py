"""Task 6 - write out by hand what 0.1 looks like in fp32, bf16 and fp8 E4M3,
showing the bits. Then say which one you would train in, and why.

"By hand" is taken to mean: derive the bits from the definition of the format,
using exact rational arithmetic, with no help from the machine's own float
encoder. `encode()` below does the long division, the normalisation, and the
round-to-nearest-even, and produces a bit string. Only afterwards is the result
compared against what PyTorch actually stores, as a check that the hand
derivation is right rather than as its source.

fp16 is included as a fourth format even though the brief asks for three,
because bf16 exists specifically to fix what fp16 gets wrong, and you cannot
show that with bf16 alone.

The closing recommendation is not recited from the lecture. Section
`gradient_experiment` casts this repo's own measured gradients into each format
and reports the relative error and how many values flush to zero, so the answer
rests on numbers from the model that was actually trained here.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
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

# name -> (exponent bits, mantissa bits, has_inf)
FORMATS = {
    "fp32": (8, 23, True),
    "fp16": (5, 10, True),
    "bf16": (8, 7, True),
    "fp8_e4m3": (4, 3, False),   # E4M3 as NVIDIA/PyTorch define it: no infinity
}


# --------------------------------------------------------------------------
# the hand encoder
# --------------------------------------------------------------------------
def binary_expansion(x: Fraction, n_bits: int = 40) -> str:
    """Long division: the fractional part of x in binary, digit by digit."""
    assert 0 <= x < 1
    out = []
    for _ in range(n_bits):
        x *= 2
        if x >= 1:
            out.append("1")
            x -= 1
        else:
            out.append("0")
    return "".join(out)


def encode(value: float, exp_bits: int, man_bits: int, has_inf: bool = True,
           overflow_mode: str = "saturate") -> dict:
    """Encode a real number into a custom float format, exactly.

    Round-to-nearest, ties-to-even, which is what every one of these formats
    specifies and what the hardware does.

    `overflow_mode` matters only for a format with no infinity (E4M3FN), and it
    is a property of the CAST, not of the format:

      "saturate" - clamp to the largest finite value (448). What NVIDIA's
                   Transformer Engine does, and what torch >= 2.13 does.
      "nan"      - emit the reserved all-ones code. What torch 2.8 does.

    An earlier version of this file hardwired "saturate" and asserted that the
    hand-derived bits matched PyTorch for every value. That claim was true on
    one PyTorch version and false on another: 487.96 encodes to 01111110 under
    2.13 and 01111111 under 2.8. The format did not change; the cast convention
    did. Both are now representable here, and the artifact records which one the
    installed PyTorch actually implements rather than assuming.
    """
    assert overflow_mode in ("saturate", "nan"), overflow_mode
    bias = 2 ** (exp_bits - 1) - 1
    max_exp = (2 ** exp_bits - 2) if has_inf else (2 ** exp_bits - 1)
    emin = 1 - bias

    v = Fraction(value)
    sign = 0 if v >= 0 else 1
    a = abs(v)
    if a == 0:
        bits = "0" * (1 + exp_bits + man_bits)
        return {"bits": bits, "value": 0.0}

    # normalise into [1, 2)
    e = math.floor(math.log2(float(a)))
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1

    subnormal = e < emin
    if subnormal:
        e = emin
        significand = a / (Fraction(2) ** e)          # < 1
        scaled = significand * (2 ** man_bits)
    else:
        significand = a / (Fraction(2) ** e)          # in [1, 2)
        scaled = (significand - 1) * (2 ** man_bits)

    # round half to even
    floor_v = scaled.numerator // scaled.denominator
    frac = scaled - floor_v
    if frac > Fraction(1, 2) or (frac == Fraction(1, 2) and floor_v % 2 == 1):
        mant = floor_v + 1
    else:
        mant = floor_v

    if not subnormal and mant == 2 ** man_bits:       # rounded up to the next binade
        mant = 0
        e += 1
    if subnormal and mant == 2 ** man_bits:           # rounded up into normal range
        subnormal = False
        mant = 0

    biased_exp = 0 if subnormal else e + bias

    # Overflow, done properly. Two ways to fall off the top:
    #   * the exponent runs past the last usable code, or
    #   * in a format with no infinity (E4M3FN), rounding lands exactly on the
    #     all-ones exponent with an all-ones mantissa, which is the reserved NaN
    #     code and not a number at all.
    # The second case is why an earlier version of this function reported
    # 487.96 as 480.0 in E4M3 while format_properties correctly said the largest
    # finite value is 448. A randomised cross-check against PyTorch caught it.
    all_ones_exp = 2 ** exp_bits - 1
    hits_nan_code = (not has_inf and biased_exp == all_ones_exp
                     and mant == 2 ** man_bits - 1)
    overflow = biased_exp > max_exp or hits_nan_code

    saturated = False
    if overflow:
        if has_inf:
            # IEEE behaviour: round-to-nearest overflows to infinity.
            biased_exp, mant = all_ones_exp, 0
            stored = Fraction(0)   # placeholder; reported as inf below
        elif overflow_mode == "saturate":
            biased_exp, mant = all_ones_exp, 2 ** man_bits - 2
            e = biased_exp - bias
            saturated = True
            subnormal = False
        else:
            biased_exp, mant = all_ones_exp, 2 ** man_bits - 1
            e = biased_exp - bias
            subnormal = False
            stored = None

    if not (overflow and (has_inf or overflow_mode == "nan")):
        stored = (
            Fraction(mant, 2 ** man_bits) * Fraction(2) ** emin
            if subnormal
            else (1 + Fraction(mant, 2 ** man_bits)) * Fraction(2) ** e
        )
    else:
        stored = None
    if sign and stored is not None:
        stored = -stored

    bits = (
        f"{sign:01b}"
        + format(biased_exp, f"0{exp_bits}b")
        + format(mant, f"0{man_bits}b")
    )
    if stored is None:
        stored_f = (math.nan if (overflow and not has_inf)
                    else math.inf * (-1 if sign else 1))
    else:
        stored_f = float(stored)
    err = (abs(stored_f - value) / abs(value)
           if value and math.isfinite(stored_f) else float("inf")
           if not math.isfinite(stored_f) else 0.0)
    return {
        "bits": bits,
        "sign_bit": f"{sign:01b}",
        "exponent_bits": format(biased_exp, f"0{exp_bits}b"),
        "mantissa_bits": format(mant, f"0{man_bits}b"),
        "biased_exponent": biased_exp,
        "bias": bias,
        "unbiased_exponent": emin if subnormal else e,
        "mantissa_integer": mant,
        "implicit_leading_bit": 0 if subnormal else 1,
        "subnormal": subnormal,
        "overflow": overflow,
        "overflow_mode": overflow_mode,
        "saturated_to_largest_finite": saturated,
        "significand_decimal": float(
            Fraction(mant, 2 ** man_bits) + (0 if subnormal else 1)
        ),
        "stored_value": stored_f,
        "absolute_error": abs(stored_f - value) if math.isfinite(stored_f)
                          else float("inf"),
        "relative_error": err,
        "relative_error_percent": err * 100,
        "hex": f"0x{int(bits, 2):0{(1 + exp_bits + man_bits) // 4}X}",
    }


# --------------------------------------------------------------------------
# what the machine actually stores, for cross-checking
# --------------------------------------------------------------------------
def torch_bits(value: float, name: str) -> dict | None:
    dt = {
        "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16,
        "fp8_e4m3": getattr(torch, "float8_e4m3fn", None),
    }[name]
    if dt is None:
        return None
    t = torch.tensor([value], dtype=torch.float32).to(dt)
    n_bits = {"fp32": 32, "fp16": 16, "bf16": 16, "fp8_e4m3": 8}[name]
    view = {32: torch.int32, 16: torch.int16, 8: torch.int8}[n_bits]
    raw = int(t.view(view).item()) & (2 ** n_bits - 1)
    return {
        "bits": format(raw, f"0{n_bits}b"),
        "stored_value": float(t.to(torch.float64).item()),
    }


# --------------------------------------------------------------------------
# format properties
# --------------------------------------------------------------------------
def detect_torch_overflow_mode() -> dict:
    """Which E4M3 overflow convention does the installed PyTorch implement?

    Not a detail: torch 2.8 and torch 2.13 disagree, and a cross-check that
    assumed one of them fails on the other for reasons that have nothing to do
    with whether the encoder is right.
    """
    dt = getattr(torch, "float8_e4m3fn", None)
    if dt is None:
        return {"available": False}
    t = torch.tensor([1e6], dtype=torch.float32).to(dt)
    raw = int(t.view(torch.int8).item()) & 0xFF
    bits = format(raw, "08b")
    mode = "saturate" if bits == "01111110" else "nan" if bits == "01111111" else "unknown"
    return {
        "available": True,
        "torch_version": torch.__version__,
        "bits_for_1e6": bits,
        "mode": mode,
        "note": "torch 2.8 emits the reserved NaN code on E4M3 overflow; "
                "torch >= 2.13 saturates to 448, as NVIDIA's Transformer Engine "
                "does. The format is identical; the cast convention differs.",
    }


def format_properties(exp_bits: int, man_bits: int, has_inf: bool) -> dict:
    bias = 2 ** (exp_bits - 1) - 1
    emin = 1 - bias
    emax = (2 ** exp_bits - 2 if has_inf else 2 ** exp_bits - 1) - bias
    if has_inf:
        # all-ones exponent is reserved for inf/NaN, so the top binade is unusable
        max_mant = 2 - Fraction(1, 2 ** man_bits)
    else:
        # E4M3FN keeps the top binade but reserves the all-ones mantissa for NaN,
        # which is why its maximum is 448 and not 480
        max_mant = 2 - Fraction(2, 2 ** man_bits)
    return {
        "total_bits": 1 + exp_bits + man_bits,
        "exponent_bits": exp_bits,
        "mantissa_bits": man_bits,
        "exponent_bias": bias,
        "smallest_normal": float(Fraction(2) ** emin),
        "smallest_subnormal": float(Fraction(1, 2 ** man_bits) * Fraction(2) ** emin),
        "largest_finite": float(max_mant * Fraction(2) ** emax),
        "machine_epsilon": float(Fraction(1, 2 ** man_bits)),
        "decimal_digits_of_precision": round((man_bits + 1) * math.log10(2), 2),
        "bytes_per_value": (1 + exp_bits + man_bits) / 8,
        "top_code_reserved": "all-ones exponent -> inf/NaN" if has_inf
                             else "all-ones exponent with all-ones mantissa -> NaN",
    }


# --------------------------------------------------------------------------
# does the choice matter for THIS model's gradients?
# --------------------------------------------------------------------------
def gradient_experiment() -> dict:
    """Cast the repo's own gradients into each format and measure the damage."""
    set_seed(1337)
    _, tok, ids = load_corpus()
    model = build_model(GPTConfig(vocab_size=tok.vocab_size, block_size=64))
    gen = torch.Generator().manual_seed(0)
    x, y = uniform_batch(ids, 16, 64, gen)
    _, loss = model(x, y)
    loss.backward()
    g = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]).double()

    nz = g[g != 0]
    stats = {
        "n_gradient_values": int(g.numel()),
        "abs_min_nonzero": float(nz.abs().min()),
        "abs_median": float(g.abs().median()),
        "abs_max": float(g.abs().max()),
        "dynamic_range_orders_of_magnitude": float(
            math.log10(float(g.abs().max()) / float(nz.abs().min()))
        ),
    }

    out = {}
    for name in ("fp16", "bf16", "fp8_e4m3"):
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16,
              "fp8_e4m3": getattr(torch, "float8_e4m3fn", None)}[name]
        if dt is None:
            continue
        r = g.to(torch.float32).to(dt).to(torch.float64)
        newly_zero = int(((g != 0) & (r == 0)).sum())
        finite = torch.isfinite(r)
        mask = (g != 0) & (r != 0) & finite
        rel = ((r[mask] - g[mask]).abs() / g[mask].abs())
        out[name] = {
            "values_flushed_to_zero": newly_zero,
            "fraction_flushed_to_zero": newly_zero / int((g != 0).sum()),
            "values_overflowed_to_inf_or_nan": int((~finite).sum()),
            "median_relative_error": float(rel.median()),
            "p99_relative_error": float(rel.quantile(0.99)),
            "max_relative_error": float(rel.max()),
        }

    # The session's suggested fix: scale the tensor outside the format so the
    # values land in the part of the range the format can actually represent,
    # then scale back. This is what NVIDIA's transformer engine does per tensor.
    scaled = {}
    for name, target_max in (("fp16", 6.0e4), ("fp8_e4m3", 4.4e2)):
        dt = {"fp16": torch.float16,
              "fp8_e4m3": getattr(torch, "float8_e4m3fn", None)}[name]
        if dt is None:
            continue
        s = (target_max / 2) / float(g.abs().max())
        r = (g.to(torch.float32) * s).to(dt).to(torch.float64) / s
        newly_zero = int(((g != 0) & (r == 0)).sum())
        mask = (g != 0) & (r != 0) & torch.isfinite(r)
        rel = ((r[mask] - g[mask]).abs() / g[mask].abs())
        scaled[name] = {
            "scale_factor": s,
            "fraction_flushed_to_zero": newly_zero / int((g != 0).sum()),
            "median_relative_error": float(rel.median()),
            "p99_relative_error": float(rel.quantile(0.99)),
        }

    return {"gradient_stats": stats, "per_format": out,
            "with_global_scaling": scaled,
            "note": "gradients from one real step of the model in this repo, "
                    "cast fp32 -> format -> fp64 and compared. The scaled variant "
                    "multiplies by a single factor before the cast and divides "
                    "after, which is the trick the session described for fp8."}


# --------------------------------------------------------------------------
def main() -> None:
    value = 0.1
    torch_overflow = detect_torch_overflow_mode()

    expansion = binary_expansion(Fraction(1, 10))
    encodings, checks = {}, {}
    for name, (eb, mb, hi) in FORMATS.items():
        enc = encode(value, eb, mb, hi)
        enc["properties"] = format_properties(eb, mb, hi)
        encodings[name] = enc
        tb = torch_bits(value, name)
        if tb:
            checks[name] = {
                "hand_bits": enc["bits"],
                "torch_bits": tb["bits"],
                "bits_match": enc["bits"] == tb["bits"],
                "hand_value": enc["stored_value"],
                "torch_value": tb["stored_value"],
                "values_match": abs(enc["stored_value"] - tb["stored_value"]) == 0.0,
            }

    # the session's claim, checked: which formats can still hold a small gradient
    probes = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-38, 1e-45]
    representable = {
        name: {
            f"{p:.0e}": {
                "stored": encode(p, eb, mb, hi)["stored_value"],
                "flushes_to_zero": encode(p, eb, mb, hi)["stored_value"] == 0.0,
                "subnormal": encode(p, eb, mb, hi)["subnormal"],
                "relative_error_percent": encode(p, eb, mb, hi)["relative_error_percent"],
            }
            for p in probes
        }
        for name, (eb, mb, hi) in FORMATS.items()
    }

    grad = gradient_experiment()

    payload = {
        "value": value,
        "why_0_1_is_hard": {
            "exact_rational": "1/10",
            "binary_expansion_of_0.1": "0." + expansion + "...",
            "repeating_block": "0011",
            "explanation": "1/10 has the prime factor 5 in its denominator, and 2 "
                           "has no power that clears it, so the binary expansion "
                           "never terminates. Every binary float format must "
                           "truncate it, so no binary format stores 0.1 exactly. "
                           "0.5 or 0.25 would have been uninteresting.",
            "normalised_form": "1.1001100110011... x 2^-4",
        },
        "encodings": encodings,
        "cross_check_against_pytorch": checks,
        "all_hand_encodings_match_pytorch": all(c["bits_match"] for c in checks.values()),
        "small_value_survival": representable,
        "gradient_experiment": grad,
        "torch_overflow_convention": torch_overflow,
        "overflow_is_a_cast_mode_not_a_format_property": {
            "saturate": encode(1e6, 4, 3, has_inf=False, overflow_mode="saturate"),
            "nan": encode(1e6, 4, 3, has_inf=False, overflow_mode="nan"),
        },
        "headline": {
            "fp32_bits": encodings["fp32"]["bits"],
            "fp32_stored": encodings["fp32"]["stored_value"],
            "bf16_bits": encodings["bf16"]["bits"],
            "bf16_stored": encodings["bf16"]["stored_value"],
            "bf16_error_percent": encodings["bf16"]["relative_error_percent"],
            "fp8_e4m3_bits": encodings["fp8_e4m3"]["bits"],
            "fp8_e4m3_stored": encodings["fp8_e4m3"]["stored_value"],
            "fp8_e4m3_error_percent": encodings["fp8_e4m3"]["relative_error_percent"],
            "bf16_and_fp32_share_exponent_width":
                encodings["bf16"]["properties"]["exponent_bits"]
                == encodings["fp32"]["properties"]["exponent_bits"],
            "bf16_smallest_normal": encodings["bf16"]["properties"]["smallest_normal"],
            "fp16_smallest_normal": encodings["fp16"]["properties"]["smallest_normal"],
            "gradient_fraction_flushed_bf16":
                grad["per_format"]["bf16"]["fraction_flushed_to_zero"],
            "gradient_fraction_flushed_fp16":
                grad["per_format"]["fp16"]["fraction_flushed_to_zero"],
            "gradient_fraction_flushed_fp8":
                grad["per_format"]["fp8_e4m3"]["fraction_flushed_to_zero"],
            "gradient_fraction_flushed_fp8_with_scaling":
                grad["with_global_scaling"]["fp8_e4m3"]["fraction_flushed_to_zero"],
            "bf16_loses_no_gradients":
                grad["per_format"]["bf16"]["fraction_flushed_to_zero"] == 0.0,
            "scaling_rescues_fp8":
                grad["with_global_scaling"]["fp8_e4m3"]["fraction_flushed_to_zero"]
                < grad["per_format"]["fp8_e4m3"]["fraction_flushed_to_zero"],
        },
    }
    write_artifact("t6_floats.json", payload)

    print(f"0.1 in binary = 0.{expansion[:32]}...  (block '0011' repeats forever)")
    print(f"{'format':<10}{'sign':>5} {'exponent':<10} {'mantissa':<24}"
          f"{'stored':>22}{'err %':>10}")
    for name, e in encodings.items():
        print(f"{name:<10}{e['sign_bit']:>5} {e['exponent_bits']:<10} "
              f"{e['mantissa_bits']:<24}{e['stored_value']:>22.18f}"
              f"{e['relative_error_percent']:>10.4f}")
    print(f"\nhand-derived bits match PyTorch for every format: "
          f"{payload['all_hand_encodings_match_pytorch']}")
    print(f"this PyTorch ({torch_overflow.get('torch_version')}) handles E4M3 "
          f"overflow by: {torch_overflow.get('mode')}")
    gs = grad["gradient_stats"]
    print(f"\ngradient magnitudes: min {gs['abs_min_nonzero']:.3e}, "
          f"median {gs['abs_median']:.3e}, max {gs['abs_max']:.3e} "
          f"({gs['dynamic_range_orders_of_magnitude']:.1f} orders of magnitude)")
    print("gradient cast damage (one real step of this repo's model):")
    for name, r in grad["per_format"].items():
        print(f"  {name:<14} flushed to zero {r['fraction_flushed_to_zero']*100:8.4f}%"
              f"   median rel err {r['median_relative_error']*100:8.4f}%")
    for name, r in grad["with_global_scaling"].items():
        print(f"  {name+' +scale':<14} flushed to zero "
              f"{r['fraction_flushed_to_zero']*100:8.4f}%"
              f"   median rel err {r['median_relative_error']*100:8.4f}%")


if __name__ == "__main__":
    main()
