"""Regenerate every artifact and figure in this repo, in order.

    python src/run_all.py

Task 4 is the expensive one (three 400-step arms plus a six-point clip sweep),
so it is split into cached stages; `combine` reads the two caches and writes the
final artifact. Task 5 is staged the same way. Deleting artifacts/_t4_*.json or
artifacts/_t5_*.json forces a full recompute of that task.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t1_shapes
import t2_finite_diff
import t3_accum
import t4_gradnorm
import t5_mfu
import t6_floats
import t6b_bf16_train
import render_readme
from common import ARTIFACTS


def main() -> None:
    steps = [
        ("Task 1  tensor shape ledger", t1_shapes.main),
        ("Task 2  gradient by hand", t2_finite_diff.main),
        ("Task 3  broken gradient accumulation", t3_accum.main),
        ("Task 4a grad norm: training arms", t4_gradnorm.stage_main),
        ("Task 4b grad norm: clip sweep", t4_gradnorm.stage_mitigation),
        ("Task 4c grad norm: analysis", t4_gradnorm.main),
        ("Task 5a MFU: peak, baseline, repeatability", t5_mfu.stage_measure),
        ("Task 5b MFU: size sweep", t5_mfu.stage_sweep),
        ("Task 5c MFU: analysis", t5_mfu.main),
        ("Task 5d MFU: regenerate the README section", render_readme.main),
        ("Task 6  float formats", t6_floats.main),
        ("Task 6b the bf16 recommendation, trained", t6b_bf16_train.main),
    ]
    t_all = time.perf_counter()
    for label, fn in steps:
        print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
        t = time.perf_counter()
        fn()
        print(f"-- {label}: {time.perf_counter() - t:.1f}s", flush=True)
    print(f"\nall tasks complete in {time.perf_counter() - t_all:.1f}s")
    print(f"artifacts in {ARTIFACTS}")


if __name__ == "__main__":
    main()
