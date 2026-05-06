"""
Batch test harness - measure decode accuracy across many random IDs and wear levels.

This is the most important test: if these numbers don't move toward the ТЗ targets,
the codec design is wrong, not the implementation.

Usage:
    python tests/test_batch.py [--n 100] [--res 96]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sibling packages importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from codec import PatternCodec
from perception import simulate_capture, ChannelExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of random IDs to test")
    ap.add_argument("--res", type=int, default=96, help="grid resolution")
    ap.add_argument("--wear-levels", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    item_ids = [int(x) for x in rng.integers(0, 2**32, size=args.n)]

    codec = PatternCodec()
    extractor = ChannelExtractor()

    print(f"running {args.n} ids x {len(args.wear_levels)} wear levels @ res={args.res}")
    print(f"{'wear':>5}  {'full':>6}  {'rs_fix':>6}  {'cat':>6}  {'lost':>6}  "
          f"{'id_acc':>7}  {'cat_acc':>7}")
    print("-" * 64)

    overall_t0 = time.time()
    for wf in args.wear_levels:
        n_full = n_rs = n_cat = n_lost = 0
        n_id_correct = 0
        n_cat_correct = 0
        for i, item_id in enumerate(item_ids):
            ch = codec.encode(item_id)
            _, obs = simulate_capture(ch, wear_factor=wf,
                                      grid_resolution=args.res,
                                      rng_seed=i)
            extraction = extractor.extract(obs)
            result = codec.decode(extraction.to_decode_dict())

            if result.fallback_level == "full":
                n_full += 1
            elif result.fallback_level == "rs_corrected":
                n_rs += 1
            elif result.fallback_level == "category_only":
                n_cat += 1
            else:
                n_lost += 1

            if result.item_id == item_id:
                n_id_correct += 1

            # Category accuracy = did we recover macro+count correctly?
            if extraction.macro == ch.macro and extraction.count == ch.count:
                n_cat_correct += 1

        n = len(item_ids)
        print(f"{wf:>5.1f}  {n_full:>6d}  {n_rs:>6d}  {n_cat:>6d}  {n_lost:>6d}  "
              f"{n_id_correct/n*100:>6.1f}%  {n_cat_correct/n*100:>6.1f}%")
    print("-" * 64)
    print(f"total time: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
