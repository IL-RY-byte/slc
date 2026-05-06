"""
Generate synthetic training data for the macro CNN and wear estimator.

Renders 200K height fields (16 macro families x 7 wear levels x ~1800 random IDs),
crops the 64x64 inner zone from each 128x128 field, normalises to [0..1], and
saves everything as a compressed NumPy archive.

Usage:
    # Full dataset (~200K samples, takes ~10-30min depending on CPU)
    python scripts/generate_cnn_data.py --out data/cnn_train.npz

    # Quick smoke-test (1120 samples, seconds)
    python scripts/generate_cnn_data.py --n 10 --out /tmp/cnn_smoke.npz

    # Parallel (uses all cores)
    python scripts/generate_cnn_data.py --workers 8 --out data/cnn_train.npz

Output npz keys:
    X        float32  (N, 64, 64)  - normalised inner-zone height field
    y_macro  uint8    (N,)         - macro family 0..15
    y_wear   float32  (N,)         - wear factor 0.0..0.6
    y_count  uint8    (N,)         - protrusion count 0..15 (bonus label)

The wear label is useful for training the wear estimator (Enhancement 2) from
the same dataset without running a separate render pass.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

# Make sibling packages importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codec.pattern_codec import Channels, PatternCodec
from perception.perception import simulate_capture


# ---------------------------------------------------------------------------
# Constants - must agree with ScanPipeline._extract64x64()
# ---------------------------------------------------------------------------
_RES         = 128   # full height field resolution
_INNER_START = 32    # first pixel of inner zone (0-based)
_INNER_END   = 96    # exclusive - yields 64 pixels
_INNER_SIDE  = _INNER_END - _INNER_START   # 64

_WEAR_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
_MACROS      = list(range(16))


# ---------------------------------------------------------------------------
# Worker - renders one (macro, wear, seed) tuple
# ---------------------------------------------------------------------------

def _render_sample(args: tuple) -> tuple[np.ndarray, int, float, int] | None:
    """Return (crop64, macro, wear, count) or None on failure."""
    macro, wear, count, height, angles, micro, seed = args
    try:
        rs_placeholder = bytes(4)  # RS bytes irrelevant for CNN - any value
        ch = Channels(
            macro=macro, count=count, height=height,
            angles=angles, micro=micro, rs=rs_placeholder,
        )
        _, observed = simulate_capture(
            ch,
            wear_factor=wear,
            noise_sigma_mm=0.05,
            grid_resolution=_RES,
            rng_seed=seed,
        )
        # observed is (_RES, _RES) float64
        crop = observed[_INNER_START:_INNER_END, _INNER_START:_INNER_END].astype(np.float32)
        v_min, v_max = crop.min(), crop.max()
        rng = v_max - v_min
        if rng > 1e-9:
            crop = (crop - v_min) / rng
        return crop, macro, float(wear), count
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=1800,
                    help='samples per (macro, wear) combination (default 1800 -> 201600 total)')
    ap.add_argument('--out', type=Path, default=Path('data/cnn_train.npz'),
                    help='output .npz path')
    ap.add_argument('--seed', type=int, default=42,
                    help='global RNG seed for reproducibility')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1),
                    help='worker processes (default: nCPU-1)')
    args = ap.parse_args()

    total = len(_MACROS) * len(_WEAR_LEVELS) * args.n
    print(f'Generating {total:,} samples '
          f'({len(_MACROS)} macros x {len(_WEAR_LEVELS)} wear levels x {args.n}/cell) ...')
    print(f'Workers : {args.workers}')
    print(f'Output  : {args.out}')

    rng = np.random.default_rng(args.seed)

    # Build task list
    tasks: list[tuple] = []
    global_seed = 0
    for macro in _MACROS:
        for wear in _WEAR_LEVELS:
            counts  = rng.integers(0, 16, size=args.n)
            heights = rng.integers(0, 256, size=args.n)
            angles  = rng.integers(0, 256, size=args.n)
            micros  = rng.integers(0, 256, size=args.n)
            for i in range(args.n):
                tasks.append((
                    macro, wear,
                    int(counts[i]), int(heights[i]),
                    int(angles[i]),  int(micros[i]),
                    global_seed,
                ))
                global_seed += 1

    # Shuffle so that batches are class-balanced (useful if training is
    # interrupted and a partial npz is loaded)
    rng.shuffle(tasks)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # X is large (total x 64 x 64 x 4 bytes ~ 3 GB for 200K samples).
    # Write directly to a memory-mapped npy file so we never hold the full
    # array in RAM.  The companion .npz holds the small label arrays.
    x_path = args.out.parent / (args.out.stem + '_X.npy')
    X_mmap = np.lib.format.open_memmap(
        str(x_path), mode='w+', dtype=np.float32, shape=(total, _INNER_SIDE, _INNER_SIDE),
    )

    y_macro_list: list[int]   = []
    y_wear_list:  list[float] = []
    y_count_list: list[int]   = []
    failed = 0
    n_ok   = 0

    t0 = time.time()
    chunk = max(1, total // 100)   # report every ~1%

    if args.workers == 1:
        for i, task in enumerate(tasks):
            result = _render_sample(task)
            if result is None:
                failed += 1
            else:
                crop, macro, wear, count = result
                X_mmap[n_ok] = crop
                y_macro_list.append(macro)
                y_wear_list.append(wear)
                y_count_list.append(count)
                n_ok += 1
            if (i + 1) % chunk == 0:
                elapsed = time.time() - t0
                pct = (i + 1) / total * 100
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f'  {pct:5.1f}%  {i+1:>7,}/{total:>7,}  '
                      f'elapsed={elapsed:.0f}s  eta={eta:.0f}s', flush=True)
    else:
        with Pool(processes=args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_render_sample, tasks, chunksize=32)):
                if result is None:
                    failed += 1
                else:
                    crop, macro, wear, count = result
                    X_mmap[n_ok] = crop
                    y_macro_list.append(macro)
                    y_wear_list.append(wear)
                    y_count_list.append(count)
                    n_ok += 1
                if (i + 1) % chunk == 0:
                    elapsed = time.time() - t0
                    pct = (i + 1) / total * 100
                    eta = elapsed / (i + 1) * (total - i - 1)
                    print(f'  {pct:5.1f}%  {i+1:>7,}/{total:>7,}  '
                          f'elapsed={elapsed:.0f}s  eta={eta:.0f}s', flush=True)

    del X_mmap  # flush memmap to disk

    print(f'\nRendered {n_ok:,} samples ({failed} failed)')

    if n_ok == 0:
        print('ERROR: no samples rendered - check perception/geometry imports', file=sys.stderr)
        sys.exit(1)

    y_macro = np.array(y_macro_list, dtype=np.uint8)
    y_wear  = np.array(y_wear_list,  dtype=np.float32)
    y_count = np.array(y_count_list, dtype=np.uint8)

    # Save label arrays + n_ok in the npz; X lives in the companion _X.npy.
    # TileDataset loads _X.npy with mmap_mode='r' so only active batches
    # are read into RAM during training.
    np.savez_compressed(
        args.out,
        y_macro=y_macro,
        y_wear=y_wear,
        y_count=y_count,
        n_ok=np.array(n_ok, dtype=np.int64),
    )
    size_mb = args.out.stat().st_size / 1e6
    x_size_mb = x_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f'Saved {args.out}  ({size_mb:.1f} MB labels)')
    print(f'Saved {x_path}  ({x_size_mb:.0f} MB X array)')
    print(f'Total: {n_ok:,} samples, {elapsed:.0f}s')
    print(f'\nClass balance check (samples per macro):')
    for m in range(16):
        cnt = int((y_macro == m).sum())
        print(f'  macro={m:2d}: {cnt:6,}')
    print(f'\nWear distribution:')
    for w in _WEAR_LEVELS:
        cnt = int(np.isclose(y_wear, w).sum())
        print(f'  wear={w:.1f}: {cnt:6,}')


if __name__ == '__main__':
    main()
