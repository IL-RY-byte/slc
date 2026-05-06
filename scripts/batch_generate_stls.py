"""
Generate a diverse set of tile STLs for Blender render training data.

Creates one STL per unique channel combination sampled to cover all 16 macro
families, a range of counts, heights, and angles.  Use the output directory
as --stl_dir for render_tiles.py.

Usage:
    python scripts/batch_generate_stls.py --out tiles/ --n 500
    python scripts/batch_generate_stls.py --out tiles/ --n 100 --seed 7

Output:
    tiles/tile_XXXXXXXX.stl   (one per sampled item ID)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codec import PatternCodec
from geometry import PatternGeometry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out',    type=Path, default=Path('tiles'),
                    help='output directory for STL files')
    ap.add_argument('--n',      type=int,  default=500,
                    help='number of STLs to generate')
    ap.add_argument('--res',    type=int,  default=128,
                    help='grid resolution for height field (default 128)')
    ap.add_argument('--radius', type=float, default=15.0,
                    help='tile radius in mm (default 15.0)')
    ap.add_argument('--seed',   type=int,  default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rng   = np.random.default_rng(args.seed)
    codec = PatternCodec()
    geom  = PatternGeometry(radius_mm=args.radius, grid_resolution=args.res)

    # Sample random item IDs, ensuring all 16 macros are represented.
    # Item ID bit layout: macro[31:28] | count[27:24] | height[23:16] | angles[15:8] | micro[7:0]
    # One fixed ID per macro (count=1, height=128, angles=128, micro=128):
    fixed_ids = [(m << 28) | (1 << 24) | (128 << 16) | (128 << 8) | 128
                 for m in range(16)]
    random_ids = [int(x) for x in rng.integers(0, 2**32, size=max(0, args.n - 16))]
    item_ids   = (fixed_ids + random_ids)[:args.n]

    print(f'Generating {len(item_ids)} STLs -> {args.out}')
    ok = failed = 0
    for item_id in item_ids:
        try:
            channels = codec.encode(item_id)
            mesh     = geom.generate(channels)
            stl_path = args.out / f'tile_{item_id:08X}.stl'
            stl_path.write_bytes(mesh.to_stl_bytes())
            ok += 1
        except Exception as e:
            print(f'  FAILED 0x{item_id:08X}: {e}', file=sys.stderr)
            failed += 1

        if (ok + failed) % 50 == 0:
            print(f'  {ok + failed}/{len(item_ids)}  ({ok} ok, {failed} failed)',
                  flush=True)

    print(f'\nDone: {ok} STLs saved to {args.out}  ({failed} failed)')
    size_mb = sum(f.stat().st_size for f in args.out.glob('*.stl')) / 1e6
    print(f'Total size: {size_mb:.1f} MB')


if __name__ == '__main__':
    main()
