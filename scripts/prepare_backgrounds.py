"""
Validate and resize fabric background photos for Blender render pipeline.

Accepts any collection of JPG/PNG images (e.g., downloaded CC0 fabric textures).
Filters out images that are too small, too uniform (solid colour), or corrupt.
Resizes survivors to 1024x1024 px and saves to --out.

Target: 1000+ images for render_tiles.py --backgrounds.

Usage:
    python scripts/prepare_backgrounds.py --src raw_fabrics/ --out fabric_photos/
    python scripts/prepare_backgrounds.py --src raw_fabrics/ --out fabric_photos/ --min-size 256

Free fabric image sources (CC0/public domain):
    https://www.transparenttextures.com/
    https://www.toptal.com/designers/subtlepatterns/
    https://polyhaven.com/textures  (filter: fabric)
    Kaggle: search "fabric texture dataset"

Output:
    fabric_photos/  -- resized 1024x1024 JPG images ready for Blender
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src',      type=Path, required=True,
                    help='input directory of fabric images')
    ap.add_argument('--out',      type=Path, default=Path('fabric_photos'),
                    help='output directory for prepared images (default: fabric_photos/)')
    ap.add_argument('--size',     type=int,  default=1024,
                    help='output resolution in pixels (default: 1024)')
    ap.add_argument('--min-size', type=int,  default=256,
                    help='skip images smaller than this in either dimension (default: 256)')
    ap.add_argument('--min-std',  type=float, default=15.0,
                    help='skip images with per-channel std below this (solid colour check)')
    ap.add_argument('--quality',  type=int,  default=92,
                    help='JPEG quality for output images (default: 92)')
    args = ap.parse_args()

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print('Pillow and numpy required. Install with: pip install pillow numpy',
              file=sys.stderr)
        sys.exit(1)

    if not args.src.exists():
        print(f'ERROR: source directory {args.src} does not exist', file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    candidates = [f for f in args.src.rglob('*') if f.suffix.lower() in exts]
    print(f'Found {len(candidates)} image files in {args.src}')

    ok = skipped_small = skipped_uniform = skipped_corrupt = 0

    for src_path in sorted(candidates):
        try:
            img = Image.open(src_path).convert('RGB')
        except Exception:
            skipped_corrupt += 1
            continue

        w, h = img.size
        if w < args.min_size or h < args.min_size:
            skipped_small += 1
            continue

        # Uniformity check: compute std of a centre crop
        cx, cy = w // 2, h // 2
        crop_size = min(w, h, 256)
        crop = img.crop((cx - crop_size // 2, cy - crop_size // 2,
                         cx + crop_size // 2, cy + crop_size // 2))
        arr = np.array(crop, dtype=np.float32)
        if arr.std() < args.min_std:
            skipped_uniform += 1
            continue

        # Resize to target (crop to square first preserving centre)
        short = min(w, h)
        img = img.crop(((w - short) // 2, (h - short) // 2,
                         (w + short) // 2, (h + short) // 2))
        img = img.resize((args.size, args.size), Image.LANCZOS)

        out_name = src_path.stem + '.jpg'
        # Avoid name collisions from nested directories
        if (args.out / out_name).exists():
            out_name = src_path.parent.name + '_' + out_name
        img.save(args.out / out_name, 'JPEG', quality=args.quality, optimize=True)
        ok += 1

        if ok % 100 == 0:
            print(f'  processed {ok} images so far ...', flush=True)

    total = ok + skipped_small + skipped_uniform + skipped_corrupt
    print(f'\nDone: {ok} images -> {args.out}')
    print(f'  Skipped: {skipped_small} too small  |  '
          f'{skipped_uniform} too uniform  |  '
          f'{skipped_corrupt} corrupt')

    if ok < 100:
        print(f'\nWARNING: only {ok} background images.')
        print('render_tiles.py works best with 1000+ diverse fabric images.')
        print('See script header for free CC0 texture sources.')
    else:
        print(f'\nOK: {ok} backgrounds ready for render_tiles.py --backgrounds {args.out}')


if __name__ == '__main__':
    main()
