"""
Copy trained model files into the Flutter asset tree.

Run this after train_macro_cnn.py, export_midas.py, and export_models.py complete.
On macOS/Linux it can also trigger the CoreML/TFLite export before copying.

Usage:
    # Copy ONNX only (Windows - no CoreML/TFLite export possible)
    python scripts/deploy_models.py --onnx-only

    # Full deploy (macOS/Linux): export CoreML + TFLite, then copy all
    python scripts/deploy_models.py

Outputs written to flutter/assets/models/:
    macro_cnn.onnx           (all platforms - for validation)
    macro_cnn.mlpackage/     (CoreML iOS)
    macro_cnn.tflite         (TFLite Android)
    midas_small.onnx         (optional - ONNX reference copy)
    midas_small.mlpackage/   (CoreML iOS - non-LiDAR fallback)
    midas_small.tflite       (TFLite Android - non-LiDAR fallback)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
ASSETS = ROOT / 'flutter' / 'assets' / 'models'


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    size = sum(f.stat().st_size for f in dst.rglob('*') if f.is_file()) \
           if dst.is_dir() else dst.stat().st_size
    print(f'  copied {src.name} -> {dst}  ({size/1e6:.1f} MB)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx',        type=Path, default=MODELS / 'macro_cnn.onnx')
    ap.add_argument('--mlpackage',   type=Path, default=MODELS / 'macro_cnn.mlpackage')
    ap.add_argument('--tflite',      type=Path, default=MODELS / 'macro_cnn.tflite')
    ap.add_argument('--onnx-only',   action='store_true',
                    help='skip CoreML/TFLite export/copy (Windows)')
    ap.add_argument('--skip-export', action='store_true',
                    help='skip export step, only copy existing files')
    args = ap.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)

    # -- Run export_models.py if requested --
    if not args.onnx_only and not args.skip_export:
        if not args.onnx.exists():
            print(f'ERROR: {args.onnx} not found. Run train_macro_cnn.py first.',
                  file=sys.stderr)
            sys.exit(1)
        print('Running export_models.py ...')
        result = subprocess.run(
            [sys.executable, 'scripts/export_models.py',
             '--onnx', str(args.onnx), '--out', str(MODELS)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print('Export failed - check coremltools/tensorflow installation.',
                  file=sys.stderr)

    # -- Copy files --
    print(f'\nCopying to {ASSETS}:')

    if args.onnx.exists():
        copy_file(args.onnx, ASSETS / 'macro_cnn.onnx')
    else:
        print(f'  SKIP: {args.onnx.name} not found')

    if not args.onnx_only:
        if args.mlpackage.exists():
            copy_file(args.mlpackage, ASSETS / 'macro_cnn.mlpackage')
        else:
            print(f'  SKIP: {args.mlpackage.name} not found (run on macOS)')

        if args.tflite.exists():
            copy_file(args.tflite, ASSETS / 'macro_cnn.tflite')
        else:
            print(f'  SKIP: {args.tflite.name} not found (run on macOS/Linux)')

    # -- Copy MiDaS models if available --
    for stem in ('midas_small',):
        for ext in ('.onnx', '.mlpackage', '.tflite'):
            src = MODELS / f'{stem}{ext}'
            if src.exists() and (not args.onnx_only or ext == '.onnx'):
                copy_file(src, ASSETS / f'{stem}{ext}')
            elif args.onnx_only and ext != '.onnx':
                pass   # skip CoreML/TFLite on Windows
            elif not src.exists() and ext == '.onnx':
                print(f'  SKIP: {src.name} not found (run export_midas.py first)')

    # -- Remind about pubspec --
    print('\nNext: uncomment relevant asset entries in flutter/pubspec.yaml:')
    for f in sorted(ASSETS.glob('*')):
        name = f.name
        if name.startswith('.'):
            continue
        print(f'  - assets/models/{name}')


if __name__ == '__main__':
    main()
