"""
Export trained YOLOv8-nano-seg to ONNX for ONNX Runtime Web.

Usage:
    python scripts/export_yolo_onnx.py
    python scripts/export_yolo_onnx.py --model models/tile_detector.pt --out flutter/assets/models/tile_detector.onnx

The exported ONNX is ~6 MB (FP32) and runs at ~100-200 ms/frame in WASM.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=Path('models/tile_detector.pt'),
                    help='trained .pt file (default: models/tile_detector.pt)')
    ap.add_argument('--out',   type=Path,
                    default=Path('flutter/assets/models/tile_detector.onnx'),
                    help='output .onnx path')
    ap.add_argument('--imgsz', type=int, default=640)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print('ultralytics not installed. Run: pip install ultralytics', file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f'ERROR: model not found: {args.model}', file=sys.stderr)
        print('Train first: python scripts/train_yolov8.py --data dataset/tile_detect.yaml')
        sys.exit(1)

    print(f'Exporting {args.model} -> ONNX ...')
    model = YOLO(str(args.model))
    # opset=12 is the highest version ONNX Runtime Web WASM supports well
    export_path = model.export(format='onnx', imgsz=args.imgsz, opset=12,
                               simplify=True, dynamic=False)
    export_path = Path(export_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(export_path, args.out)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f'Saved: {args.out}  ({size_mb:.1f} MB)')
    print()
    print('Next: rebuild Flutter web')
    print('  flutter build web --release --pwa-strategy=none')
    print('  Then copy ORT WASM files and serve.')


if __name__ == '__main__':
    main()
