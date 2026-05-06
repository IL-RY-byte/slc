"""
Download MiDaS-small from torch.hub and export to ONNX.

MiDaS-small is a ~15 MB monocular depth estimation model used as the
non-LiDAR fallback on devices without ARKit LiDAR (iOS) / ARCore (Android).

The exported ONNX can then be converted to:
    CoreML  (macOS) : python scripts/export_models.py --midas-only
    TFLite  (Linux) : python scripts/export_models.py --midas-only

Usage:
    python scripts/export_midas.py
    python scripts/export_midas.py --out models/midas_small.onnx

Input:  [1, 3, 256, 256] float32 NCHW  (ImageNet mean/std normalised RGB)
Output: [1, 1, 256, 256] float32       (inverse relative depth, larger = closer)

The output is NOT calibrated to absolute depth; the native plugin calibrates
using the known 30mm tile diameter from the segmentation mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('models/midas_small.onnx'),
                    help='output ONNX path (default: models/midas_small.onnx)')
    ap.add_argument('--cache', type=Path, default=None,
                    help='torch.hub cache dir (default: ~/.cache/torch/hub)')
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print('PyTorch not found. Install with: pip install torch', file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print('Downloading MiDaS-small from torch.hub (intel-isl/MiDaS) ...')
    hub_kwargs = {}
    if args.cache is not None:
        import os
        os.environ['TORCH_HOME'] = str(args.cache)

    try:
        model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small',
                               pretrained=True, trust_repo=True)
    except Exception as e:
        print(f'torch.hub.load failed: {e}', file=sys.stderr)
        print('Ensure you have internet access and git installed.', file=sys.stderr)
        sys.exit(1)

    model.eval().cpu()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'MiDaS-small loaded: {n_params:,} parameters')

    # MiDaS_small outputs [B, H, W] (no channel dim).
    # Wrap to emit [B, 1, H, W] so CoreML / TFLite plugins can use consistent indexing.
    import torch.nn as nn
    class _MiDaSWrapper(nn.Module):
        def __init__(self, base): super().__init__(); self.base = base
        def forward(self, x):
            out = self.base(x)
            return out.unsqueeze(1) if out.dim() == 3 else out
    wrapped = _MiDaSWrapper(model)

    # Dummy input: batch=1, RGB, 256x256, ImageNet-normalised
    dummy = torch.zeros(1, 3, 256, 256)

    # Verify output shape before export
    with torch.no_grad():
        test_out = wrapped(dummy)
    print(f'Output shape check: {tuple(test_out.shape)}  (expected [1, 1, 256, 256])')

    print(f'Exporting to {args.out} ...')
    try:
        torch.onnx.export(
            wrapped, dummy, str(args.out),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
            opset_version=18,
            dynamo=False,
        )
    except Exception as e:
        print(f'ONNX export failed: {e}', file=sys.stderr)
        sys.exit(1)

    size_mb = args.out.stat().st_size / 1e6
    print(f'Saved: {args.out}  ({size_mb:.1f} MB)')

    # Quick sanity check with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.out), providers=['CPUExecutionProvider'])
        x = np.zeros((1, 3, 256, 256), dtype=np.float32)
        out = sess.run(None, {'input': x})
        print(f'Sanity check: output shape = {out[0].shape}  (expected [1, 1, 256, 256])')
    except ImportError:
        print('onnxruntime not installed - skipping sanity check')

    print('\nNext steps:')
    print('  macOS:  python scripts/export_models.py --midas-only')
    print('           -> models/midas_small.mlpackage')
    print('  Linux:  python scripts/export_models.py --midas-only')
    print('           -> models/midas_small.tflite')
    print('  Deploy: python scripts/deploy_models.py --onnx-only')
    print('           -> flutter/assets/models/midas_small.onnx')


if __name__ == '__main__':
    main()
