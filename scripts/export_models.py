"""
Export trained macro CNN and wear estimator to CoreML and TFLite INT8.

Platform requirements:
    CoreML export (--no-tflite): macOS only (coremltools only runs on macOS)
    TFLite export (--no-coreml): Linux/macOS (tensorflow + onnx2tf)
    ONNX input (from train_macro_cnn.py): any platform

Requires:
    pip install coremltools onnx onnx2tf tensorflow  (macOS/Linux)
    pip install onnx                                 (Windows, for validation only)

Usage:
    # Export both CoreML and TFLite from the best checkpoint (macOS/Linux)
    python scripts/export_models.py --onnx models/macro_cnn.onnx

    # CoreML only (macOS, faster, iOS-first)
    python scripts/export_models.py --onnx models/macro_cnn.onnx --no-tflite

Outputs (in models/):
    macro_cnn.mlpackage     CoreML package (iOS)
    macro_cnn.tflite        TFLite INT8 (Android)

The exported models accept input shape (1, 1, 64, 64) float32 and produce:
    macro_logits   float32 (1, 16)   - softmax -> class probabilities
    wear_estimate  float32 (1,)      - sigmoid output in [0, 1]

Integration:
    In the Flutter app, the platform channel passes a flattened 64x64 float32
    array. The iOS plugin wraps the CoreML call; the Android plugin wraps TFLite.
    Results are returned as a map {macro_probs: [16 floats], wear: float}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_coreml(onnx_path: Path, out_dir: Path,
                  out_stem: str = 'macro_cnn',
                  input_name: str = 'height_field',
                  output_names: list | None = None,
                  input_shape: tuple = (1, 1, 64, 64)) -> Path | None:
    try:
        import coremltools as ct
    except ImportError:
        print('coremltools not installed - skipping CoreML export', file=sys.stderr)
        print('Install with: pip install coremltools', file=sys.stderr)
        return None

    out_path = out_dir / f'{out_stem}.mlpackage'
    print(f'Exporting CoreML -> {out_path} ...')

    if output_names is None:
        output_names = ['macro_logits', 'wear_estimate']

    model = ct.convert(
        str(onnx_path),
        inputs=[ct.TensorType(name=input_name, shape=input_shape,
                              dtype=ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32)],
        outputs=[ct.TensorType(name=n) for n in output_names],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        convert_to='mlprogram',
    )
    if out_stem == 'macro_cnn':
        model.short_description = 'SLC macro CNN + wear estimator (Enhancement 1+2)'
    elif out_stem == 'midas_small':
        model.short_description = 'MiDaS-small monocular depth estimator (non-LiDAR fallback)'
    model.save(str(out_path))
    print(f'CoreML saved: {out_path}')
    return out_path


def export_tflite(onnx_path: Path, out_dir: Path,
                  out_stem: str = 'macro_cnn') -> Path | None:
    try:
        import onnx2tf
    except ImportError:
        print('onnx2tf not installed - skipping TFLite export', file=sys.stderr)
        print('Install with: pip install onnx2tf', file=sys.stderr)
        return None

    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    out_path = out_dir / f'{out_stem}.tflite'
    print(f'Converting ONNX -> TF SavedModel via onnx2tf ...')
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tmp),
            non_verbose=True,
        )
        sm_dirs = list(tmp.glob('*/saved_model'))
        sm_path = sm_dirs[0] if sm_dirs else tmp

        print(f'Converting TF SavedModel -> TFLite INT8 ...')
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(str(sm_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type  = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()
        out_path.write_bytes(tflite_model)
        print(f'TFLite saved: {out_path}  ({len(tflite_model)/1e6:.1f} MB)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx',       type=Path, default=Path('models/macro_cnn.onnx'),
                    help='macro CNN ONNX from train_macro_cnn.py')
    ap.add_argument('--midas-onnx', type=Path, default=Path('models/midas_small.onnx'),
                    help='MiDaS-small ONNX from export_midas.py')
    ap.add_argument('--out',        type=Path, default=Path('models'),
                    help='output directory')
    ap.add_argument('--no-tflite',  action='store_true',
                    help='skip TFLite export (CoreML only)')
    ap.add_argument('--no-coreml',  action='store_true',
                    help='skip CoreML export (TFLite only)')
    ap.add_argument('--midas-only', action='store_true',
                    help='export only the MiDaS model, skip macro CNN')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.midas_only:
        if not args.onnx.exists():
            print(f'ERROR: {args.onnx} not found. '
                  f'Run train_macro_cnn.py first.', file=sys.stderr)
            sys.exit(1)
        if not args.no_coreml:
            export_coreml(args.onnx, args.out)
        if not args.no_tflite:
            export_tflite(args.onnx, args.out)

    if args.midas_onnx.exists():
        print(f'\nExporting MiDaS-small from {args.midas_onnx} ...')
        if not args.no_coreml:
            export_coreml(args.midas_onnx, args.out,
                          out_stem='midas_small',
                          input_name='input', output_names=['output'],
                          input_shape=(1, 3, 256, 256))
        if not args.no_tflite:
            export_tflite(args.midas_onnx, args.out, out_stem='midas_small')
    elif args.midas_only:
        print(f'ERROR: {args.midas_onnx} not found. '
              f'Run export_midas.py first.', file=sys.stderr)
        sys.exit(1)
    else:
        print(f'SKIP: {args.midas_onnx} not found '
              f'(run export_midas.py to generate it)')

    print('\nExport complete. Model files in models/:')
    for ext in ('*.mlpackage', '*.tflite'):
        for f in args.out.glob(ext):
            size = sum(ff.stat().st_size for ff in f.rglob('*') if ff.is_file()) \
                   if f.is_dir() else f.stat().st_size
            print(f'  {f.name}  ({size/1e6:.1f} MB)')

    print('\nThen run: python scripts/deploy_models.py')


if __name__ == '__main__':
    main()
