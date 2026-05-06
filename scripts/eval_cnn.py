"""
Evaluate the exported macro CNN against fresh synthetic samples.

Loads models/macro_cnn.onnx (or a custom path), generates N test samples
per (macro, wear) cell using simulate_capture, runs ONNX inference, and
reports top-1 accuracy across families and wear levels.

Usage:
    python scripts/eval_cnn.py
    python scripts/eval_cnn.py --onnx models/macro_cnn.onnx --n 50

Requires: onnxruntime  (pip install onnxruntime)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codec.pattern_codec import Channels
from perception.perception import simulate_capture

_RES         = 128
_INNER_START = 32
_INNER_END   = 96
_WEAR_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
_MACROS      = list(range(16))
_MACRO_NAMES = [
    'circle','triangle','hexagon','square','star_5','star_6','star_8',
    'rosette','wave','pinwheel','diamond','octagon','petal','gear',
    'spiral','cross',
]


def _render_crop(macro: int, wear: float, rng: np.random.Generator, seed: int) -> np.ndarray:
    count  = int(rng.integers(0, 16))
    height = int(rng.integers(0, 256))
    angles = int(rng.integers(0, 256))
    micro  = int(rng.integers(0, 256))
    ch = Channels(macro=macro, count=count, height=height,
                  angles=angles, micro=micro, rs=bytes(4))
    _, obs = simulate_capture(ch, wear_factor=wear, noise_sigma_mm=0.05,
                              grid_resolution=_RES, rng_seed=seed)
    crop = obs[_INNER_START:_INNER_END, _INNER_START:_INNER_END].astype(np.float32)
    v_min, v_max = crop.min(), crop.max()
    rng_v = v_max - v_min
    if rng_v > 1e-9:
        crop = (crop - v_min) / rng_v
    return crop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx', type=Path, default=Path('models/macro_cnn.onnx'))
    ap.add_argument('--n',    type=int,  default=20,
                    help='samples per (macro, wear) cell (default 20 -> 2240 total)')
    ap.add_argument('--seed', type=int,  default=999,
                    help='RNG seed (use different seed than training to get fresh samples)')
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed. Run: pip install onnxruntime', file=sys.stderr)
        sys.exit(1)

    if not args.onnx.exists():
        print(f'ERROR: {args.onnx} not found. Train first with train_macro_cnn.py.',
              file=sys.stderr)
        sys.exit(1)

    sess = ort.InferenceSession(str(args.onnx),
                                providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(args.seed)
    total = len(_MACROS) * len(_WEAR_LEVELS) * args.n

    print(f'Model : {args.onnx}')
    print(f'Test  : {total:,} samples  ({len(_MACROS)} macros x '
          f'{len(_WEAR_LEVELS)} wear levels x {args.n}/cell)  seed={args.seed}')

    # Per-wear accuracy table
    print(f'\n{"wear":>5}  {"acc":>7}  ' + '  '.join(f'{w:.1f}' for w in _WEAR_LEVELS))
    print('-' * (14 + 5 * len(_WEAR_LEVELS)))

    # Per-macro, per-wear confusion
    correct = np.zeros((len(_MACROS), len(_WEAR_LEVELS)), dtype=int)
    total_m = np.zeros((len(_MACROS), len(_WEAR_LEVELS)), dtype=int)

    # Also track wear predictions
    wear_abs_err = np.zeros((len(_MACROS), len(_WEAR_LEVELS)))  # sum of |pred_wear - true_wear|
    outputs = sess.get_outputs()
    output_names = [o.name for o in outputs]
    has_wear = len(output_names) >= 2

    seed = args.seed * 10000
    for wi, wear in enumerate(_WEAR_LEVELS):
        for mi, macro in enumerate(_MACROS):
            for _ in range(args.n):
                crop = _render_crop(macro, wear, rng, seed)
                seed += 1
                x = crop[np.newaxis, np.newaxis, :, :]   # (1,1,64,64)
                out = sess.run(None, {input_name: x})
                logits = out[0]
                pred = int(np.argmax(logits[0]))
                total_m[mi, wi] += 1
                if pred == macro:
                    correct[mi, wi] += 1
                if has_wear:
                    pred_wear = float(np.clip(out[1].ravel()[0], 0.0, 1.0))
                    wear_abs_err[mi, wi] += abs(pred_wear - wear)

    # --- Macro accuracy table ---
    print(f'\n{"wear":>5}  {"macro_acc":>9}  bar')
    print('-' * 40)
    for wi, wear in enumerate(_WEAR_LEVELS):
        acc = correct[:, wi].sum() / total_m[:, wi].sum()
        bar = '#' * int(acc * 30)
        print(f'{wear:>5.1f}  {acc*100:>8.1f}%  |{bar:<30}|')
    print('-' * 40)
    overall = correct.sum() / total_m.sum()
    print(f'Overall macro accuracy: {overall*100:.1f}%')

    # Targets from PLAN.md
    acc_low  = correct[:, :4].sum() / total_m[:, :4].sum()   # wear 0.0-0.3
    acc_high = correct[:, 4:].sum() / total_m[:, 4:].sum()   # wear 0.4-0.6
    target_low  = 0.92
    target_high = 0.75
    print(f'\nTarget >92% at wear<=0.3: {acc_low*100:.1f}%  '
          f'{"PASS" if acc_low >= target_low else "FAIL"}')
    print(f'Target >75% at wear 0.3-0.6: {acc_high*100:.1f}%  '
          f'{"PASS" if acc_high >= target_high else "FAIL"}')

    # --- Per-macro accuracy bar chart ---
    print(f'\nPer-macro accuracy (all wear levels):')
    for mi, name in enumerate(_MACRO_NAMES):
        acc = correct[mi].sum() / total_m[mi].sum()
        bar = '#' * int(acc * 25)
        print(f'  {mi:2d} {name:10s} {acc*100:5.1f}%  |{bar:<25}|')

    # --- Wear MAE table ---
    if has_wear:
        n_total_per_wear = len(_MACROS) * args.n
        print(f'\nWear estimation MAE per wear level:')
        print(f'{"wear":>5}  {"MAE":>6}')
        print('-' * 14)
        total_mae = 0.0
        for wi, wear in enumerate(_WEAR_LEVELS):
            mae = wear_abs_err[:, wi].sum() / (n_total_per_wear)
            total_mae += mae
            bar = '#' * int((1 - mae / 0.3) * 20) if mae < 0.3 else ''
            print(f'{wear:>5.1f}  {mae:>6.4f}  |{bar:<20}|')
        print('-' * 14)
        print(f'Overall wear MAE: {total_mae/len(_WEAR_LEVELS):.4f}  '
              f'(target < 0.05 for well-trained model)')


if __name__ == '__main__':
    main()
