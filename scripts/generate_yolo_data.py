"""
Generate synthetic YOLO-seg training images for tile detection.

No Blender required. Uses PIL + NumPy to shade height fields from
simulate_capture() and composite them onto procedural fabric backgrounds.

Usage:
    python scripts/generate_yolo_data.py --out dataset/ --n 8000
    python scripts/generate_yolo_data.py --out dataset/ --n 200 --quick

Outputs (YOLO-seg format):
    dataset/images/train/  .jpg
    dataset/images/val/    .jpg
    dataset/labels/train/  .txt  (class cx cy w h x1 y1 ... all normalised)
    dataset/labels/val/    .txt
    dataset/tile_detect.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageFilter
except ImportError:
    print("Pillow required: pip install pillow", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codec.pattern_codec import Channels, PatternCodec
from perception.perception import simulate_capture


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE       = 640
TILE_SCALE_MIN = 0.18   # tile diameter / img width
TILE_SCALE_MAX = 0.42
N_MASK_PTS     = 36     # polygon approximation of disc boundary
FIELD_RES      = 128    # simulate_capture resolution
VAL_FRAC       = 0.10


# ---------------------------------------------------------------------------
# Background generation
# ---------------------------------------------------------------------------

def _make_fabric_background(rng: random.Random, size: int) -> np.ndarray:
    """
    Produce a (size, size, 3) uint8 array simulating fabric.
    Cycles through several procedural modes so the detector sees variety.
    """
    mode = rng.randint(0, 4)
    base_color = np.array([rng.randint(40, 220) for _ in range(3)], dtype=np.float32)
    arr = np.zeros((size, size, 3), dtype=np.float32)

    if mode == 0:
        # Plain weave: horizontal + vertical stripes with slight noise
        freq = rng.randint(3, 12)
        xs = np.arange(size)
        stripe_h = np.sin(xs * math.pi * 2 * freq / size) * 15
        stripe_v = np.sin(xs * math.pi * 2 * freq / size) * 15
        arr[:, :] = base_color
        arr += stripe_h[np.newaxis, :, np.newaxis]
        arr += stripe_v[:, np.newaxis, np.newaxis]

    elif mode == 1:
        # Random noise (jersey-like)
        noise = np.random.RandomState(rng.randint(0, 99999)).normal(0, 18, (size, size, 3))
        arr[:] = base_color + noise

    elif mode == 2:
        # Coarse twill — diagonal lines
        freq = rng.randint(4, 10)
        diag = np.fromfunction(
            lambda y, x: np.sin((y + x) * math.pi * 2 * freq / size) * 20,
            (size, size)
        )
        arr[:, :] = base_color
        arr += diag[:, :, np.newaxis]

    elif mode == 3:
        # Gradient (shadows / folds in fabric)
        gy = np.linspace(-30, 30, size)
        arr[:, :] = base_color + gy[:, np.newaxis, np.newaxis]

    else:
        # Checkerboard (patchwork / plaid)
        cell = rng.randint(10, 40)
        checker = (
            np.fromfunction(lambda y, x: ((y // cell + x // cell) % 2).astype(float), (size, size))
            * 40 - 20
        )
        arr[:, :] = base_color + checker[:, :, np.newaxis]

    arr += np.random.RandomState(rng.randint(0, 99999)).normal(0, 6, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------

def _shade_height_field(observed: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) float64 height field (mm) to a (H, W, 4) RGBA uint8.
    Uses simple Phong shading; alpha = 0 where height == 0.
    """
    h = observed.astype(np.float32)
    inside = h > 0.01

    # Normalise relief for shading
    h_norm = np.zeros_like(h)
    if inside.any():
        v_min = h[inside].min()
        v_max = h[inside].max()
        rng = v_max - v_min
        if rng > 1e-9:
            h_norm = np.where(inside, (h - v_min) / rng, 0.0)

    # Surface normals via finite differences
    dy, dx = np.gradient(h_norm * 6.0)
    nz = np.full_like(h_norm, 0.5)
    nlen = np.sqrt(dx**2 + dy**2 + nz**2) + 1e-9
    nx, ny, nz_n = -dx / nlen, -dy / nlen, nz / nlen

    # Light direction (angled ~30 deg from above-right)
    lx, ly, lz = 0.45, 0.25, 1.0
    llen = math.sqrt(lx**2 + ly**2 + lz**2)
    lx, ly, lz = lx / llen, ly / llen, lz / llen

    diffuse = np.clip(nx * lx + ny * ly + nz_n * lz, 0.0, 1.0)
    ambient = 0.32
    intensity = np.clip(ambient + (1.0 - ambient) * diffuse, 0.0, 1.0)

    # Off-white thermoplastic appearance
    r = np.where(inside, intensity * 228, 0).astype(np.uint8)
    g = np.where(inside, intensity * 218, 0).astype(np.uint8)
    b = np.where(inside, intensity * 205, 0).astype(np.uint8)
    a = np.where(inside, 255, 0).astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------

def _perspective_warp(img_pil: Image.Image,
                      rng: random.Random,
                      max_tilt: float = 0.12) -> Image.Image:
    """
    Apply a subtle 8-point projective transform to simulate non-perpendicular
    camera angle. max_tilt controls magnitude of corner jitter as fraction of size.
    """
    w, h = img_pil.size
    jitter = max_tilt * min(w, h)

    def j() -> float:
        return rng.uniform(-jitter, jitter)

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(j(), j()), (w + j(), j()), (w + j(), h + j()), (j(), h + j())]

    # PIL transform expects 8 coefficients a..h for:
    # X = (a*x + b*y + c) / (g*x + h*y + 1)
    # Y = (d*x + e*y + f) / (g*x + h*y + 1)
    # We use the inverse: map dst -> src.
    def _solve_perspective(src, dst):
        # Build matrix equation (8x8)
        A = []
        b_vec = []
        for (x, y), (X, Y) in zip(dst, src):
            A.append([x, y, 1, 0, 0, 0, -X*x, -X*y])
            b_vec.append(X)
            A.append([0, 0, 0, x, y, 1, -Y*x, -Y*y])
            b_vec.append(Y)
        A = np.array(A, dtype=np.float64)
        b_arr = np.array(b_vec, dtype=np.float64)
        try:
            coeffs = np.linalg.solve(A, b_arr)
        except np.linalg.LinAlgError:
            return None
        return tuple(coeffs)

    coeffs = _solve_perspective(src, dst)
    if coeffs is None:
        return img_pil
    return img_pil.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


# ---------------------------------------------------------------------------
# Single image generation
# ---------------------------------------------------------------------------

def _gen_one(idx: int, out_dir: Path, split: str, rng: random.Random) -> None:
    # 1. Random tile channels
    macro  = rng.randint(0, 15)
    count  = rng.randint(0, 15)
    height = rng.randint(0, 255)
    angles = rng.randint(0, 255)
    micro  = rng.randint(0, 255)
    wear   = rng.uniform(0.0, 0.5)
    seed   = rng.randint(0, 2**31)

    rs_placeholder = bytes(4)
    ch = Channels(macro=macro, count=count, height=height,
                  angles=angles, micro=micro, rs=rs_placeholder)
    try:
        _, observed = simulate_capture(
            ch, wear_factor=wear, noise_sigma_mm=0.06,
            grid_resolution=FIELD_RES, rng_seed=seed,
        )
    except Exception:
        return

    # 2. Shade to RGBA
    tile_rgba = _shade_height_field(observed)           # (H, W, 4)
    tile_pil  = Image.fromarray(tile_rgba, mode='RGBA')

    # Optional slight blur to reduce aliasing
    if rng.random() < 0.4:
        tile_pil = tile_pil.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.7)))

    # 3. Background
    bg_arr = _make_fabric_background(rng, IMG_SIZE)
    bg     = Image.fromarray(bg_arr, mode='RGB').convert('RGBA')

    # 4. Random scale + rotation + position
    tile_diam = rng.uniform(TILE_SCALE_MIN, TILE_SCALE_MAX) * IMG_SIZE
    tile_pil  = tile_pil.resize((int(tile_diam), int(tile_diam)), Image.LANCZOS)

    angle = rng.uniform(0, 360)
    tile_pil = tile_pil.rotate(angle, expand=True)

    # After rotation, size may have increased; get current size
    tw, th = tile_pil.size

    margin = int(max(tw, th) * 0.05 + 5)
    max_x  = max(0, IMG_SIZE - tw - margin)
    max_y  = max(0, IMG_SIZE - th - margin)
    px = rng.randint(margin, max(margin + 1, max_x))
    py = rng.randint(margin, max(margin + 1, max_y))

    # 5. Perspective warp on the background (whole frame)
    bg = _perspective_warp(bg, rng, max_tilt=0.04)

    # 6. Composite tile onto background
    composite = bg.copy()
    composite.paste(tile_pil, (px, py), mask=tile_pil.split()[3])
    final = composite.convert('RGB')

    # Optional: random JPEG-like blur on whole frame
    if rng.random() < 0.3:
        final = final.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.5)))

    # 7. Compute disc mask in YOLO format
    # The disc center in field coordinates is (FIELD_RES/2, FIELD_RES/2)
    # with radius ≈ FIELD_RES/2 * 0.92 (inner disc — see PatternGeometry)
    # After resizing to tile_diam and rotating, the disc is approximately
    # inscribed in the bounding box. We approximate with a circle polygon.

    # The tile_pil alpha channel IS the disc mask; compute tighter bounding box
    # from the alpha channel to avoid using the expanded-by-rotation bounding box.
    alpha = np.array(tile_pil.split()[3])           # (th, tw) uint8
    ys, xs = np.where(alpha > 128)
    if len(xs) == 0:
        return

    # Tight disc center and radius from alpha mask
    cx_tile = (xs.min() + xs.max()) / 2.0
    cy_tile = (ys.min() + ys.max()) / 2.0
    radius_px = max((xs.max() - xs.min()) / 2.0, (ys.max() - ys.min()) / 2.0)

    # Map to composite image coordinates
    cx_img = px + cx_tile
    cy_img = py + cy_tile

    # Polygon mask (circle approximation)
    poly_pts: list[float] = []
    for k in range(N_MASK_PTS):
        angle_k = 2 * math.pi * k / N_MASK_PTS
        px_k = (cx_img + radius_px * math.cos(angle_k)) / IMG_SIZE
        py_k = (cy_img + radius_px * math.sin(angle_k)) / IMG_SIZE
        poly_pts.extend([max(0.0, min(1.0, px_k)), max(0.0, min(1.0, py_k))])

    # YOLO bbox (normalised)
    x_min = max(0.0, (cx_img - radius_px) / IMG_SIZE)
    y_min = max(0.0, (cy_img - radius_px) / IMG_SIZE)
    x_max = min(1.0, (cx_img + radius_px) / IMG_SIZE)
    y_max = min(1.0, (cy_img + radius_px) / IMG_SIZE)
    bw    = x_max - x_min
    bh    = y_max - y_min
    bcx   = x_min + bw / 2
    bcy   = y_min + bh / 2

    if bw < 0.05 or bh < 0.05:
        return

    # 8. Save
    name = f"tile_{idx:06d}"
    img_path = out_dir / 'images' / split / f"{name}.jpg"
    lbl_path = out_dir / 'labels' / split / f"{name}.txt"

    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)

    final.save(str(img_path), 'JPEG',
               quality=rng.randint(75, 95), optimize=True)

    poly_str = ' '.join(f"{v:.6f}" for v in poly_pts)
    with open(lbl_path, 'w') as f:
        f.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f} {poly_str}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out',    type=Path, default=Path('dataset'),
                    help='output directory (default: dataset/)')
    ap.add_argument('--n',      type=int,  default=8000,
                    help='total training images (default: 8000)')
    ap.add_argument('--seed',   type=int,  default=42)
    ap.add_argument('--quick',  action='store_true',
                    help='generate only 300 images for a smoke test')
    args = ap.parse_args()

    n_total = 300 if args.quick else args.n
    n_val   = max(50, int(n_total * VAL_FRAC))
    n_train = n_total - n_val

    rng = random.Random(args.seed)

    print(f"Generating {n_train} train + {n_val} val images -> {args.out}")
    print(f"IMG_SIZE={IMG_SIZE}  FIELD_RES={FIELD_RES}  N_MASK_PTS={N_MASK_PTS}")

    for i in range(n_train):
        _gen_one(i, args.out, 'train', rng)
        if (i + 1) % 200 == 0:
            print(f"  train {i + 1}/{n_train} ...", flush=True)

    for i in range(n_val):
        _gen_one(n_train + i, args.out, 'val', rng)
        if (i + 1) % 50 == 0:
            print(f"  val {i + 1}/{n_val} ...", flush=True)

    # Write dataset YAML
    yaml_path = args.out / 'tile_detect.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {args.out.resolve().as_posix()}
train: images/train
val:   images/val

nc: 1
names: ['tile']
""")
    print(f"\nDataset YAML: {yaml_path}")
    print(f"Train: {n_train} images  Val: {n_val} images")
    print(f"\nNext step:")
    print(f"  python scripts/train_yolov8.py --data {yaml_path} --epochs 80 --export")


if __name__ == '__main__':
    main()
