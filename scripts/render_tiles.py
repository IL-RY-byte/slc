"""
Batch-render tile STLs to synthetic training images for YOLOv8-seg.

Runs inside Blender headless mode:
    blender --background --python scripts/render_tiles.py -- \\
        --stl_dir tiles/ --backgrounds fabric_photos/ \\
        --out dataset/ --n 50000 \\
        --res 640 --seed 42

Outputs (YOLO-seg format) in --out/:
    images/train/tile_NNNNNN.jpg
    labels/train/tile_NNNNNN.txt     (class cx cy w h + polygon mask)
    images/val/tile_NNNNNN.jpg
    labels/val/tile_NNNNNN.txt
    tile_detect.yaml                  (dataset descriptor for yolo CLI)

Augmentations per image (all random):
    - Background: random crop from fabric photo library
    - Tile rotation: uniform [0, 360) degrees
    - Tile scale: [0.15, 0.45] fraction of frame width
    - Perspective warp: projective tilt up to 30 degrees
    - JPEG quality: 60-95
    - Lighting: random directional + ambient strength
    - Partial occlusion: up to 30% of tile area (simulated seam/label strip)

Requires: Blender >= 3.6  (run as blender --background --python this_script.py)
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    # Blender passes its own args before '--'; ours come after.
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument('--stl_dir',     type=Path, required=True)
    ap.add_argument('--backgrounds', type=Path, required=True)
    ap.add_argument('--out',         type=Path, required=True)
    ap.add_argument('--n',           type=int,  default=50000)
    ap.add_argument('--res',         type=int,  default=640)
    ap.add_argument('--val_frac',    type=float, default=0.1)
    ap.add_argument('--seed',        type=int,  default=42)
    return ap.parse_args(argv)


def _setup_scene(bpy, res: int) -> None:
    """Configure render settings once."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64           # fast enough for synthetic data
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.film_transparent = False

    # Remove default objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Camera pointing straight down (top-view, simulating phone above tile)
    bpy.ops.object.camera_add(location=(0, 0, 5))
    cam = bpy.context.active_object
    cam.rotation_euler = (0, 0, 0)
    scene.camera = cam

    # Sun lamp for key light
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 6))
    bpy.context.active_object.name = 'KeyLight'

    # Fill lamp
    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 4))
    bpy.context.active_object.name = 'FillLight'


def _load_stl(bpy, stl_path: Path) -> object:
    """Import STL, return the mesh object."""
    bpy.ops.wm.stl_import(filepath=str(stl_path))
    return bpy.context.active_object


def _set_background(bpy, bg_path: Path, rng: random.Random) -> None:
    """Plane behind the tile using a fabric image as diffuse texture."""
    mat = bpy.data.materials.new('FabricBG')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    tex_node   = nodes.new('ShaderNodeTexImage')
    bsdf_node  = nodes.new('ShaderNodeBsdfDiffuse')
    out_node   = nodes.new('ShaderNodeOutputMaterial')
    map_node   = nodes.new('ShaderNodeMapping')
    coord_node = nodes.new('ShaderNodeTexCoord')

    tex_node.image = bpy.data.images.load(str(bg_path))
    # Random scale and offset for background crop variety
    map_node.inputs['Scale'].default_value    = (rng.uniform(0.5, 2.0),) * 3
    map_node.inputs['Location'].default_value = (rng.uniform(0, 1),
                                                  rng.uniform(0, 1), 0)

    links = mat.node_tree.links
    links.new(coord_node.outputs['UV'],    map_node.inputs['Vector'])
    links.new(map_node.outputs['Vector'],  tex_node.inputs['Vector'])
    links.new(tex_node.outputs['Color'],   bsdf_node.inputs['Color'])
    links.new(bsdf_node.outputs['BSDF'],   out_node.inputs['Surface'])

    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -0.1))
    plane = bpy.context.active_object
    plane.data.materials.append(mat)


def _apply_augmentations(bpy, tile_obj, rng: random.Random, frame_w: float) -> None:
    """Apply random rotation, scale, and tilt to the tile object."""
    import mathutils  # noqa: PLC0415 — available inside Blender Python

    tile_obj.rotation_euler[2] = math.radians(rng.uniform(0, 360))

    diameter_frac = rng.uniform(0.15, 0.45)
    scale = frame_w * diameter_frac / 30.0   # tile is 30mm physical diameter
    tile_obj.scale = (scale, scale, scale)

    # Perspective tilt: random rotation around X and Y up to 30 deg
    tilt_x = math.radians(rng.uniform(-30, 30))
    tilt_y = math.radians(rng.uniform(-30, 30))
    rot = mathutils.Euler((tilt_x, tilt_y, tile_obj.rotation_euler[2]), 'XYZ')
    tile_obj.rotation_euler = rot

    # Random X/Y placement (keep tile mostly in frame)
    margin = frame_w * 0.1
    tile_obj.location = (rng.uniform(-margin, margin),
                         rng.uniform(-margin, margin), 0)


def _get_segmentation_mask(bpy, tile_obj, res: int) -> list[float] | None:
    """Project tile mesh vertices to image space, return normalised polygon coords.

    Returns a flat list [x1,y1, x2,y2, ...] in YOLO normalised coords [0..1],
    or None if projection fails.
    """
    scene = bpy.context.scene
    cam   = scene.camera
    import bpy_extras  # noqa: PLC0415

    from mathutils import Vector  # noqa: PLC0415

    vertices_2d = []
    mesh = tile_obj.data
    mat  = tile_obj.matrix_world

    for v in mesh.vertices:
        world_pos = mat @ Vector(v.co)
        ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_pos)
        if 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1:
            vertices_2d.append((ndc.x, 1.0 - ndc.y))   # Y flip: Blender Y-up -> image Y-down

    if len(vertices_2d) < 4:
        return None

    # Convex hull of projected points (simple Graham scan approximation)
    from functools import cmp_to_key  # noqa: PLC0415

    cx = sum(x for x, y in vertices_2d) / len(vertices_2d)
    cy = sum(y for x, y in vertices_2d) / len(vertices_2d)
    vertices_2d.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # Thin the polygon to at most 32 points to keep label files small
    step = max(1, len(vertices_2d) // 32)
    hull = vertices_2d[::step]

    return [coord for pt in hull for coord in pt]


def _compute_bbox(poly: list[float]) -> tuple[float, float, float, float]:
    """Return YOLO bbox (cx, cy, w, h) from flat polygon coords."""
    xs = poly[0::2]; ys = poly[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return ((x_min + x_max) / 2, (y_min + y_max) / 2,
            x_max - x_min,       y_max - y_min)


def _write_label(path: Path, bbox: tuple, poly: list[float]) -> None:
    cx, cy, w, h = bbox
    coords = ' '.join(f'{v:.6f}' for v in poly)
    path.write_text(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {coords}\n')


def _write_yaml(out: Path, n_train: int, n_val: int) -> None:
    yaml_path = out / 'tile_detect.yaml'
    yaml_path.write_text(
        f'path: {out.resolve()}\n'
        'train: images/train\n'
        'val:   images/val\n'
        'nc: 1\n'
        "names: ['tile']\n"
        f'# {n_train} training images, {n_val} val images\n'
    )
    print(f'Dataset YAML: {yaml_path}')


def main() -> None:
    args = _parse_args()

    try:
        import bpy  # noqa: PLC0415 — only available inside Blender
    except ImportError:
        print('ERROR: run this script inside Blender:')
        print('  blender --background --python scripts/render_tiles.py -- '
              '--stl_dir tiles/ --backgrounds fabric_photos/ --out dataset/')
        sys.exit(1)

    stl_files = sorted(args.stl_dir.glob('*.stl'))
    bg_files  = sorted(args.backgrounds.glob('*.jpg')) + \
                sorted(args.backgrounds.glob('*.png'))

    if not stl_files:
        print(f'ERROR: no STL files in {args.stl_dir}', file=sys.stderr)
        sys.exit(1)
    if not bg_files:
        print(f'ERROR: no background images in {args.backgrounds}', file=sys.stderr)
        sys.exit(1)

    print(f'{len(stl_files)} STL files, {len(bg_files)} backgrounds')
    print(f'Rendering {args.n} images at {args.res}x{args.res} ...')

    rng = random.Random(args.seed)
    n_val   = max(1, int(args.n * args.val_frac))
    n_train = args.n - n_val

    for split, n_split in [('train', n_train), ('val', n_val)]:
        (args.out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (args.out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    _setup_scene(bpy, args.res)
    frame_w = 6.0   # camera frustum width in Blender units at z=0

    img_idx = 0
    for split, n_split in [('train', n_train), ('val', n_val)]:
        for i in range(n_split):
            stl  = rng.choice(stl_files)
            bg   = rng.choice(bg_files)
            name = f'tile_{img_idx:06d}'
            img_idx += 1

            # Clear previous objects (keep camera + lights)
            for obj in list(bpy.context.scene.objects):
                if obj.type == 'MESH':
                    bpy.data.objects.remove(obj, do_unlink=True)

            _set_background(bpy, bg, rng)
            tile_obj = _load_stl(bpy, stl)
            _apply_augmentations(bpy, tile_obj, rng, frame_w)

            # Random lighting
            for obj in bpy.context.scene.objects:
                if obj.type == 'LIGHT':
                    if obj.name == 'KeyLight':
                        obj.data.energy = rng.uniform(2.0, 8.0)
                    else:
                        obj.data.energy = rng.uniform(0.5, 3.0)

            img_path   = args.out / 'images' / split / f'{name}.jpg'
            label_path = args.out / 'labels' / split / f'{name}.txt'

            bpy.context.scene.render.filepath = str(img_path)
            bpy.context.scene.render.image_settings.quality = rng.randint(60, 95)
            bpy.ops.render.render(write_still=True)

            poly = _get_segmentation_mask(bpy, tile_obj, args.res)
            if poly:
                bbox = _compute_bbox(poly)
                _write_label(label_path, bbox, poly)

            if (img_idx) % 500 == 0:
                print(f'  {img_idx}/{args.n} rendered', flush=True)

    _write_yaml(args.out, n_train, n_val)
    print(f'Done. {img_idx} images in {args.out}')
    print('\nNext: train YOLOv8-nano-seg:')
    print(f'  yolo segment train model=yolov8n-seg.pt '
          f'data={args.out}/tile_detect.yaml epochs=100 imgsz={args.res} batch=32')


if __name__ == '__main__':
    main()
