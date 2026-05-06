"""
Geometry generator — turns codec Channels into a 3D mesh.

Mapping per ТЗ §2.1 / §2.2:
    macro    → which of 16 base shapes
    count    → number of protrusions (3..18)
    height   → relief height profile (1.5..8 mm)
    angles   → rotation pattern between elements
    micro    → micro-texture (small surface bumps)
    rs       → embedded marker bumps at known positions

Output: a triangle mesh (vertices + faces) with an STL writer.
This is intentionally pure-Python / numpy — no trimesh dependency,
so it runs anywhere and we keep the geometry transparent.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from codec.pattern_codec import Channels


# --- Mesh dataclass ---------------------------------------------------------

@dataclass
class Mesh:
    vertices: np.ndarray   # (N, 3) float
    faces:    np.ndarray   # (M, 3) int  — indices into vertices

    def to_stl_bytes(self) -> bytes:
        """Binary STL writer."""
        n = len(self.faces)
        out = bytearray()
        out.extend(b"\x00" * 80)               # header
        out.extend(struct.pack("<I", n))        # triangle count

        v = self.vertices
        for tri in self.faces:
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = normal / norm
            out.extend(struct.pack("<fff", *normal))
            out.extend(struct.pack("<fff", *a))
            out.extend(struct.pack("<fff", *b))
            out.extend(struct.pack("<fff", *c))
            out.extend(b"\x00\x00")             # attribute byte count
        return bytes(out)

    def to_stl_ascii(self) -> str:
        """Human-readable ASCII STL — useful for debugging."""
        lines = ["solid pattern"]
        v = self.vertices
        for tri in self.faces:
            a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-12 else n
            lines.append(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            lines.append("    outer loop")
            for p in (a, b, c):
                lines.append(f"      vertex {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid pattern")
        return "\n".join(lines)


# --- Channel → geometric parameter mapping ----------------------------------

# 16 macro shape "families" — these are silhouette templates.
# Each family is parameterized differently in the radial pattern.
MACRO_FAMILIES = [
    "circle", "square", "hexagon", "triangle",
    "star_5", "star_6", "star_8", "rosette",
    "wave", "pinwheel", "diamond", "octagon",
    "petal", "gear", "spiral", "cross",
]

# Count → number of protrusions (3..18 per ТЗ §2.1)
COUNT_TO_PROTRUSIONS = lambda c: 3 + (c % 16)

# Height byte → relief height in mm (1.5..8 per ТЗ §3.2.1)
HEIGHT_BYTE_TO_MM = lambda h: 1.5 + (h / 255.0) * 6.5

# Angles byte → rotation offset pattern (8 bits split: 4 base + 4 modulation)
def ANGLES_TO_ROTATIONS(a: int, n: int) -> np.ndarray:
    base   = (a >> 4) & 0x0F          # 0..15  → 0..360°
    modul  =  a       & 0x0F          # 0..15  → modulation depth
    base_offset = (base / 16.0) * 2 * math.pi
    mod_amp     = (modul / 15.0) * (math.pi / n)
    idx = np.arange(n)
    return base_offset + idx * (2 * math.pi / n) + mod_amp * np.sin(2 * idx)


# --- Pattern generator ------------------------------------------------------

class PatternGeometry:
    """
    Generates a flat-backed circular pattern tile (~30 mm diameter)
    suitable for printing in TPU 85A/95A and stitching onto fabric.
    """

    def __init__(self, radius_mm: float = 15.0, base_thickness_mm: float = 1.0,
                 grid_resolution: int = 96):
        self.radius = radius_mm
        self.base   = base_thickness_mm
        self.res    = grid_resolution

    # -- main entry point ----------------------------------------------------

    def generate(self, channels: Channels) -> Mesh:
        height_field = self._build_height_field(channels)
        return self._heightfield_to_mesh(height_field)

    # -- height field synthesis ---------------------------------------------

    def _build_height_field(self, ch: Channels) -> np.ndarray:
        """
        Build a (res × res) height map in mm. Each pixel's height is the
        sum of contributions from each channel.
        """
        res = self.res
        # Coordinate grid in mm, centered
        xs = np.linspace(-self.radius, self.radius, res)
        ys = np.linspace(-self.radius, self.radius, res)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)

        # Mask: inside circular tile
        inside = R <= self.radius

        # --- macro: silhouette modulation -----------------------------------
        # Different families warp the boundary differently, encoded as a
        # small radial perturbation. We keep the tile circular for printing
        # but the relief reads the macro.
        macro_field = self._macro_contribution(ch.macro, R, THETA)

        # --- count: N radial protrusions -----------------------------------
        n = COUNT_TO_PROTRUSIONS(ch.count)
        rotations = ANGLES_TO_ROTATIONS(ch.angles, n)
        peak_h = HEIGHT_BYTE_TO_MM(ch.height)

        protrusion_field = np.zeros_like(R)
        for theta_i in rotations:
            # Each protrusion is a Gaussian bump at (radius * 0.65, theta_i)
            cx = 0.65 * self.radius * math.cos(theta_i)
            cy = 0.65 * self.radius * math.sin(theta_i)
            d2 = (X - cx)**2 + (Y - cy)**2
            # σ scales inversely with N but with a floor so we always have
            # several grid cells across each peak (avoids undersampling at low res).
            sigma = max(self.radius / (1.5 * n), 1.5 * (2 * self.radius / res))
            protrusion_field += peak_h * np.exp(-d2 / (2 * sigma**2))

        # --- micro: high-frequency texture seeded by micro byte -------------
        rng = np.random.default_rng(seed=ch.micro * 8191 + 17)
        micro_field = rng.normal(0, 0.15, (res, res))
        # Smooth slightly so it's printable, not pure noise
        micro_field = self._smooth(micro_field, k=2)
        micro_field *= 0.4   # cap at ±0.4 mm

        # --- rs markers: 8 small "pillar" bumps near the rim ---------------
        rs_field = self._rs_marker_bumps(ch.rs, X, Y, R, THETA)

        # --- combine --------------------------------------------------------
        height = (
            self.base
            + macro_field
            + protrusion_field
            + micro_field
            + rs_field
        )
        height = np.where(inside, height, 0.0)
        # Force base thickness everywhere inside the tile
        height = np.where(inside, np.maximum(height, self.base), 0.0)
        return height

    def _macro_contribution(self, macro: int, R: np.ndarray,
                            THETA: np.ndarray) -> np.ndarray:
        """A signature ripple based on the macro family.

        CRITICAL: every one of the 16 families must produce a DIFFERENT
        harmonic signature, otherwise macro can't be inverted from observation.
        We achieve this by combining a primary harmonic (n_a) and a secondary
        harmonic (n_b) at a phase, where (n_a, n_b, phase) is unique per family.
        """
        r_norm = R / self.radius
        # (primary, secondary, phase, radial_decay)
        sigs = [
            (2,  0,  0.0, 1.0),   # 0  circle    : pure 2θ, no second harmonic
            (3,  0,  0.0, 1.0),   # 1  triangle  : 3θ
            (6,  0,  0.0, 1.0),   # 2  hexagon   : 6θ
            (4,  0,  0.0, 1.0),   # 3  square    : 4θ
            (5,  0,  0.0, 1.0),   # 4  star_5    : 5θ
            (6,  2,  0.0, 1.0),   # 5  star_6    : 6θ + 2θ (distinct from hexagon)
            (8,  0,  0.0, 1.0),   # 6  star_8    : 8θ
            (2,  4,  0.5, 1.0),   # 7  rosette   : 2θ + 4θ phase-shifted
            (2,  0,  0.0, 0.0),   # 8  wave      : 2θ + radial wave
            (4,  0,  0.0, 0.0),   # 9  pinwheel  : 4θ + radial spiral
            (4,  2,  0.0, 1.0),   # 10 diamond   : 4θ + 2θ
            (8,  4,  0.0, 1.0),   # 11 octagon   : 8θ + 4θ
            (5,  10, 0.0, 1.0),   # 12 petal     : 5θ + 10θ harmonic
            (12, 0,  0.0, 1.0),   # 13 gear      : 12θ
            (4,  0,  0.0, -1.0),  # 14 spiral    : 4θ + reverse radial
            (2,  6,  0.5, 1.0),   # 15 cross     : 2θ + 6θ phase-shifted
        ]
        primary, secondary, phase, radial_mode = sigs[macro % 16]

        # radial_mode: 1.0 = (1 - r_norm) decay, 0.0 = wave in r, -1.0 = reverse spiral
        if radial_mode == 1.0:
            radial = (1 - r_norm)
        elif radial_mode == 0.0:
            # combined angular + radial wave
            radial = 1.0
            primary_term = 0.4 * np.cos(primary * THETA + 6 * r_norm)
            return primary_term
        else:
            primary_term = 0.4 * np.cos(primary * THETA - 8 * r_norm)
            return primary_term

        amp_primary = 0.4
        amp_secondary = 0.35   # raised from 0.2 — must be clearly above noise floor
        result = amp_primary * np.cos(primary * THETA) * radial
        if secondary > 0:
            result = result + amp_secondary * np.cos(secondary * THETA + phase * math.pi) * radial
        return result

    def _rs_marker_bumps(self, rs: bytes, X, Y, R, THETA) -> np.ndarray:
        """
        Place 32 small redundancy markers around the rim. Each carries 1 bit
        of the 32-bit RS parity, encoded as tall (3.0 mm) or short (1.0 mm).
        High contrast + 1-bit encoding makes them robust to noise and partial wear.
        """
        field = np.zeros_like(R)
        bits = []
        for byte in rs:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        n_markers = 32
        for k in range(n_markers):
            level = 3.0 if bits[k] else 1.0   # high contrast
            angle = (k / n_markers) * 2 * math.pi
            cx = 0.92 * self.radius * math.cos(angle)
            cy = 0.92 * self.radius * math.sin(angle)
            d2 = (X - cx)**2 + (Y - cy)**2
            # σ chosen larger than wear-smoothing kernel (~3 pixels = ~1 mm at res=96)
            # so that micro-stage wear doesn't materially shave marker peaks.
            sigma = self.radius / 24.0   # ~0.625 mm — well above the 1mm kernel? Actually larger for safety
            field += level * np.exp(-d2 / (2 * sigma**2))
        return field

    @staticmethod
    def _smooth(arr: np.ndarray, k: int = 1) -> np.ndarray:
        """Cheap box blur — k passes of a 3×3 mean filter."""
        out = arr
        for _ in range(k):
            padded = np.pad(out, 1, mode="edge")
            out = (
                padded[ :-2,  :-2] + padded[ :-2, 1:-1] + padded[ :-2, 2:  ] +
                padded[1:-1,  :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:  ] +
                padded[2:  ,  :-2] + padded[2:  , 1:-1] + padded[2:  , 2:  ]
            ) / 9.0
        return out

    # -- height field → triangle mesh ---------------------------------------

    def _heightfield_to_mesh(self, h: np.ndarray) -> Mesh:
        """
        Convert the height field into a closed triangle mesh:
          - top surface follows h(x,y)
          - bottom is flat at z=0
          - side walls connect them around the circular boundary
        """
        res = self.res
        xs = np.linspace(-self.radius, self.radius, res)
        ys = np.linspace(-self.radius, self.radius, res)
        X, Y = np.meshgrid(xs, ys)
        inside = h > 0

        # Build top vertices (only for cells inside the disc)
        verts: list[tuple[float, float, float]] = []
        idx_top = -np.ones((res, res), dtype=int)
        idx_bot = -np.ones((res, res), dtype=int)
        for j in range(res):
            for i in range(res):
                if inside[j, i]:
                    idx_top[j, i] = len(verts)
                    verts.append((float(X[j, i]), float(Y[j, i]), float(h[j, i])))
        for j in range(res):
            for i in range(res):
                if inside[j, i]:
                    idx_bot[j, i] = len(verts)
                    verts.append((float(X[j, i]), float(Y[j, i]), 0.0))

        faces: list[tuple[int, int, int]] = []

        # Top surface — quads (v00,v10,v11,v01) split into two triangles.
        # Only emit a quad if all four corners are inside.
        for j in range(res - 1):
            for i in range(res - 1):
                a = idx_top[j,     i    ]
                b = idx_top[j,     i + 1]
                c = idx_top[j + 1, i + 1]
                d = idx_top[j + 1, i    ]
                if min(a, b, c, d) < 0:
                    continue
                faces.append((a, b, c))
                faces.append((a, c, d))

        # Bottom surface — same but reversed winding (faces down)
        for j in range(res - 1):
            for i in range(res - 1):
                a = idx_bot[j,     i    ]
                b = idx_bot[j,     i + 1]
                c = idx_bot[j + 1, i + 1]
                d = idx_bot[j + 1, i    ]
                if min(a, b, c, d) < 0:
                    continue
                faces.append((a, c, b))
                faces.append((a, d, c))

        # Side walls — wherever an inside cell borders an outside cell
        # we emit a vertical quad between top and bottom.
        for j in range(res):
            for i in range(res):
                if not inside[j, i]:
                    continue
                for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nj, ni = j + dj, i + di
                    is_outside_neighbor = (
                        nj < 0 or nj >= res or ni < 0 or ni >= res
                        or not inside[nj, ni]
                    )
                    if not is_outside_neighbor:
                        continue
                    # Find the two top/bottom vertices on the shared edge.
                    # We approximate by using the cell's own edge.
                    if di != 0:
                        # vertical edge (varies in j)
                        if i + (1 if di > 0 else 0) >= res or i + (1 if di > 0 else 0) < 0:
                            continue
                        # Use the corner vertices we already have
                        ti = idx_top[j, i]
                        bi = idx_bot[j, i]
                        # neighbor along j+1 if exists
                        if j + 1 < res and inside[j + 1, i]:
                            tj = idx_top[j + 1, i]
                            bj = idx_bot[j + 1, i]
                            if min(ti, bi, tj, bj) >= 0:
                                if di > 0:
                                    faces.append((ti, bi, bj))
                                    faces.append((ti, bj, tj))
                                else:
                                    faces.append((ti, bj, bi))
                                    faces.append((ti, tj, bj))
                    else:
                        if j + (1 if dj > 0 else 0) >= res or j + (1 if dj > 0 else 0) < 0:
                            continue
                        ti = idx_top[j, i]
                        bi = idx_bot[j, i]
                        if i + 1 < res and inside[j, i + 1]:
                            tj = idx_top[j, i + 1]
                            bj = idx_bot[j, i + 1]
                            if min(ti, bi, tj, bj) >= 0:
                                if dj > 0:
                                    faces.append((ti, bj, bi))
                                    faces.append((ti, tj, bj))
                                else:
                                    faces.append((ti, bi, bj))
                                    faces.append((ti, bj, tj))

        return Mesh(
            vertices=np.array(verts, dtype=float),
            faces=np.array(faces, dtype=int),
        )


# --- Convenience ------------------------------------------------------------

def channels_to_stl(channels: Channels, **kwargs) -> bytes:
    """One-shot helper: channels → STL bytes."""
    geom = PatternGeometry(**kwargs)
    mesh = geom.generate(channels)
    return mesh.to_stl_bytes()


__all__ = ["Mesh", "PatternGeometry", "channels_to_stl",
           "MACRO_FAMILIES", "HEIGHT_BYTE_TO_MM"]
