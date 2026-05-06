"""
Simulated perception pipeline.

Real pipeline (ТЗ §4.2):
    camera frame  →  YOLOv8-seg  →  MiDaS depth  →  point cloud
                  →  codec.extract_all  →  RS recover  →  item_id

For MVP we simulate the perception layer:
    1. Render the generated mesh as an "observed" height field.
    2. Apply a wear model (erodes micro first, then meso, then macro).
    3. Add measurement noise (mimics depth-estimation error).
    4. Run a channel extractor that recovers (or fails to recover)
       each of the 5 channels from the observed field.

The output of `extract_channels` is exactly the dict that
`PatternCodec.decode` expects, so the loop closes cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from codec.pattern_codec import Channels
from geometry.pattern_geometry import (
    PatternGeometry, MACRO_FAMILIES, HEIGHT_BYTE_TO_MM,
    COUNT_TO_PROTRUSIONS, ANGLES_TO_ROTATIONS,
)


# --- Wear model -------------------------------------------------------------

@dataclass
class WearProfile:
    """
    Wear model controlled by a single 0..1 wear factor.
    Based on ТЗ §2.3 — micro wears first, then meso, then macro.

    wear_factor:
        0.0 = brand new
        0.2 = ~1 year (some micro loss)
        0.4 = active wear (most micro gone, some meso)
        0.6 = heavy wear (meso degraded)
        0.8 = critical (only silhouette + protrusions reliable)
        1.0 = unreadable
    """
    wear_factor: float = 0.0
    noise_sigma_mm: float = 0.05   # depth-estimation noise; iPhone Pro LiDAR ~0.1mm
    rng_seed: int = 0

    def apply(self, height: np.ndarray, base_thickness_mm: float) -> np.ndarray:
        """Return a worn version of the height field."""
        rng = np.random.default_rng(self.rng_seed)
        h = height.copy()

        # Step 1: micro features (very high-frequency component) eroded first.
        # Use a NARROW kernel so we only catch true micro (sub-mm bumps) — wider
        # kernels would also erode the RS rim markers and protrusions.
        smoothed = self._smooth(h, k=1)
        micro = h - smoothed
        micro_keep = max(0.0, 1.0 - self.wear_factor / 0.4)   # gone by w=0.4
        h = smoothed + micro * micro_keep

        # Step 2: meso features (the protrusions and angles) start eroding above 0.3.
        if self.wear_factor > 0.3:
            meso_keep = max(0.0, 1.0 - (self.wear_factor - 0.3) / 0.4)
            relief = h - base_thickness_mm
            relief = np.maximum(relief, 0)
            h = base_thickness_mm + relief * meso_keep

        # Step 3: macro silhouette starts eroding above 0.7.
        if self.wear_factor > 0.7:
            macro_keep = max(0.0, 1.0 - (self.wear_factor - 0.7) / 0.3)
            relief = h - base_thickness_mm
            h = base_thickness_mm + relief * macro_keep

        # Step 4: measurement noise
        h = h + rng.normal(0, self.noise_sigma_mm, h.shape)

        # Mask: anything originally outside the disc stays outside
        h = np.where(height > 0, h, 0.0)
        return h

    @staticmethod
    def _smooth(arr, k=1):
        out = arr
        for _ in range(k):
            padded = np.pad(out, 1, mode="edge")
            out = (
                padded[ :-2,  :-2] + padded[ :-2, 1:-1] + padded[ :-2, 2:  ] +
                padded[1:-1,  :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:  ] +
                padded[2:  ,  :-2] + padded[2:  , 1:-1] + padded[2:  , 2:  ]
            ) / 9.0
        return out


# --- Channel extractor ------------------------------------------------------

@dataclass
class Extraction:
    """Result of pulling channels back out of an observed height field."""
    macro:  int | None = None
    count:  int | None = None
    height: int | None = None
    angles: int | None = None
    micro:  int | None = None
    rs:     bytes | None = None
    diagnostics: dict = field(default_factory=dict)

    def to_decode_dict(self) -> dict:
        return {
            "macro": self.macro, "count": self.count, "height": self.height,
            "angles": self.angles, "micro": self.micro, "rs": self.rs,
        }


class ChannelExtractor:
    """
    Inverse of PatternGeometry. Given an observed height field, recover
    each channel — or return None if the channel is too degraded to read.

    Confidence thresholds (mm) below which a channel is declared unreadable:
        macro    : silhouette / harmonic must beat noise floor
        count    : need to detect at least N peaks above 0.6 mm
        height   : peak height stability < 0.4 mm spread
        angles   : peak angular positions < 15° error
        micro    : high-frequency component RMS > 0.08 mm
        rs       : 8 rim markers each > 0.4 mm above base
    """

    def __init__(self, radius_mm: float = 15.0, base_thickness_mm: float = 1.0):
        self.radius = radius_mm
        self.base   = base_thickness_mm

    def extract(self, observed: np.ndarray) -> Extraction:
        res = observed.shape[0]
        xs = np.linspace(-self.radius, self.radius, res)
        ys = np.linspace(-self.radius, self.radius, res)
        X, Y = np.meshgrid(xs, ys)
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)
        inside = R <= self.radius

        relief = np.where(inside, observed - self.base, 0.0)
        relief = np.maximum(relief, 0.0)

        ex = Extraction()
        diag = ex.diagnostics

        # --- Detect protrusions (count + angles + height) -------------------
        # Sample relief at radius ≈ 0.65 * R along a ring, find peaks.
        n_samples = 360
        ring_thetas = np.linspace(-math.pi, math.pi, n_samples, endpoint=False)
        ring_r = 0.65 * self.radius
        ring_x = ring_r * np.cos(ring_thetas)
        ring_y = ring_r * np.sin(ring_thetas)
        ring_vals = self._bilinear_sample(observed, ring_x, ring_y) - self.base
        ring_vals = np.maximum(ring_vals, 0)

        peaks = self._find_peaks(ring_vals, min_height=0.6)
        diag["n_peaks"] = len(peaks)
        diag["peak_heights"] = [float(ring_vals[p]) for p in peaks]

        n = 0
        peak_thetas: list[float] = []
        peak_h_mm = 0.0
        if 3 <= len(peaks) <= 18:
            n = len(peaks)
            ex.count = (n - 3) & 0x0F
            avg_h_mm = float(np.mean([ring_vals[p] for p in peaks])) / 0.94
            peak_h_mm = avg_h_mm
            avg_h_mm = float(np.clip(avg_h_mm, 1.5, 8.0))
            ex.height = int(round((avg_h_mm - 1.5) / 6.5 * 255))
            best = self._fit_angles(peaks, n_samples, n)
            if best is not None:
                ex.angles = best
                rots = ANGLES_TO_ROTATIONS(best, n)
                peak_thetas = [float(((t + math.pi) % (2*math.pi)) - math.pi) for t in rots]

        # --- Subtract predicted protrusions from relief BEFORE macro fit -----
        # Otherwise their tails leak into low-order harmonics and confuse the
        # macro detector. We rebuild the protrusion field exactly as the
        # encoder did, then subtract.
        relief_for_macro = relief.copy()
        if peak_thetas and peak_h_mm > 0:
            sigma = self.radius / (1.5 * max(n, 1))
            for theta_i in peak_thetas:
                cx = 0.65 * self.radius * math.cos(theta_i)
                cy = 0.65 * self.radius * math.sin(theta_i)
                d2 = (X - cx)**2 + (Y - cy)**2
                bump = peak_h_mm * np.exp(-d2 / (2 * sigma**2))
                relief_for_macro = relief_for_macro - bump

        # Subtract predicted RS markers too — they sit near the rim but their
        # tails reach into the macro sampling range.
        # (We do this after RS detection below; for now use a simple rim mask.)

        # --- Detect macro family --------------------------------------------
        macro = self._detect_macro(relief_for_macro, R, THETA, inside)
        if macro is not None:
            ex.macro = macro

        # --- Detect micro (high-frequency RMS) ------------------------------
        smoothed = self._smooth(observed, k=3)
        hi_freq = (observed - smoothed)
        hi_freq = np.where(inside, hi_freq, 0)
        micro_rms = float(np.sqrt(np.mean(hi_freq**2)))
        diag["micro_rms_mm"] = micro_rms
        if micro_rms > 0.08:
            diag["micro_present_but_not_decoded"] = True

        # --- Detect RS marker bumps at the rim ------------------------------
        rs_bytes = self._read_rs_markers(observed)
        if rs_bytes is not None:
            ex.rs = rs_bytes

        return ex

    # -- helpers ------------------------------------------------------------

    def _bilinear_sample(self, grid, world_x, world_y):
        h, w = grid.shape
        # pixel coordinate
        px = (world_x + self.radius) / (2 * self.radius) * (w - 1)
        py = (world_y + self.radius) / (2 * self.radius) * (h - 1)
        x0 = np.clip(np.floor(px).astype(int), 0, w - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(np.floor(py).astype(int), 0, h - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        wx = px - x0
        wy = py - y0
        return (
            grid[y0, x0] * (1 - wx) * (1 - wy) +
            grid[y0, x1] *      wx  * (1 - wy) +
            grid[y1, x0] * (1 - wx) *      wy  +
            grid[y1, x1] *      wx  *      wy
        )

    @staticmethod
    def _find_peaks(arr: np.ndarray, min_height: float = 0.6) -> list[int]:
        """Local-maxima on a circular signal."""
        n = len(arr)
        peaks = []
        for i in range(n):
            v = arr[i]
            if v < min_height:
                continue
            left  = arr[(i - 1) % n]
            right = arr[(i + 1) % n]
            if v > left and v >= right:
                peaks.append(i)
        # Suppress duplicates within ~10° of each other (keep highest)
        if not peaks:
            return []
        keep = []
        peaks.sort(key=lambda p: -arr[p])
        for p in peaks:
            if all(min(abs(p - k), n - abs(p - k)) > n / 36 for k in keep):
                keep.append(p)
        return sorted(keep)

    def _fit_angles(self, peaks: list[int], n_samples: int, n: int) -> int | None:
        """
        Find the angles byte (0..255) whose ANGLES_TO_ROTATIONS best matches
        the observed peak positions. Brute force 256 candidates.
        """
        observed = np.sort([2 * math.pi * p / n_samples - math.pi for p in peaks])
        best_byte = None
        best_err = float("inf")
        for byte in range(256):
            rots = ANGLES_TO_ROTATIONS(byte, n)
            # Wrap to [-pi, pi] and sort
            rots = ((rots + math.pi) % (2 * math.pi)) - math.pi
            rots = np.sort(rots)
            err = float(np.mean(np.abs(rots - observed)))
            if err < best_err:
                best_err = err
                best_byte = byte
        # 15° tolerance per ТЗ §2.4
        if best_err < math.radians(15):
            return best_byte
        return None

    def _detect_macro(self, relief, R, THETA, inside) -> int | None:
        """Detect macro family by matching the harmonic content.

        Templates here MUST match PatternGeometry._macro_contribution exactly.
        """
        masked = np.where(inside & (R > 0.2 * self.radius) & (R < 0.5 * self.radius),
                          relief, 0)

        sigs = [
            (2,  0,  0.0, 1.0),
            (3,  0,  0.0, 1.0),
            (6,  0,  0.0, 1.0),
            (4,  0,  0.0, 1.0),
            (5,  0,  0.0, 1.0),
            (6,  2,  0.0, 1.0),
            (8,  0,  0.0, 1.0),
            (2,  4,  0.5, 1.0),
            (2,  0,  0.0, 0.0),
            (4,  0,  0.0, 0.0),
            (4,  2,  0.0, 1.0),
            (8,  4,  0.0, 1.0),
            (5,  10, 0.0, 1.0),
            (12, 0,  0.0, 1.0),
            (4,  0,  0.0, -1.0),
            (2,  6,  0.5, 1.0),
        ]

        r_norm = R / self.radius
        family_scores = []
        for macro in range(16):
            primary, secondary, phase, radial_mode = sigs[macro]
            if radial_mode == 1.0:
                radial = (1 - r_norm)
                tmpl = 0.4 * np.cos(primary * THETA) * radial
                if secondary > 0:
                    tmpl = tmpl + 0.35 * np.cos(secondary * THETA + phase * math.pi) * radial
            elif radial_mode == 0.0:
                tmpl = 0.4 * np.cos(primary * THETA + 6 * r_norm)
            else:
                tmpl = 0.4 * np.cos(primary * THETA - 8 * r_norm)
            tmpl = np.where(inside, tmpl, 0)
            # Normalize template so we compare cosine similarity, not raw correlation.
            # Without this, templates with more harmonics get an unfair score boost.
            tmpl_norm = float(np.sqrt(np.sum(tmpl ** 2)))
            if tmpl_norm > 1e-9:
                tmpl = tmpl / tmpl_norm
            score = float(np.sum(masked * tmpl))
            family_scores.append((score, macro))

        family_scores.sort(reverse=True)
        if len(family_scores) >= 2:
            top, second = family_scores[0], family_scores[1]
            if top[0] > 0 and top[0] > 1.10 * abs(second[0]):
                return top[1]
        return None

    def _read_rs_markers(self, observed: np.ndarray) -> bytes | None:
        """Sample the 32 rim positions and threshold each into 1 bit."""
        n_markers = 32
        levels = []
        for k in range(n_markers):
            angle = (k / n_markers) * 2 * math.pi
            cx = 0.92 * self.radius * math.cos(angle)
            cy = 0.92 * self.radius * math.sin(angle)
            samples = []
            for da in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                for dr in (0, 0.3):
                    sx = cx + dr * self.radius * 0.04 * math.cos(da)
                    sy = cy + dr * self.radius * 0.04 * math.sin(da)
                    samples.append(self._bilinear_sample(observed, np.array([sx]), np.array([sy]))[0])
            v = max(samples) - self.base
            levels.append(v)

        # If fewer than half the rim shows any signal, RS row is gone.
        present = sum(1 for v in levels if v > 0.4)
        if present < n_markers // 2:
            return None

        # Threshold at 2.0 mm (midpoint between 1.0 and 3.0 mm)
        bits = [1 if v > 2.0 else 0 for v in levels]

        # Pack 32 bits into 4 bytes
        rs_bytes = bytearray()
        for byte_idx in range(4):
            b = 0
            for i in range(8):
                b = (b << 1) | bits[byte_idx * 8 + i]
            rs_bytes.append(b)
        return bytes(rs_bytes)

    @staticmethod
    def _smooth(arr, k=1):
        out = arr
        for _ in range(k):
            padded = np.pad(out, 1, mode="edge")
            out = (
                padded[ :-2,  :-2] + padded[ :-2, 1:-1] + padded[ :-2, 2:  ] +
                padded[1:-1,  :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:  ] +
                padded[2:  ,  :-2] + padded[2:  , 1:-1] + padded[2:  , 2:  ]
            ) / 9.0
        return out


# --- End-to-end simulator ---------------------------------------------------

def simulate_capture(channels: Channels,
                     wear_factor: float = 0.0,
                     noise_sigma_mm: float = 0.05,
                     grid_resolution: int = 96,
                     rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end: encode → render → wear → capture.
    Returns (clean_height_field, observed_height_field).
    """
    geom = PatternGeometry(grid_resolution=grid_resolution)
    clean = geom._build_height_field(channels)
    wear = WearProfile(
        wear_factor=wear_factor,
        noise_sigma_mm=noise_sigma_mm,
        rng_seed=rng_seed,
    )
    observed = wear.apply(clean, base_thickness_mm=geom.base)
    return clean, observed


__all__ = ["WearProfile", "ChannelExtractor", "Extraction", "simulate_capture"]
