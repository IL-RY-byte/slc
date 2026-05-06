/// Channel extractor — Dart port of the JavaScript extractor in web2/index.html.
///
/// Port is byte-for-byte against the JS reference: same constants, same ring
/// smoothing, same peak thresholds, same calibration divisor (1.40), same RS
/// presence threshold. Do NOT diverge from the JS algorithm without updating
/// the test fixtures in flutter/test/fixtures/extractor_fixtures.json and
/// re-running the JS fixture generator.
///
/// Constants that changed from the older Python reference (perception.py):
///   BASE_MM      1.5  (Python used 1.0 in the extractor constructor)
///   PROT_H_MIN   0.40 (Python used 1.5)
///   PROT_H_MAX   0.90 (Python used 8.0)
///   RS_LOW_H     0.35 (Python used 1.0)
///   RS_HIGH_H    0.95 (Python used 3.0)
///   peak min-height  0.28 post-smoothing (Python used 0.6 raw)
///   calibration  /1.40 (Python used /0.94)
library;

import 'dart:math' as math;
import 'dart:typed_data';

// ---------------------------------------------------------------------------
// Constants — must match web2/index.html
// ---------------------------------------------------------------------------

const double _radiusMm  = 15.0;
const double _baseMm    = 1.5;
const int    _res       = 128;
const double _protHMin  = 0.40;
const double _protHMax  = 0.90;
const double _rsLowH    = 0.35;
const double _rsHighH   = 0.95;

// MACRO_SIGS: [primary, secondary, phase, mode]
// mode: 1 = radial-taper, 0 = radial-spiral-CW, -1 = radial-spiral-CCW
const List<List<num>> macroSigs = [
  [2,  0,  0,    1],
  [3,  0,  0,    1],
  [6,  0,  0,    1],
  [4,  0,  0,    1],
  [5,  0,  0,    1],
  [6,  2,  0,    1],
  [8,  0,  0,    1],
  [2,  4,  0.5,  1],
  [2,  0,  0,    0],
  [4,  0,  0,    0],
  [4,  2,  0,    1],
  [8,  4,  0,    1],
  [5,  10, 0,    1],
  [12, 0,  0,    1],
  [4,  0,  0,   -1],
  [2,  6,  0.5,  1],
];

// ---------------------------------------------------------------------------
// Public result type
// ---------------------------------------------------------------------------

class ExtractionResult {
  final int?  macro;
  final int?  count;
  final int?  height;
  final int?  angles;
  // micro is always null in v1 — requires RNG inversion; RS recovers it
  final int?  micro;
  final Uint8List? rs;
  final Map<String, dynamic> diagnostics;

  const ExtractionResult({
    this.macro,
    this.count,
    this.height,
    this.angles,
    this.micro,
    this.rs,
    this.diagnostics = const {},
  });

  @override
  String toString() =>
      'ExtractionResult(macro=$macro, count=$count, height=$height, '
      'angles=$angles, micro=$micro, rs=${rs?.map((b) => b.toRadixString(16).padLeft(2, "0")).join()})';
}

// ---------------------------------------------------------------------------
// Channel extractor
// ---------------------------------------------------------------------------

class ChannelExtractor {
  const ChannelExtractor();

  /// Extract channels from a 128×128 height field (flat Float32List, row-major).
  ///
  /// [field] must have exactly res*res elements.
  /// [radiusMm] defaults to 15.0 (tile radius).
  ExtractionResult extract(
    Float32List field, {
    int res = _res,
    double radius = _radiusMm,
  }) {
    assert(field.length == res * res);

    final dx = (2 * radius) / (res - 1);

    // -----------------------------------------------------------------------
    // Ring sample at r = 0.65*radius for protrusion detection
    // -----------------------------------------------------------------------
    const nSamples = 360;
    final ringR = 0.65 * radius;
    final ringRaw = Float32List(nSamples);
    for (int i = 0; i < nSamples; i++) {
      final theta = -math.pi + (i / nSamples) * 2 * math.pi;
      final x = ringR * math.cos(theta);
      final y = ringR * math.sin(theta);
      final v = _bilinear(field, res, x, y, radius) - _baseMm;
      ringRaw[i] = v > 0 ? v : 0.0;
    }

    // 5-tap (halfW=2) moving average — matches JS exactly.
    // Suppresses bilinear-sampling jitter that Python's cubic interpolation
    // handles natively; without this, wide bumps for low-count tiles produce
    // spurious local maxima inside one true peak.
    final ringVals = Float32List(nSamples);
    const halfW = 2;
    for (int i = 0; i < nSamples; i++) {
      double s = 0;
      for (int k = -halfW; k <= halfW; k++) {
        s += ringRaw[(i + k + nSamples) % nSamples];
      }
      ringVals[i] = s / (2 * halfW + 1);
    }

    // Peak detection: 0.28 mm threshold (post-smoothing).
    final peaks = findPeaks(ringVals, 0.28);

    int? resultCount, resultHeight, resultAngles;
    var peakThetas = <double>[];
    double peakH = 0;

    if (peaks.length >= 3 && peaks.length <= 18) {
      final n = peaks.length;
      resultCount = (n - 3) & 0x0F;

      double avgH = 0;
      for (final p in peaks) avgH += ringVals[p];
      avgH /= peaks.length;
      // Calibration: ring sample receives the protrusion's own contribution
      // plus overlap from neighbours — average ratio ~1.40 across the count range.
      avgH /= 1.40;
      peakH = avgH;
      avgH = avgH.clamp(_protHMin, _protHMax);
      resultHeight = (((avgH - _protHMin) / (_protHMax - _protHMin)) * 255).round();

      final fitted = fitAngles(peaks, nSamples, n);
      if (fitted != null) {
        resultAngles = fitted;
        final base  = (fitted >> 4) & 0x0F;
        final modul = fitted & 0x0F;
        final baseOff = (base / 16) * 2 * math.pi;
        final modAmp  = (modul / 15) * (math.pi / n);
        for (int i = 0; i < n; i++) {
          double t = baseOff + i * (2 * math.pi / n) + modAmp * math.sin(2 * i);
          t = ((t + math.pi) % (2 * math.pi)) - math.pi;
          peakThetas.add(t);
        }
      }
    }

    // -----------------------------------------------------------------------
    // Build relief field for macro detection
    // -----------------------------------------------------------------------
    final relief = Float32List(field.length);
    for (int i = 0; i < field.length; i++) {
      final v = field[i] - _baseMm;
      relief[i] = v > 0 ? v : 0.0;
    }

    final int? resultMacro = detectMacro(
      relief, res, radius, dx, peakThetas, peakH,
    );

    // RS markers
    final resultRs = readRsMarkers(field, res, radius);

    return ExtractionResult(
      macro:  resultMacro,
      count:  resultCount,
      height: resultHeight,
      angles: resultAngles,
      micro:  null,
      rs:     resultRs,
      diagnostics: {
        'nPeaks': peaks.length,
        'peakH': peakH,
      },
    );
  }

  // -------------------------------------------------------------------------
  // Bilinear sampler — matches JS bilinearSample exactly
  // -------------------------------------------------------------------------

  static double _bilinear(
    Float32List field, int res, double worldX, double worldY, double radius,
  ) {
    final px = (worldX + radius) / (2 * radius) * (res - 1);
    final py = (worldY + radius) / (2 * radius) * (res - 1);
    final x0 = px.floor().clamp(0, res - 1);
    final x1 = (x0 + 1).clamp(0, res - 1);
    final y0 = py.floor().clamp(0, res - 1);
    final y1 = (y0 + 1).clamp(0, res - 1);
    final wx = px - x0;
    final wy = py - y0;
    return field[y0 * res + x0] * (1 - wx) * (1 - wy) +
           field[y0 * res + x1] *      wx  * (1 - wy) +
           field[y1 * res + x0] * (1 - wx) *      wy  +
           field[y1 * res + x1] *      wx  *      wy;
  }

  // -------------------------------------------------------------------------
  // Peak detection — matches JS findPeaks exactly
  // -------------------------------------------------------------------------

  static List<int> findPeaks(Float32List arr, double minHeight) {
    final n = arr.length;
    final raw = <int>[];
    for (int i = 0; i < n; i++) {
      final v = arr[i];
      if (v < minHeight) continue;
      final left  = arr[(i - 1 + n) % n];
      final right = arr[(i + 1) % n];
      if (v > left && v >= right) raw.add(i);
    }
    if (raw.isEmpty) return [];
    raw.sort((a, b) => arr[b].compareTo(arr[a]));
    final keep = <int>[];
    final minSep = n / 36.0; // ~10°
    for (final p in raw) {
      final ok = keep.every((k) {
        final diff = (p - k).abs();
        return math.min(diff, n - diff) > minSep;
      });
      if (ok) keep.add(p);
    }
    keep.sort();
    return keep;
  }

  // -------------------------------------------------------------------------
  // Angle fitting — matches JS fitAngles exactly
  // -------------------------------------------------------------------------

  static int? fitAngles(List<int> peaks, int nSamples, int nProt) {
    final observed = peaks
        .map((p) => 2 * math.pi * p / nSamples - math.pi)
        .toList()
      ..sort();

    int? bestByte;
    double bestErr = double.infinity;

    for (int byte = 0; byte < 256; byte++) {
      final base  = (byte >> 4) & 0x0F;
      final modul = byte & 0x0F;
      final baseOff = (base / 16) * 2 * math.pi;
      final modAmp  = (modul / 15) * (math.pi / nProt);
      final rots = <double>[];
      for (int i = 0; i < nProt; i++) {
        double r = baseOff + i * (2 * math.pi / nProt) + modAmp * math.sin(2 * i);
        r = ((r + math.pi) % (2 * math.pi)) - math.pi;
        rots.add(r);
      }
      rots.sort();
      double err = 0;
      for (int k = 0; k < nProt; k++) {
        err += (rots[k] - observed[k]).abs();
      }
      err /= nProt;
      if (err < bestErr) {
        bestErr = err;
        bestByte = byte;
      }
    }
    // 15° tolerance (matches JS and Python)
    return bestErr < (15 * math.pi / 180) ? bestByte : null;
  }

  // -------------------------------------------------------------------------
  // Macro family detection — matches JS detectMacro exactly
  // -------------------------------------------------------------------------

  static int? detectMacro(
    Float32List relief,
    int res,
    double radius,
    double dx,
    List<double> peakThetas,
    double peakH,
  ) {
    // Subtract predicted protrusions (same logic as JS)
    final subbed = Float32List.fromList(relief);
    if (peakThetas.isNotEmpty && peakH > 0) {
      final n = peakThetas.length;
      final sigma = math.max(radius / (1.5 * n), 1.5 * dx);
      final sig2 = 2 * sigma * sigma;
      for (int j = 0; j < res; j++) {
        final y = -radius + j * dx;
        for (int i = 0; i < res; i++) {
          final x = -radius + i * dx;
          double bump = 0;
          for (final t in peakThetas) {
            final cx = 0.65 * radius * math.cos(t);
            final cy = 0.65 * radius * math.sin(t);
            final d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            bump += peakH * math.exp(-d2 / sig2);
          }
          subbed[j * res + i] -= bump;
        }
      }
    }

    // Score each macro template with L2 normalization
    final scores = <({int macro, double score})>[];
    for (int m = 0; m < 16; m++) {
      final primary   = macroSigs[m][0].toInt();
      final secondary = macroSigs[m][1].toInt();
      final phase     = macroSigs[m][2].toDouble();
      final mode      = macroSigs[m][3].toDouble();

      double score = 0, normSq = 0;
      for (int j = 0; j < res; j++) {
        final y = -radius + j * dx;
        for (int i = 0; i < res; i++) {
          final x = -radius + i * dx;
          final r = math.sqrt(x * x + y * y);
          if (r > radius) continue;
          if (r < 0.2 * radius || r > 0.5 * radius) continue;
          final theta = math.atan2(y, x);
          final rNorm = r / radius;
          double t;
          if (mode == 1) {
            final radial = 1 - rNorm;
            t = 0.4 * math.cos(primary * theta) * radial;
            if (secondary > 0) {
              t += 0.35 * math.cos(secondary * theta + phase * math.pi) * radial;
            }
          } else if (mode == 0) {
            t = 0.4 * math.cos(primary * theta + 6 * rNorm);
          } else {
            t = 0.4 * math.cos(primary * theta - 8 * rNorm);
          }
          score  += subbed[j * res + i] * t;
          normSq += t * t;
        }
      }
      final norm = math.sqrt(normSq);
      scores.add((macro: m, score: norm > 1e-9 ? score / norm : 0.0));
    }

    scores.sort((a, b) => b.score.compareTo(a.score));
    if (scores[0].score > 0 &&
        scores[0].score > 1.10 * scores[1].score.abs()) {
      return scores[0].macro;
    }
    return null;
  }

  // -------------------------------------------------------------------------
  // RS marker reader — matches JS readRsMarkers exactly
  // -------------------------------------------------------------------------

  static Uint8List? readRsMarkers(Float32List field, int res, double radius) {
    const nMarkers = 32;
    final levels = <double>[];

    for (int k = 0; k < nMarkers; k++) {
      final angle = (k / nMarkers) * 2 * math.pi;
      final cx = 0.92 * radius * math.cos(angle);
      final cy = 0.92 * radius * math.sin(angle);
      double maxV = double.negativeInfinity;
      for (int da = 0; da < 8; da++) {
        final a = (da / 8) * 2 * math.pi;
        for (final dr in [0.0, 0.3]) {
          final sx = cx + dr * radius * 0.04 * math.cos(a);
          final sy = cy + dr * radius * 0.04 * math.sin(a);
          final v = _bilinear(field, res, sx, sy, radius);
          if (v > maxV) maxV = v;
        }
      }
      levels.add(maxV - _baseMm);
    }

    // Presence threshold and bit threshold — match JS exactly
    final presentThr   = _rsLowH * 0.5;          // ~0.175 mm
    final highVsLowThr = (_rsLowH + _rsHighH) / 2; // ~0.65 mm
    final present = levels.where((v) => v > presentThr).length;
    if (present < 16) return null;

    final bits = levels.map((v) => v > highVsLowThr ? 1 : 0).toList();
    final out = Uint8List(4);
    for (int b = 0; b < 4; b++) {
      int v = 0;
      for (int i = 0; i < 8; i++) {
        v = (v << 1) | bits[b * 8 + i];
      }
      out[b] = v;
    }
    return out;
  }
}
