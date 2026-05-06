/// Height field renderer — Dart port of buildHeightField() from web2/index.html.
///
/// Used only for testing (generating synthetic fields to validate the extractor)
/// and as a reference. Not called in the production scan path; production uses
/// real depth data from ARKit / ARCore / MiDaS.
///
/// Port is byte-for-byte against the JS reference. Constants must match
/// channel_extractor.dart exactly.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../codec/pattern_codec.dart';
import 'channel_extractor.dart' show macroSigs;

// Constants — must match web2/index.html and channel_extractor.dart
const double _radiusMm = 15.0;
const double _baseMm   = 1.5;
const int    _res      = 128;
const double _protHMin = 0.40;
const double _protHMax = 0.90;
const double _rsLowH   = 0.35;
const double _rsHighH  = 0.95;
const double _macroAmp = 0.20;
const double _microAmp = 0.15;

class HeightFieldRenderer {
  const HeightFieldRenderer();

  /// Render a [Channels] object to a 128×128 height field (flat Float32List).
  ///
  /// [wear] controls simulated degradation (0.0 = new, 1.0 = unreadable).
  Float32List render(
    Channels ch, {
    double wear = 0.0,
    int res = _res,
    double radius = _radiusMm,
  }) {
    final peak_h = _protHMin + (ch.height / 255.0) * (_protHMax - _protHMin);
    final n = 3 + (ch.count % 16);

    final base  = (ch.angles >> 4) & 0x0F;
    final modul = ch.angles & 0x0F;
    final baseOff = (base / 16) * 2 * math.pi;
    final modAmp  = (modul / 15) * (math.pi / n);
    final rotations = <double>[];
    for (int i = 0; i < n; i++) {
      rotations.add(baseOff + i * (2 * math.pi / n) + modAmp * math.sin(2 * i));
    }

    final sig = macroSigs[ch.macro % 16];

    // Unpack RS parity bits
    final rsBits = List<int>.filled(32, 0);
    for (int b = 0; b < 4; b++) {
      final byte = (b < ch.rs.length) ? ch.rs[b] : 0;
      for (int i = 0; i < 8; i++) {
        rsBits[b * 8 + i] = (byte >> (7 - i)) & 1;
      }
    }

    // Wear factors — mirror JS exactly
    final microKeep = math.max(0.0, 1.0 - wear / 0.4);
    final mesoKeep  = wear > 0.3 ? math.max(0.0, 1.0 - (wear - 0.3) / 0.4) : 1.0;
    final macroKeep = wear > 0.7 ? math.max(0.0, 1.0 - (wear - 0.7) / 0.3) : 1.0;

    final dx = (2 * radius) / (res - 1);
    final protSigma = math.max(radius / (1.5 * n), 1.5 * dx);
    final protSigSq = protSigma * protSigma;
    final rsSigma   = radius / 24.0;
    final rsSigSq   = rsSigma * rsSigma;

    final field = Float32List(res * res);

    final primary   = sig[0].toInt();
    final secondary = sig[1].toInt();
    final phase     = sig[2].toDouble();
    final mode      = sig[3].toDouble();

    for (int j = 0; j < res; j++) {
      final y = -radius + j * dx;
      for (int i = 0; i < res; i++) {
        final x = -radius + i * dx;
        final r = math.sqrt(x * x + y * y);
        final idx = j * res + i;

        if (r > radius) {
          field[idx] = 0;
          continue;
        }

        final theta = math.atan2(y, x);
        final rNorm = r / radius;

        // Macro contribution
        double m;
        if (mode == 1) {
          final radial = 1 - rNorm;
          m = _macroAmp * math.cos(primary * theta) * radial;
          if (secondary > 0) {
            m += _macroAmp * 0.85 * math.cos(secondary * theta + phase * math.pi) * radial;
          }
        } else if (mode == 0) {
          m = _macroAmp * math.cos(primary * theta + 6 * rNorm);
        } else {
          m = _macroAmp * math.cos(primary * theta - 8 * rNorm);
        }
        m *= macroKeep;

        // Protrusions — flat-topped super-Gaussian (exp(-u²) where u = d²/σ²)
        double prot = 0;
        for (final t in rotations) {
          final cx = 0.65 * radius * math.cos(t);
          final cy = 0.65 * radius * math.sin(t);
          final d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
          final u  = d2 / protSigSq;
          prot += peak_h * math.exp(-u * u);
        }
        prot *= mesoKeep;

        // Micro (deterministic seeded noise — must match JS hash exactly)
        final ix   = (x + 15) * 8;
        final iy   = (y + 15) * 8;
        final seed = ((ix.floor() * 73856093) ^
                      (iy.floor() * 19349663) ^
                      (ch.micro * 8191 + 17)) & 0xFFFFFFFF;
        final pseudoRand = ((seed * 9301 + 49297) % 233280) / 233280 - 0.5;
        final mc = pseudoRand * _microAmp * microKeep;

        // RS markers
        double rsField = 0;
        for (int k = 0; k < 32; k++) {
          final angle = (k / 32) * 2 * math.pi;
          final mcx = 0.92 * radius * math.cos(angle);
          final mcy = 0.92 * radius * math.sin(angle);
          final d2  = (x - mcx) * (x - mcx) + (y - mcy) * (y - mcy);
          final cutoff = (3.5 * rsSigma) * (3.5 * rsSigma);
          if (d2 < cutoff) {
            final level = rsBits[k] == 1 ? _rsHighH : _rsLowH;
            final u = d2 / rsSigSq;
            rsField += level * math.exp(-u * u);
          }
        }
        rsField *= mesoKeep;

        double h = _baseMm + m + prot + mc + rsField;
        if (h < _baseMm) h = _baseMm;
        field[idx] = h;
      }
    }
    return field;
  }
}
