/// Integration tests for ScanPipeline and depth estimator fallback chain.
import 'dart:typed_data';
import 'dart:ui' show Offset, Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:slc_scanner/codec/pattern_codec.dart';
import 'package:slc_scanner/ml/depth_estimator.dart';
import 'package:slc_scanner/ml/tile_detector.dart';
import 'package:slc_scanner/pipeline/scan_pipeline.dart';

// ---------------------------------------------------------------------------
// Test doubles for FallbackDepthEstimator tests
// ---------------------------------------------------------------------------

class _NullEstimator implements DepthEstimator {
  @override DepthSource get source => DepthSource.lidar;
  @override Future<Float32List?> estimate({
    required Uint8List yuvBytes, required int imageWidth,
    required int imageHeight, required TileDetection detection,
  }) async => null;
  @override void dispose() {}
}

class _ThrowingEstimator implements DepthEstimator {
  @override DepthSource get source => DepthSource.lidar;
  @override Future<Float32List?> estimate({
    required Uint8List yuvBytes, required int imageWidth,
    required int imageHeight, required TileDetection detection,
  }) async => throw Exception('simulated hardware failure');
  @override void dispose() {}
}

class _FixedEstimator implements DepthEstimator {
  final DepthSource _source;
  final Float32List _field;
  _FixedEstimator(this._source, this._field);
  @override DepthSource get source => _source;
  @override Future<Float32List?> estimate({
    required Uint8List yuvBytes, required int imageWidth,
    required int imageHeight, required TileDetection detection,
  }) async => _field;
  @override void dispose() {}
}

void main() {
  final codec = PatternCodec();
  final emptyImage = Uint8List(0);

  Future<ScanResult> runPipeline(int itemId, {double wear = 0.0}) async {
    final ch = codec.encode(itemId);
    final pipeline = ScanPipeline(
      detector:      MockTileDetector(),
      depthEstimator: MockDepthEstimator(channels: ch, wear: wear),
    );
    return pipeline.run(emptyImage, 640, 480);
  }

  // -------------------------------------------------------------------------
  group('end_to_end synthetic', () {
    // IDs chosen for low count (n=4–11 protrusions) so that peak detection
    // is reliable and RS correction has headroom. High-count IDs (n≥16)
    // are excluded here — they're documented in LIMITATIONS.md §1.
    final ids = [
      0x21708090, // macro=2, count=1, n=4
      0x71F082A4, // macro=7, count=1, n=4
      0x18BE5C09, // macro=1, count=8, n=11
      0x9A5713D7, // macro=9, count=10, n=13
      0x12345678, // macro=1, count=2, n=5
      0xE0607080, // macro=14, count=0, n=3
      0x5A5B5C5D, // macro=5, count=10, n=13
      0xCAFEBABE, // macro=12, count=10, n=13
    ];

    for (final id in ids) {
      test('id=0x${id.toRadixString(16).padLeft(8, '0')} wear=0.0', () async {
        final result = await runPipeline(id, wear: 0.0);
        // Exact decode is probabilistic; test that the pipeline produces a
        // valid output (not 'lost') — RS may or may not correct all errors.
        expect(result.decode.fallbackLevel, isNot(equals('lost')));
        expect(result.depthSource, equals(DepthSource.mock));
      });
    }

    test('batch wear=0.0: ≥15% exact-decode rate', () async {
      // Exact decode (itemId == id) requires RS to correct the height channel
      // error introduced by the 1.40 ring-calibration factor. At wear=0.0 the
      // full micro-noise amplitude perturbs ring peaks, leaving height
      // systematically below encoded value. When height error coincides with an
      // angle error, two simultaneous payload errors exhaust RS capacity
      // (2×2+1 erasure = 5 > nsym=4). JS reference achieves ~0/16 exact decode
      // at wear=0.0 for similar ID sets; Dart achieves ≥3/16 = 18.75% due to
      // float32 arithmetic differences that occasionally produce accurate peaks.
      // The 47% figure cited in README §"Honest accuracy numbers" covers the
      // 'category_only' fallback, not itemId equality — this test catches
      // total RS-decoder regressions (e.g., breaking GF arithmetic).
      final allIds = [
        0x21708090, // macro=2, n=4
        0x71F082A4, // macro=7, n=4
        0x18BE5C09, // macro=1, n=11
        0x9A5713D7, // macro=9, n=13
        0x12345678, // macro=1, n=5
        0xE0607080, // macro=14, n=3
        0x5A5B5C5D, // macro=5, n=13
        0xCAFEBABE, // macro=12, n=13
        0xA1B2C3D4, // macro=10, n=4
        0xB2607080, // macro=11, n=5
        0xD0708090, // macro=13, n=3
        0xF0F0F0F0, // macro=15, n=3
        0x00000001, // macro=0,  n=3
        0xF1809000, // macro=15, n=4
        0x13572468, // macro=1,  n=6
        0x55AA55AA, // macro=5,  n=8
      ];
      int success = 0;
      for (final id in allIds) {
        final r = await runPipeline(id, wear: 0.0);
        if (r.decode.itemId == id) success++;
      }
      expect(success / allIds.length, greaterThanOrEqualTo(0.15));
    });
  });

  // -------------------------------------------------------------------------
  group('fallback levels', () {
    test('MockTileDetector produces detected tile', () async {
      final ch = codec.encode(0x2DCA3791);
      final pipeline = ScanPipeline(
        detector:      MockTileDetector(),
        depthEstimator: MockDepthEstimator(channels: ch),
      );
      final result = await pipeline.run(emptyImage, 640, 480);
      expect(result.tileDetection, isNotNull);
      expect(result.tileDetection!.confidence, greaterThan(0.9));
    });

    test('diagnostics has non-zero totalMs', () async {
      final result = await runPipeline(0x2DCA3791);
      expect(result.diagnostics.totalMs, greaterThan(0));
    });

    test('genuineConfidence is 1.0', () async {
      final result = await runPipeline(0x2DCA3791);
      expect(result.decode.genuineConfidence, equals(1.0));
    });
  });

  // -------------------------------------------------------------------------
  group('FallbackDepthEstimator', () {
    final dummyDetection = TileDetection(
      bbox: const Rect.fromLTWH(160, 80, 320, 320),
      center: const Offset(320, 240),
      confidence: 0.97,
      diameterPixels: 320,
    );
    final emptyYuv = Uint8List(0);
    final fixedField = Float32List(128 * 128);

    test('returns first estimator result when it succeeds', () async {
      final primary   = _FixedEstimator(DepthSource.lidar, fixedField);
      final secondary = _FixedEstimator(DepthSource.midas, Float32List(128 * 128));
      final fallback  = FallbackDepthEstimator([primary, secondary]);

      final result = await fallback.estimate(
        yuvBytes: emptyYuv, imageWidth: 640, imageHeight: 480,
        detection: dummyDetection,
      );
      expect(result, isNotNull);
      expect(fallback.source, equals(DepthSource.lidar));
    });

    test('falls back to second when first returns null', () async {
      final fallback = FallbackDepthEstimator([
        _NullEstimator(),
        _FixedEstimator(DepthSource.midas, fixedField),
      ]);

      final result = await fallback.estimate(
        yuvBytes: emptyYuv, imageWidth: 640, imageHeight: 480,
        detection: dummyDetection,
      );
      expect(result, isNotNull);
      expect(fallback.source, equals(DepthSource.midas));
    });

    test('falls back past throwing estimator', () async {
      final fallback = FallbackDepthEstimator([
        _ThrowingEstimator(),
        _FixedEstimator(DepthSource.midas, fixedField),
      ]);

      final result = await fallback.estimate(
        yuvBytes: emptyYuv, imageWidth: 640, imageHeight: 480,
        detection: dummyDetection,
      );
      expect(result, isNotNull);
      expect(fallback.source, equals(DepthSource.midas));
    });

    test('returns null when all estimators fail', () async {
      final fallback = FallbackDepthEstimator([
        _NullEstimator(),
        _ThrowingEstimator(),
      ]);

      final result = await fallback.estimate(
        yuvBytes: emptyYuv, imageWidth: 640, imageHeight: 480,
        detection: dummyDetection,
      );
      expect(result, isNull);
    });

    test('reports source of first successful estimator on repeated calls', () async {
      final fallback = FallbackDepthEstimator([
        _NullEstimator(),
        _FixedEstimator(DepthSource.arCore, fixedField),
      ]);
      await fallback.estimate(
        yuvBytes: emptyYuv, imageWidth: 640, imageHeight: 480,
        detection: dummyDetection,
      );
      // Source should reflect last successful estimator
      expect(fallback.source, equals(DepthSource.arCore));
    });
  });

  // -------------------------------------------------------------------------
  group('performance', () {
    test('pipeline completes in under 500ms (synthetic, no ML)', () async {
      final ch = codec.encode(0x2DCA3791);
      final pipeline = ScanPipeline(
        detector:      MockTileDetector(),
        depthEstimator: MockDepthEstimator(channels: ch),
      );
      final start = DateTime.now();
      await pipeline.run(emptyImage, 640, 480);
      final elapsed = DateTime.now().difference(start).inMilliseconds;
      // No ML inference — should be well under 500ms even on slow CI
      expect(elapsed, lessThan(500));
    });
  });
}
