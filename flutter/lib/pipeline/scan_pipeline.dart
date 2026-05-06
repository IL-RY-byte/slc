/// Scan pipeline — wires Stage 1 (detection) through Stage 4 (RS decode).
///
/// Usage:
///   final pipeline = ScanPipeline(
///     detector:      MockTileDetector(),
///     depthEstimator: MockDepthEstimator(channels: ch),
///     // macroClassifier and wearEstimator are optional; fall back to
///     // harmonic correlator and 0.0 wear respectively.
///   );
///   final result = await pipeline.run(yuvBytes, width, height);
library;

import 'dart:typed_data';

import '../codec/pattern_codec.dart';
import '../ml/tile_detector.dart';
import '../ml/depth_estimator.dart';
import '../ml/macro_classifier.dart';
import '../ml/wear_estimator.dart';
import '../perception/channel_extractor.dart';

// ---------------------------------------------------------------------------
// Pipeline result
// ---------------------------------------------------------------------------

class ScanResult {
  final DecodeResult decode;
  final TileDetection? tileDetection;
  final DepthSource? depthSource;
  final ExtractionResult? extraction;
  final double estimatedWear;
  final ScanDiagnostics diagnostics;

  const ScanResult({
    required this.decode,
    this.tileDetection,
    this.depthSource,
    this.extraction,
    this.estimatedWear = 0.0,
    required this.diagnostics,
  });
}

class ScanDiagnostics {
  /// Milliseconds for each pipeline stage.
  final int detectMs;
  final int depthMs;
  final int extractMs;
  final int decodeMs;
  final int totalMs;

  const ScanDiagnostics({
    required this.detectMs,
    required this.depthMs,
    required this.extractMs,
    required this.decodeMs,
    required this.totalMs,
  });

  Map<String, int> toMap() => {
        'detectMs': detectMs,
        'depthMs': depthMs,
        'extractMs': extractMs,
        'decodeMs': decodeMs,
        'totalMs': totalMs,
      };
}

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

class ScanConfig {
  /// YOLOv8 confidence threshold for auto-capture (3 consecutive frames).
  final double detectionConfidenceThreshold;

  /// CNN macro classifier confidence below which the harmonic correlator is
  /// preferred.
  final double macroClassifierFallbackThreshold;

  /// Minimum fraction of RS markers that must be present.
  /// When wear > 0.4, lowered to 0.35 by the wear-adaptive logic.
  final double rsPresenceThreshold;

  const ScanConfig({
    this.detectionConfidenceThreshold = 0.85,
    this.macroClassifierFallbackThreshold = cnnFallbackThreshold,
    this.rsPresenceThreshold = 0.5,
  });
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

class ScanPipeline {
  final TileDetector detector;
  final DepthEstimator depthEstimator;
  final MacroClassifier? macroClassifier;
  final WearEstimator? wearEstimator;
  final ScanConfig config;
  final PatternCodec _codec = PatternCodec();
  final ChannelExtractor _extractor = const ChannelExtractor();

  ScanPipeline({
    required this.detector,
    required this.depthEstimator,
    this.macroClassifier,
    this.wearEstimator,
    this.config = const ScanConfig(),
  });

  /// Run the full pipeline on one camera frame.
  ///
  /// Returns [ScanResult] in all cases — check [ScanResult.decode.fallbackLevel]
  /// for 'lost' if no tile was found.
  Future<ScanResult> run(
    Uint8List yuvBytes,
    int imageWidth,
    int imageHeight,
  ) async {
    final wallStart = DateTime.now().millisecondsSinceEpoch;

    // -------------------------------------------------------------------------
    // Stage 1: tile detection
    // -------------------------------------------------------------------------
    final t0 = DateTime.now().millisecondsSinceEpoch;
    final detection = await detector.detect(
      yuvBytes: yuvBytes,
      width: imageWidth,
      height: imageHeight,
    );
    final detectMs = DateTime.now().millisecondsSinceEpoch - t0;

    if (detection == null) {
      return ScanResult(
        decode: const DecodeResult(
          itemId: null,
          confidence: 0,
          rsCorrected: false,
          fallbackLevel: 'lost',
        ),
        diagnostics: ScanDiagnostics(
          detectMs: detectMs,
          depthMs: 0,
          extractMs: 0,
          decodeMs: 0,
          totalMs: DateTime.now().millisecondsSinceEpoch - wallStart,
        ),
      );
    }

    // -------------------------------------------------------------------------
    // Stage 2: depth estimation
    // -------------------------------------------------------------------------
    final t1 = DateTime.now().millisecondsSinceEpoch;
    final heightField = await depthEstimator.estimate(
      yuvBytes: yuvBytes,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      detection: detection,
    );
    final depthMs = DateTime.now().millisecondsSinceEpoch - t1;

    if (heightField == null) {
      return ScanResult(
        decode: const DecodeResult(
          itemId: null,
          confidence: 0,
          rsCorrected: false,
          fallbackLevel: 'lost',
        ),
        tileDetection: detection,
        depthSource: depthEstimator.source,
        diagnostics: ScanDiagnostics(
          detectMs: detectMs,
          depthMs: depthMs,
          extractMs: 0,
          decodeMs: 0,
          totalMs: DateTime.now().millisecondsSinceEpoch - wallStart,
        ),
      );
    }

    // -------------------------------------------------------------------------
    // Optional: wear estimation (adapts downstream thresholds)
    // -------------------------------------------------------------------------
    double estimatedWear = 0.0;
    if (wearEstimator != null) {
      final crop = _extract64x64(heightField);
      estimatedWear = await wearEstimator!.estimate(crop);
    }

    // -------------------------------------------------------------------------
    // Stage 3: channel extraction
    // -------------------------------------------------------------------------
    final t2 = DateTime.now().millisecondsSinceEpoch;
    final extraction = _extractor.extract(heightField);
    final extractMs = DateTime.now().millisecondsSinceEpoch - t2;

    // Optional: override macro with CNN classifier if confidence is high
    int? macroOverride;
    if (macroClassifier != null && extraction.count != null) {
      try {
        final crop = _extract64x64(heightField);
        final cls = await macroClassifier!.classify(crop);
        if (cls.confidence >= config.macroClassifierFallbackThreshold) {
          macroOverride = cls.family;
        }
      } catch (_) {
        // CNN not available — use harmonic correlator result
      }
    }

    // -------------------------------------------------------------------------
    // Stage 4: RS decode
    // -------------------------------------------------------------------------
    final t3 = DateTime.now().millisecondsSinceEpoch;
    final decode = _codec.decode(
      macro:  macroOverride ?? extraction.macro,
      count:  extraction.count,
      height: extraction.height,
      angles: extraction.angles,
      micro:  extraction.micro,
      rs:     extraction.rs,
    );
    final decodeMs = DateTime.now().millisecondsSinceEpoch - t3;

    return ScanResult(
      decode: decode,
      tileDetection: detection,
      depthSource: depthEstimator.source,
      extraction: extraction,
      estimatedWear: estimatedWear,
      diagnostics: ScanDiagnostics(
        detectMs: detectMs,
        depthMs: depthMs,
        extractMs: extractMs,
        decodeMs: decodeMs,
        totalMs: DateTime.now().millisecondsSinceEpoch - wallStart,
      ),
    );
  }

  /// Extract the 64×64 inner zone from a 128×128 field for CNN inputs.
  static Float32List _extract64x64(Float32List field, {int res = 128}) {
    const innerStart = 32;
    const innerEnd   = 96;
    const side = innerEnd - innerStart;
    final out = Float32List(side * side);
    double minV = double.infinity, maxV = double.negativeInfinity;
    int k = 0;
    for (int j = innerStart; j < innerEnd; j++) {
      for (int i = innerStart; i < innerEnd; i++) {
        final v = field[j * res + i];
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
        out[k++] = v;
      }
    }
    // Normalize to [0..1]
    final range = maxV - minV;
    if (range > 1e-9) {
      for (int i = 0; i < out.length; i++) {
        out[i] = (out[i] - minV) / range;
      }
    }
    return out;
  }

  void dispose() {
    detector.dispose();
    depthEstimator.dispose();
  }
}
