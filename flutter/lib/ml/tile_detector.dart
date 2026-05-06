/// YOLOv8-seg tile detector interface.
///
/// In production, [YoloTileDetector] wraps the on-device model via a
/// platform channel to CoreML (iOS) or TFLite (Android).
///
/// During development / testing, [MockTileDetector] returns a synthetic
/// detection so the rest of the pipeline can be exercised without ML models.
library;

import 'package:flutter/services.dart';

/// A detected tile in a camera frame.
class TileDetection {
  /// Bounding box in image-pixel coordinates.
  final Rect bbox;

  /// Center of the tile in image-pixel coordinates.
  final Offset center;

  /// Detection confidence [0..1] from YOLOv8.
  final double confidence;

  /// Segmentation mask as a flat Uint8List (width × height, 0=outside 255=inside).
  final Uint8List? mask;

  /// Estimated tile diameter in pixels (derived from mask or bbox).
  final double diameterPixels;

  const TileDetection({
    required this.bbox,
    required this.center,
    required this.confidence,
    required this.diameterPixels,
    this.mask,
  });
}

/// Abstract interface — platform implementations provide the actual model.
abstract class TileDetector {
  Future<TileDetection?> detect({
    required Uint8List yuvBytes,
    required int width,
    required int height,
  });

  /// Free native resources.
  void dispose();
}

/// Mock detector — always returns a centered detection with high confidence.
/// Used for integration tests and simulator builds.
class MockTileDetector implements TileDetector {
  @override
  Future<TileDetection?> detect({
    required Uint8List yuvBytes,
    required int width,
    required int height,
  }) async {
    // Simulate a tile centered in the frame at 30% of frame width
    final diameter = width * 0.30;
    final cx = width / 2.0;
    final cy = height / 2.0;
    return TileDetection(
      bbox: Rect.fromCenter(
        center: Offset(cx, cy),
        width: diameter,
        height: diameter,
      ),
      center: Offset(cx, cy),
      confidence: 0.97,
      diameterPixels: diameter,
    );
  }

  @override
  void dispose() {}
}

/// Production YOLOv8-nano-seg detector via platform channel.
///
/// The native plugin (SlcDetectorPlugin.kt / SlcDetectorPlugin.swift) loads
/// the bundled CoreML / TFLite model and returns a detection map.
class YoloTileDetector implements TileDetector {
  static const _ch = MethodChannel('slc/tile_detector');

  @override
  Future<TileDetection?> detect({
    required Uint8List yuvBytes,
    required int width,
    required int height,
  }) async {
    final result = await _ch.invokeMapMethod<String, dynamic>('detect', {
      'yuv': yuvBytes,
      'width': width,
      'height': height,
    });
    if (result == null) return null;
    final cx = (result['cx'] as num).toDouble();
    final cy = (result['cy'] as num).toDouble();
    final w  = (result['w']  as num).toDouble();
    final h  = (result['h']  as num).toDouble();
    return TileDetection(
      bbox: Rect.fromCenter(center: Offset(cx, cy), width: w, height: h),
      center: Offset(cx, cy),
      confidence:    (result['confidence']   as num).toDouble(),
      diameterPixels:(result['diameter_px']  as num).toDouble(),
      mask: result['mask'] as Uint8List?,
    );
  }

  @override
  void dispose() {}
}
