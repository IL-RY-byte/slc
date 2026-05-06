/// Depth estimator interface — ARKit LiDAR / ARCore / MiDaS-small fallback.
///
/// All implementations return a 128×128 Float32List height field in mm,
/// calibrated to the known 30mm tile diameter using the segmentation mask.
library;

import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/services.dart';

import 'tile_detector.dart';
import '../perception/height_field_renderer.dart' show HeightFieldRenderer;
import '../codec/pattern_codec.dart';

/// Source of depth data used for a particular scan.
enum DepthSource {
  lidar,    // ARKit LiDAR (iPhone Pro / iPad Pro)
  arCore,   // ARCore Depth API (supported Android devices)
  midas,    // MiDaS-small monocular estimation
  mock,     // Synthetic (test only)
}

/// Abstract depth estimation interface.
abstract class DepthEstimator {
  /// Which depth source this estimator uses.
  DepthSource get source;

  /// Estimate depth from [yuvBytes] and crop to tile region indicated by
  /// [detection]. Returns a 128×128 Float32List height field in mm, or null
  /// if depth estimation fails.
  Future<Float32List?> estimate({
    required Uint8List yuvBytes,
    required int imageWidth,
    required int imageHeight,
    required TileDetection detection,
  });

  void dispose();
}

/// Mock estimator — renders the height field from known channels.
///
/// Used for integration tests: inject known [Channels] and verify that the
/// full pipeline (render → extract → decode) recovers the correct item_id.
class MockDepthEstimator implements DepthEstimator {
  final Channels channels;
  final double wear;
  final _renderer = const HeightFieldRenderer();

  MockDepthEstimator({required this.channels, this.wear = 0.0});

  @override
  DepthSource get source => DepthSource.mock;

  @override
  Future<Float32List?> estimate({
    required Uint8List yuvBytes,
    required int imageWidth,
    required int imageHeight,
    required TileDetection detection,
  }) async {
    return _renderer.render(channels, wear: wear);
  }

  @override
  void dispose() {}
}

/// ARKit LiDAR depth (iOS only, <30ms on iPhone 12 Pro+).
///
/// Delegates to SlcDetectorPlugin via the slc/depth_estimator channel.
/// Returns null if LiDAR is unavailable; caller should fall back to [MidasDepthEstimator].
class ArKitDepthEstimator implements DepthEstimator {
  static const _ch = MethodChannel('slc/depth_estimator');

  @override
  DepthSource get source => DepthSource.lidar;

  @override
  Future<Float32List?> estimate({
    required Uint8List yuvBytes,
    required int imageWidth,
    required int imageHeight,
    required TileDetection detection,
  }) async {
    try {
      final raw = await _ch.invokeMethod<Uint8List>('estimate', {
        'cx':          detection.center.dx,
        'cy':          detection.center.dy,
        'w':           detection.bbox.width,
        'h':           detection.bbox.height,
        'diameter_px': detection.diameterPixels,
      });
      if (raw == null) return null;
      return raw.buffer.asFloat32List();
    } on PlatformException catch (e) {
      if (e.code == 'NO_DEPTH') return null;  // LiDAR unavailable — let caller fall back
      rethrow;
    }
  }

  @override
  void dispose() {}
}

/// ARCore Depth API (supported Android devices).
///
/// Delegates to SlcDetectorPlugin via the slc/depth_estimator channel.
class ArCoreDepthEstimator implements DepthEstimator {
  static const _ch = MethodChannel('slc/depth_estimator');

  @override
  DepthSource get source => DepthSource.arCore;

  @override
  Future<Float32List?> estimate({
    required Uint8List yuvBytes,
    required int imageWidth,
    required int imageHeight,
    required TileDetection detection,
  }) async {
    try {
      final raw = await _ch.invokeMethod<Uint8List>('estimate', {
        'cx':          detection.center.dx,
        'cy':          detection.center.dy,
        'w':           detection.bbox.width,
        'h':           detection.bbox.height,
        'diameter_px': detection.diameterPixels,
      });
      if (raw == null) return null;
      return raw.buffer.asFloat32List();
    } on PlatformException catch (e) {
      if (e.code == 'NO_DEPTH') return null;
      rethrow;
    }
  }

  @override
  void dispose() {}
}

/// MiDaS-small monocular depth (non-LiDAR fallback, ~100ms).
///
/// Delegates to the native MiDaS CoreML / TFLite plugin via slc/midas_depth.
/// Required assets: assets/models/midas_small.mlpackage (iOS) or .tflite (Android).
class MidasDepthEstimator implements DepthEstimator {
  static const _ch = MethodChannel('slc/midas_depth');

  @override
  DepthSource get source => DepthSource.midas;

  @override
  Future<Float32List?> estimate({
    required Uint8List yuvBytes,
    required int imageWidth,
    required int imageHeight,
    required TileDetection detection,
  }) async {
    final raw = await _ch.invokeMethod<Uint8List>('estimate', {
      'yuv':         yuvBytes,
      'width':       imageWidth,
      'height':      imageHeight,
      'cx':          detection.center.dx,
      'cy':          detection.center.dy,
      'w':           detection.bbox.width,
      'h':           detection.bbox.height,
      'diameter_px': detection.diameterPixels,
    });
    if (raw == null) return null;
    return raw.buffer.asFloat32List();
  }

  @override
  void dispose() {}
}

/// Tries each estimator in order, returning the first non-null result.
///
/// Typical usage: FallbackDepthEstimator([ArKitDepthEstimator(), MidasDepthEstimator()])
/// so LiDAR is preferred on Pro devices and MiDaS is the fallback.
class FallbackDepthEstimator implements DepthEstimator {
  final List<DepthEstimator> _chain;
  DepthSource? _lastUsedSource;

  FallbackDepthEstimator(this._chain) : assert(_chain.isNotEmpty);

  @override
  DepthSource get source => _lastUsedSource ?? _chain.first.source;

  @override
  Future<Float32List?> estimate({
    required Uint8List yuvBytes,
    required int imageWidth,
    required int imageHeight,
    required TileDetection detection,
  }) async {
    for (final est in _chain) {
      try {
        final result = await est.estimate(
          yuvBytes: yuvBytes,
          imageWidth: imageWidth,
          imageHeight: imageHeight,
          detection: detection,
        );
        if (result != null) {
          _lastUsedSource = est.source;
          return result;
        }
      } catch (_) {
        // Try next estimator in chain
      }
    }
    return null;
  }

  @override
  void dispose() {
    for (final est in _chain) {
      est.dispose();
    }
  }
}
