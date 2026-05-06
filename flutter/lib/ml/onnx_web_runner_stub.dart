// Stub for non-web platforms — never instantiated; native builds use
// platform-channel implementations instead.
import 'dart:typed_data';
import 'macro_classifier.dart';
import 'wear_estimator.dart';
import 'depth_estimator.dart';
import 'tile_detector.dart';

class OnnxWebMacroClassifier implements MacroClassifier {
  @override
  Future<MacroClassification> classify(Float32List crop64x64) =>
      throw UnsupportedError('web-only');
}

class OnnxWebWearEstimator implements WearEstimator {
  @override
  Future<double> estimate(Float32List crop64x64) =>
      throw UnsupportedError('web-only');
}

class MidasWebDepthEstimator implements DepthEstimator {
  @override DepthSource get source => DepthSource.midas;
  @override
  Future<Float32List?> estimate({
    required Uint8List yuvBytes, required int imageWidth,
    required int imageHeight,   required TileDetection detection,
  }) => throw UnsupportedError('web-only');
  @override void dispose() {}
}

class YoloWebTileDetector implements TileDetector {
  @override
  Future<TileDetection?> detect({
    required Uint8List yuvBytes, required int width, required int height,
  }) => throw UnsupportedError('web-only');
  @override void dispose() {}
}
