import 'dart:js_interop';
import 'dart:typed_data';
import 'dart:ui' show Rect, Offset;

import '../ml/tile_detector.dart';
import 'macro_classifier.dart';
import 'wear_estimator.dart';
import '../ml/depth_estimator.dart';

// ---------------------------------------------------------------------------
// JS interop — macro CNN
// ---------------------------------------------------------------------------

@JS('_slcOrt.runInference')
external JSPromise<_CnnResult> _jsRunInference(JSFloat32Array data);

extension type _CnnResult._(JSObject _) implements JSObject {
  external int get macroClass;
  external double get macroConf;
  external double get wear;
}

// ---------------------------------------------------------------------------
// JS interop — MiDaS depth
// ---------------------------------------------------------------------------

@JS('_slcOrt.runMidas')
external JSPromise<JSFloat32Array> _jsRunMidas(
  JSUint8Array jpegBytes,
  JSNumber tileX,
  JSNumber tileY,
  JSNumber tileW,
  JSNumber tileH,
);

// ---------------------------------------------------------------------------
// Macro CNN session (shared, lazy-init)
// ---------------------------------------------------------------------------

class _CnnSession {
  static Future<({int macroClass, double macroConf, double wear})> run(
      Float32List crop64x64) async {
    final result = await _jsRunInference(crop64x64.toJS).toDart;
    return (
      macroClass: result.macroClass,
      macroConf:  result.macroConf,
      wear:       result.wear,
    );
  }
}

// ---------------------------------------------------------------------------
// MacroClassifier backed by ONNX Runtime Web
// ---------------------------------------------------------------------------

class OnnxWebMacroClassifier implements MacroClassifier {
  @override
  Future<MacroClassification> classify(Float32List crop64x64) async {
    final r = await _CnnSession.run(crop64x64);
    return MacroClassification(family: r.macroClass, confidence: r.macroConf);
  }
}

// ---------------------------------------------------------------------------
// WearEstimator backed by ONNX Runtime Web (same forward pass as macro)
// ---------------------------------------------------------------------------

class OnnxWebWearEstimator implements WearEstimator {
  static double? _cachedWear;
  static Float32List? _cachedInput;

  @override
  Future<double> estimate(Float32List crop64x64) async {
    if (identical(_cachedInput, crop64x64)) return _cachedWear!;
    final r = await _CnnSession.run(crop64x64);
    _cachedInput = crop64x64;
    _cachedWear  = r.wear;
    return r.wear;
  }
}

// ---------------------------------------------------------------------------
// MiDaS depth estimator — runs on real JPEG bytes captured by the camera.
// Falls back to null if the model isn't loaded yet (pipeline handles it).
// ---------------------------------------------------------------------------

class MidasWebDepthEstimator implements DepthEstimator {
  @override
  DepthSource get source => DepthSource.midas;

  @override
  Future<Float32List?> estimate({
    required Uint8List yuvBytes,
    required int imageWidth,
    required int imageHeight,
    required TileDetection detection,
  }) async {
    if (yuvBytes.isEmpty) return null;
    try {
      final jsResult = await _jsRunMidas(
        yuvBytes.toJS,
        detection.center.dx.toJS,
        detection.center.dy.toJS,
        detection.bbox.width.toJS,
        detection.bbox.height.toJS,
      ).toDart;
      return jsResult.toDart;
    } catch (_) {
      return null;
    }
  }

  @override
  void dispose() {}
}

// ---------------------------------------------------------------------------
// JS interop — YOLOv8 tile detector
// ---------------------------------------------------------------------------

@JS('_slcOrt.runYolo')
external JSPromise<_YoloResult?> _jsRunYolo(
  JSUint8Array jpegBytes,
  JSNumber imgW,
  JSNumber imgH,
);

extension type _YoloResult._(JSObject _) implements JSObject {
  external double get cx;
  external double get cy;
  external double get w;
  external double get h;
  external double get conf;
}

// ---------------------------------------------------------------------------
// YOLOv8 tile detector — ONNX Runtime Web
// ---------------------------------------------------------------------------

class YoloWebTileDetector implements TileDetector {
  @override
  Future<TileDetection?> detect({
    required Uint8List yuvBytes,
    required int width,
    required int height,
  }) async {
    if (yuvBytes.isEmpty) return _mockCentered(width, height);
    try {
      final r = await _jsRunYolo(
        yuvBytes.toJS,
        width.toDouble().toJS,
        height.toDouble().toJS,
      ).toDart;
      if (r == null) return _mockCentered(width, height);
      final center = Offset(r.cx, r.cy);
      return TileDetection(
        bbox: Rect.fromCenter(center: center, width: r.w, height: r.h),
        center: center,
        confidence: r.conf,
        diameterPixels: (r.w + r.h) / 2,
      );
    } catch (_) {
      return _mockCentered(width, height);
    }
  }

  static TileDetection _mockCentered(int w, int h) {
    final d = w * 0.30;
    final c = Offset(w / 2.0, h / 2.0);
    return TileDetection(
      bbox: Rect.fromCenter(center: c, width: d, height: d),
      center: c,
      confidence: 0.0,
      diameterPixels: d,
    );
  }

  @override
  void dispose() {}
}
