/// Macro family classifier interface.
///
/// In production, [CnnMacroClassifier] wraps the trained CNN exported to
/// CoreML / TFLite via a platform channel. When CNN confidence < 0.70 or the
/// channel throws, the pipeline falls back to the harmonic correlator result
/// from [channel_extractor.dart].
///
/// Platform channel contract (slc/macro_classifier):
///   Input:  {'crop': List<double> of length 4096 (64x64 normalized)}
///   Output: {'family': int 0..15, 'confidence': double 0..1}
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';

const double cnnFallbackThreshold = 0.70;

class MacroClassification {
  final int family;         // 0..15
  final double confidence;  // 0..1

  const MacroClassification({required this.family, required this.confidence});
}

abstract class MacroClassifier {
  /// Classify the macro family from a 64x64 height field crop.
  ///
  /// The crop should be the inner zone of the 128x128 field (pixels 32..96 in
  /// both axes), normalized to [0..1]: (v - v_min) / (v_max - v_min).
  Future<MacroClassification> classify(Float32List crop64x64);
}

/// CNN-based macro classifier via platform channel.
///
/// Sends the 64x64 crop to the native iOS (CoreML) or Android (TFLite) plugin
/// and returns the top-1 class with its softmax confidence.
class CnnMacroClassifier implements MacroClassifier {
  static const _channel = MethodChannel('slc/macro_classifier');

  @override
  Future<MacroClassification> classify(Float32List crop64x64) async {
    final result = await _channel.invokeMapMethod<String, dynamic>(
      'classify',
      {'crop': crop64x64.toList()},
    );
    return MacroClassification(
      family: (result!['family'] as num).toInt(),
      confidence: (result['confidence'] as num).toDouble(),
    );
  }
}

/// Returns a fixed mock classification — for widget tests and simulator runs.
class MockMacroClassifier implements MacroClassifier {
  final int _family;
  const MockMacroClassifier(this._family);

  @override
  Future<MacroClassification> classify(Float32List crop64x64) async {
    return MacroClassification(family: _family, confidence: 0.99);
  }
}
