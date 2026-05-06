/// Wear estimator interface (Enhancement 2).
///
/// Outputs an estimated wear factor [0..1] from a 64x64 inner-zone height
/// field crop. Used by the pipeline to:
///   1. Adapt RS-marker presence threshold (higher wear -> lower threshold)
///   2. Surface wear context in the result UX
///
/// Platform channel contract (slc/wear_estimator):
///   Input:  {'crop': List<double> of length 4096 (64x64 normalized)}
///   Output: {'wear': double 0..1}
library;

import 'dart:typed_data';

import 'package:flutter/services.dart';

abstract class WearEstimator {
  /// Estimate wear from a 64x64 normalized height field crop.
  /// Returns [0..1]: 0 = brand new, 1 = unreadable.
  Future<double> estimate(Float32List crop64x64);
}

/// Hardcoded zero wear -- for initial release and testing.
class MockWearEstimator implements WearEstimator {
  const MockWearEstimator();

  @override
  Future<double> estimate(Float32List crop64x64) async => 0.0;
}

/// CNN wear regressor via platform channel.
class CnnWearEstimator implements WearEstimator {
  static const _channel = MethodChannel('slc/wear_estimator');

  @override
  Future<double> estimate(Float32List crop64x64) async {
    final result = await _channel.invokeMapMethod<String, dynamic>(
      'estimate',
      {'crop': crop64x64.toList()},
    );
    return (result!['wear'] as num).toDouble();
  }
}
