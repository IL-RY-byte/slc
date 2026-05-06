/// In-app telemetry counters — stored locally via SharedPreferences.
///
/// No data is transmitted remotely by default. See PRIVACY.md.
/// Call [Telemetry.recordScan] from the scan pipeline after each attempt.
library;

import 'dart:collection';

import 'package:shared_preferences/shared_preferences.dart';

import 'ml/depth_estimator.dart';

class Telemetry {
  Telemetry._();
  static final Telemetry instance = Telemetry._();

  // Rolling window for latency percentile estimation (last 200 scans)
  final Queue<int> _latencySamples = Queue();
  static const _windowSize = 200;

  Future<void> recordScan({
    required String fallbackLevel,
    required int totalMs,
    required DepthSource depthSource,
  }) async {
    final prefs = await SharedPreferences.getInstance();

    // Increment attempt counter
    final attempts = (prefs.getInt('tel_scan_attempts') ?? 0) + 1;
    await prefs.setInt('tel_scan_attempts', attempts);

    // Increment outcome counter
    final outcomeKey = 'tel_$fallbackLevel';
    await prefs.setInt(outcomeKey, (prefs.getInt(outcomeKey) ?? 0) + 1);

    // Depth source counter
    final depthKey = switch (depthSource) {
      DepthSource.lidar  => 'tel_depth_lidar',
      DepthSource.arCore => 'tel_depth_arcore',
      DepthSource.midas  => 'tel_depth_midas',
      DepthSource.mock   => 'tel_depth_mock',
    };
    await prefs.setInt(depthKey, (prefs.getInt(depthKey) ?? 0) + 1);

    // Update rolling latency window
    _latencySamples.add(totalMs);
    if (_latencySamples.length > _windowSize) _latencySamples.removeFirst();

    final sorted = _latencySamples.toList()..sort();
    if (sorted.isNotEmpty) {
      final p50idx = (sorted.length * 0.5).floor().clamp(0, sorted.length - 1);
      final p95idx = (sorted.length * 0.95).floor().clamp(0, sorted.length - 1);
      await prefs.setInt('tel_p50_total_ms', sorted[p50idx]);
      await prefs.setInt('tel_p95_total_ms', sorted[p95idx]);
    }
  }
}
