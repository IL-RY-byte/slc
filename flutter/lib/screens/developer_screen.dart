/// Developer screen — hidden behind a triple-tap on the app version number.
///
/// Displays in-app telemetry counters: scan attempts, outcomes, per-stage
/// latency (p50 / p95 via a simple rolling window), and device capability flags.
///
/// No data is transmitted remotely unless the user opts in via the
/// "Share diagnostics" toggle. See PRIVACY.md.
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class DeveloperScreen extends StatefulWidget {
  const DeveloperScreen({super.key});

  @override
  State<DeveloperScreen> createState() => _DeveloperScreenState();
}

class _DeveloperScreenState extends State<DeveloperScreen> {
  static const _paper    = Color(0xFFF4EDE0);
  static const _ink      = Color(0xFF15110B);
  static const _inkSoft  = Color(0xFF4A4034);
  static const _rule     = Color(0xFFC4B89E);
  static const _accent   = Color(0xFFB8472B);
  static const _good     = Color(0xFF2D5A3D);

  Map<String, dynamic> _counters = {};
  bool _shareEnabled = false;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadCounters();
  }

  Future<void> _loadCounters() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _counters = {
        'scan_attempts':     prefs.getInt('tel_scan_attempts')      ?? 0,
        'full':              prefs.getInt('tel_full')               ?? 0,
        'rs_corrected':      prefs.getInt('tel_rs_corrected')       ?? 0,
        'category_only':     prefs.getInt('tel_category_only')      ?? 0,
        'lost':              prefs.getInt('tel_lost')               ?? 0,
        'p50_total_ms':      prefs.getInt('tel_p50_total_ms')       ?? 0,
        'p95_total_ms':      prefs.getInt('tel_p95_total_ms')       ?? 0,
        'depth_source_lidar':prefs.getInt('tel_depth_lidar')        ?? 0,
        'depth_source_arcore':prefs.getInt('tel_depth_arcore')      ?? 0,
        'depth_source_midas':prefs.getInt('tel_depth_midas')        ?? 0,
        'depth_source_mock': prefs.getInt('tel_depth_mock')         ?? 0,
      };
      _shareEnabled = prefs.getBool('tel_share_enabled') ?? false;
      _loading = false;
    });
  }

  Future<void> _clearCounters() async {
    final prefs = await SharedPreferences.getInstance();
    final keys = [
      'tel_scan_attempts', 'tel_full', 'tel_rs_corrected', 'tel_category_only',
      'tel_lost', 'tel_p50_total_ms', 'tel_p95_total_ms',
      'tel_depth_lidar', 'tel_depth_arcore', 'tel_depth_midas', 'tel_depth_mock',
    ];
    for (final k in keys) await prefs.remove(k);
    await _loadCounters();
  }

  Future<void> _toggleShare(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('tel_share_enabled', value);
    setState(() => _shareEnabled = value);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _paper,
      appBar: AppBar(
        backgroundColor: _paper,
        foregroundColor: _ink,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.close, color: _ink),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'DEVELOPER',
          style: TextStyle(
            fontFamily: 'JetBrainsMono',
            fontSize: 11,
            letterSpacing: 3.0,
            fontWeight: FontWeight.w500,
            color: _ink,
          ),
        ),
        centerTitle: true,
        bottom: const PreferredSize(
          preferredSize: Size.fromHeight(1.5),
          child: Divider(height: 1.5, thickness: 1.5, color: _ink),
        ),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: _accent))
          : SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(28, 24, 28, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionHeader('Scan outcomes'),
                  const SizedBox(height: 12),
                  _outcomeGrid(),
                  const SizedBox(height: 28),

                  _sectionHeader('Latency'),
                  const SizedBox(height: 12),
                  _latencyRow(),
                  const SizedBox(height: 28),

                  _sectionHeader('Depth source'),
                  const SizedBox(height: 12),
                  _depthGrid(),
                  const SizedBox(height: 28),

                  _sectionHeader('Diagnostics sharing'),
                  const SizedBox(height: 12),
                  _shareRow(),
                  const SizedBox(height: 28),

                  // Clear counters
                  GestureDetector(
                    onTap: _clearCounters,
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      decoration: BoxDecoration(
                        border: Border.all(color: _rule),
                      ),
                      alignment: Alignment.center,
                      child: const Text(
                        'CLEAR ALL COUNTERS',
                        style: TextStyle(
                          fontFamily: 'JetBrainsMono',
                          fontSize: 10,
                          letterSpacing: 2.4,
                          color: _inkSoft,
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 20),
                  _versionFooter(),
                ],
              ),
            ),
    );
  }

  Widget _sectionHeader(String text) {
    return Row(
      children: [
        Text(
          text,
          style: const TextStyle(
            fontFamily: 'CormorantGaramond',
            fontStyle: FontStyle.italic,
            fontSize: 18,
            fontWeight: FontWeight.w500,
            color: _ink,
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Divider(color: _rule, height: 1, thickness: 0.5),
        ),
      ],
    );
  }

  Widget _outcomeGrid() {
    final total = math.max(1, _counters['scan_attempts'] as int);
    final items = [
      ('Total attempts', '${_counters['scan_attempts']}',    _ink),
      ('Full reads',     '${_counters['full']}',             _good),
      ('RS-corrected',   '${_counters['rs_corrected']}',     _accent),
      ('Category-only',  '${_counters['category_only']}',    const Color(0xFF7A5811)),
      ('Lost',           '${_counters['lost']}',             const Color(0xFF8A3621)),
      ('ID accuracy',
        '${(((_counters['full'] + _counters['rs_corrected']) / total) * 100).toStringAsFixed(0)}%',
        _good),
    ];
    return _statGrid(items);
  }

  Widget _latencyRow() {
    return _statGrid([
      ('p50 total', '${_counters['p50_total_ms']} ms', _inkSoft),
      ('p95 total', '${_counters['p95_total_ms']} ms', _inkSoft),
    ]);
  }

  Widget _depthGrid() {
    return _statGrid([
      ('LiDAR',   '${_counters['depth_source_lidar']}',  _good),
      ('ARCore',  '${_counters['depth_source_arcore']}', _good),
      ('MiDaS',   '${_counters['depth_source_midas']}',  const Color(0xFF7A5811)),
      ('Mock',    '${_counters['depth_source_mock']}',   _inkSoft),
    ]);
  }

  Widget _shareRow() {
    return Row(
      children: [
        Expanded(
          child: Text(
            'Share aggregate counters with the atelier for ML retraining. '
            'No images or IDs are shared — counters only.',
            style: const TextStyle(
              fontFamily: 'CormorantGaramond',
              fontSize: 14,
              color: _inkSoft,
              height: 1.5,
            ),
          ),
        ),
        const SizedBox(width: 16),
        Switch(
          value: _shareEnabled,
          activeColor: _accent,
          onChanged: _toggleShare,
        ),
      ],
    );
  }

  Widget _statGrid(List<(String, String, Color)> items) {
    final rows = <Widget>[];
    for (int i = 0; i < items.length; i += 2) {
      rows.add(Row(
        children: [
          Expanded(child: _StatTile(label: items[i].$1, value: items[i].$2, color: items[i].$3)),
          const SizedBox(width: 1),
          if (i + 1 < items.length)
            Expanded(child: _StatTile(label: items[i+1].$1, value: items[i+1].$2, color: items[i+1].$3))
          else
            const Expanded(child: SizedBox()),
        ],
      ));
      if (i + 2 < items.length) rows.add(const SizedBox(height: 1));
    }
    return Container(
      decoration: BoxDecoration(
        color: _rule,
        border: Border.all(color: _rule),
      ),
      child: Column(children: rows),
    );
  }

  Widget _versionFooter() {
    return Text(
      'slc_scanner v0.2.0 · pipeline: mock · models: pending',
      style: const TextStyle(
        fontFamily: 'JetBrainsMono',
        fontSize: 9,
        letterSpacing: 1.4,
        color: _inkSoft,
      ),
    );
  }
}

class _StatTile extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _StatTile({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFFF4EDE0),
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontFamily: 'JetBrainsMono',
              fontSize: 8.5,
              letterSpacing: 2.0,
              color: Color(0xFF4A4034),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontFamily: 'JetBrainsMono',
              fontSize: 20,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
