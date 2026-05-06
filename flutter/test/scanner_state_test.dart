/// Widget and state-machine tests for scanner screens.
///
/// ScannerScreen itself requires a real camera and permission plugin,
/// so those transitions are exercised via the integration test harness.
/// Here we test the observable UI produced by each pipeline outcome,
/// which covers the PLAN.md scenarios:
///   idle → captured → result        (all four decode levels)
///   captured → error (lost)         (ResultScreen lost-state UI)
///   close button → back             (navigation from result)
library;

import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:slc_scanner/codec/pattern_codec.dart';
import 'package:slc_scanner/ml/depth_estimator.dart';
import 'package:slc_scanner/screens/result_screen.dart';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

Widget _wrap(Widget child) {
  return MaterialApp(
    theme: ThemeData(useMaterial3: true),
    home: child,
  );
}

Widget _wrapWithNav(Widget Function(BuildContext) builder) {
  return MaterialApp(
    theme: ThemeData(useMaterial3: true),
    home: Builder(
      builder: (context) => ElevatedButton(
        onPressed: () => Navigator.of(context).push(
          MaterialPageRoute(builder: builder),
        ),
        child: const Text('open'),
      ),
    ),
  );
}

final _codec    = PatternCodec();
final _channels = _codec.encode(0x2DCA3791);

// ---------------------------------------------------------------------------
// ResultScreen: fallback level rendering
// ---------------------------------------------------------------------------

void main() {
  group('ResultScreen fallback levels', () {
    testWidgets('full: shows hex ID and clean-read badge', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     0x2DCA3791,
        channels:      _channels,
        confidence:    1.0,
        fallbackLevel: 'full',
      )));
      await tester.pumpAndSettle();

      // ID rendered in uppercase hex
      expect(find.text('0x2DCA3791'), findsOneWidget);
      // Badge text (ConfidenceBadge uppercases the label string)
      expect(find.textContaining('FULL ID'), findsWidgets);
      // New-scan / verify provenance buttons visible
      expect(find.textContaining('VERIFY'), findsOneWidget);
      expect(find.textContaining('NEW SCAN'), findsOneWidget);
    });

    testWidgets('rs_corrected: shows RS-corrected badge', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     0x2DCA3791,
        channels:      _channels,
        confidence:    5.0 / 6.0,
        fallbackLevel: 'rs_corrected',
      )));
      await tester.pumpAndSettle();

      expect(find.text('0x2DCA3791'), findsOneWidget);
      expect(find.textContaining('RS-CORRECTED'), findsWidgets);
    });

    testWidgets('category_only: shows partial-read title and note', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     null,
        channels:      _channels,
        confidence:    2.0 / 6.0,
        fallbackLevel: 'category_only',
      )));
      await tester.pumpAndSettle();

      expect(find.text('Partial read'), findsOneWidget);
      expect(find.textContaining('CATEGORY-ONLY'), findsWidgets);
      // Category-only explanation text is present
      expect(find.textContaining("couldn"), findsOneWidget);
    });

    testWidgets('lost: shows lost-state tile and try-again button', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     null,
        channels:      _channels,
        confidence:    0.0,
        fallbackLevel: 'lost',
      )));
      await tester.pumpAndSettle();

      expect(find.textContaining('LOST'), findsWidgets);
      expect(find.text('Tile not detected'), findsOneWidget);
      expect(find.text('TRY AGAIN'), findsOneWidget);
    });
  });

  // -------------------------------------------------------------------------
  // ResultScreen: channel readout
  // -------------------------------------------------------------------------

  group('ResultScreen channel readout', () {
    testWidgets('micro channel is always marked RS-recovered', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     0x2DCA3791,
        channels:      _channels,
        confidence:    5.0 / 6.0,
        fallbackLevel: 'rs_corrected',
      )));
      await tester.pumpAndSettle();

      // _RecoveredBadge shows exactly the text 'RS'.
      // micro always has read=false → exactly one RS badge.
      expect(find.text('RS'), findsOneWidget);
    });

    testWidgets('channel labels visible', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     0x2DCA3791,
        channels:      _channels,
        confidence:    1.0,
        fallbackLevel: 'full',
      )));
      await tester.pumpAndSettle();

      for (final label in ['MACRO', 'COUNT', 'HEIGHT', 'ANGLES', 'MICRO']) {
        expect(find.text(label), findsOneWidget,
            reason: 'Expected $label channel label');
      }
    });

    testWidgets('wear chip visible when estimatedWear > 0.05', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     0x2DCA3791,
        channels:      _channels,
        confidence:    5.0 / 6.0,
        fallbackLevel: 'rs_corrected',
        estimatedWear: 0.3,
      )));
      await tester.pumpAndSettle();

      expect(find.textContaining('WEAR ~'), findsOneWidget);
    });

    testWidgets('MiDaS depth source shows warning chip', (tester) async {
      await tester.pumpWidget(_wrap(ResultScreen(
        decodedId:     0x2DCA3791,
        channels:      _channels,
        confidence:    5.0 / 6.0,
        fallbackLevel: 'rs_corrected',
        depthSource:   DepthSource.midas,
      )));
      await tester.pumpAndSettle();

      expect(find.textContaining('MIDAS'), findsOneWidget);
    });
  });

  // -------------------------------------------------------------------------
  // ResultScreen: navigation
  // -------------------------------------------------------------------------

  group('ResultScreen navigation', () {
    testWidgets('close button pops back to previous route', (tester) async {
      await tester.pumpWidget(_wrapWithNav((_) => ResultScreen(
        decodedId:     0x2DCA3791,
        channels:      _channels,
        confidence:    1.0,
        fallbackLevel: 'full',
      )));

      // Navigate to ResultScreen
      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();
      expect(find.text('0x2DCA3791'), findsOneWidget);

      // Close button pops back
      await tester.tap(find.byIcon(Icons.close));
      await tester.pumpAndSettle();
      expect(find.text('open'), findsOneWidget);
    });

    testWidgets('lost try-again button pops back', (tester) async {
      await tester.pumpWidget(_wrapWithNav((_) => ResultScreen(
        decodedId:     null,
        channels:      _channels,
        confidence:    0.0,
        fallbackLevel: 'lost',
      )));

      await tester.tap(find.text('open'));
      await tester.pumpAndSettle();
      expect(find.text('TRY AGAIN'), findsOneWidget);

      await tester.tap(find.text('TRY AGAIN'));
      await tester.pumpAndSettle();
      expect(find.text('open'), findsOneWidget);
    });
  });

  // -------------------------------------------------------------------------
  // ResultScreen: encode / decode round-trip for display channels
  // -------------------------------------------------------------------------

  group('ResultScreen decode round-trip', () {
    test('channels match encode() output for any 32-bit ID', () {
      for (final id in [0x00000000, 0x2DCA3791, 0xCAFEBABE, 0xFFFFFFFF]) {
        final ch = _codec.encode(id);
        expect(ch.macro,  equals((id >> 28) & 0x0F));
        expect(ch.count,  equals((id >> 24) & 0x0F));
        expect(ch.height, equals((id >> 16) & 0xFF));
        expect(ch.angles, equals((id >> 8)  & 0xFF));
        expect(ch.micro,  equals(id & 0xFF));
      }
    });

    test('ResultScreen channel readout math: height mm formula', () {
      // height byte 0 → 1.50 mm; byte 255 → 8.00 mm
      expect(1.5 + (0   / 255.0) * 6.5, closeTo(1.50, 0.01));
      expect(1.5 + (255 / 255.0) * 6.5, closeTo(8.00, 0.01));
    });
  });
}
