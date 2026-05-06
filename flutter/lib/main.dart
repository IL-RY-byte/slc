import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/scanner_screen.dart';
import 'screens/developer_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SlcScannerApp());
}

class SlcScannerApp extends StatelessWidget {
  const SlcScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Second-Life Couture',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFB8472B),
          surface: const Color(0xFFEBE2D0),
          onSurface: const Color(0xFF15110B),
        ),
        scaffoldBackgroundColor: const Color(0xFFF4EDE0),
        textTheme: GoogleFonts.cormorantGaramondTextTheme(
          TextTheme(
            labelSmall: GoogleFonts.jetBrainsMono(letterSpacing: 2.0),
          ),
        ),
        useMaterial3: true,
      ),
      // Navigator key for developer screen access
      navigatorKey: _navigatorKey,
      home: const _AppRoot(),
    );
  }
}

final _navigatorKey = GlobalKey<NavigatorState>();

/// Root widget that embeds the scanner screen and a triple-tap zone
/// in the top-left corner to access the developer screen.
class _AppRoot extends StatelessWidget {
  const _AppRoot();

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        const ScannerScreen(),
        // Triple-tap developer screen trigger — top-left 44×44 invisible zone
        Positioned(
          top: 0, left: 0,
          child: GestureDetector(
            onTap: () {}, // absorb single taps silently
            child: _TripleTapZone(
              onTripleTap: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => const DeveloperScreen(),
                  ),
                );
              },
              child: const SizedBox(width: 44, height: 44),
            ),
          ),
        ),
      ],
    );
  }
}

class _TripleTapZone extends StatefulWidget {
  final VoidCallback onTripleTap;
  final Widget child;
  const _TripleTapZone({required this.onTripleTap, required this.child});

  @override
  State<_TripleTapZone> createState() => _TripleTapZoneState();
}

class _TripleTapZoneState extends State<_TripleTapZone> {
  int _tapCount = 0;
  DateTime _lastTap = DateTime.fromMillisecondsSinceEpoch(0);

  void _handleTap() {
    final now = DateTime.now();
    if (now.difference(_lastTap).inMilliseconds > 800) {
      _tapCount = 1;
    } else {
      _tapCount++;
    }
    _lastTap = now;
    if (_tapCount >= 3) {
      _tapCount = 0;
      widget.onTripleTap();
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _handleTap,
      behavior: HitTestBehavior.opaque,
      child: widget.child,
    );
  }
}
