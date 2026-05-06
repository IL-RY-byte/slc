import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';


import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'package:flutter/foundation.dart' show kIsWeb, kReleaseMode;

import '../codec/pattern_codec.dart';
import '../ml/macro_classifier.dart';
import '../ml/tile_detector.dart';
import '../ml/depth_estimator.dart';
import '../ml/wear_estimator.dart';
import '../ml/onnx_web_runner.dart'
    if (dart.library.io) '../ml/onnx_web_runner_stub.dart';
import '../pipeline/scan_pipeline.dart';
import '../telemetry.dart';
import 'result_screen.dart';

// ---------------------------------------------------------------------------
// Scanner state machine
// ---------------------------------------------------------------------------

enum _ScanState {
  idle,       // camera live, no tile detected
  detecting,  // tile visible, confidence accumulating
  captured,   // shutter pressed, pipeline running
  error,      // camera or permission failure
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen>
    with TickerProviderStateMixin {
  CameraController? _camera;
  bool _initializing = true;
  _ScanState _state = _ScanState.idle;
  String? _errorMessage;

  // Animated reticle
  late final AnimationController _reticlePulse;
  late final AnimationController _scanLinePulse;

  // Detection confidence state
  double _detectionConfidence = 0.0;
  TileDetection? _currentDetection;
  int _consecutiveHighConfidenceFrames = 0;
  static const _autoCaptureMsThreshold = 0.85;
  static const _autoCapturFrames = 3;

  // Frame processing
  bool _processingFrame = false;

  // Pipeline (swap real implementations here for production)
  late ScanPipeline _pipeline;

  // Telemetry counters
  int _scanAttempts = 0;
  final Map<String, int> _outcomeCounters = {
    'full': 0, 'rs_corrected': 0, 'category_only': 0, 'lost': 0,
  };

  @override
  void initState() {
    super.initState();
    _reticlePulse =
        AnimationController(vsync: this, duration: const Duration(seconds: 3))
          ..repeat(reverse: true);
    _scanLinePulse =
        AnimationController(vsync: this, duration: const Duration(seconds: 2))
          ..repeat();
    _buildPipeline();
    _bootCamera();
    // Auto-capture only in native debug/simulator — on web the user presses
    // the shutter button after pointing the camera at a tile.
    if (!kIsWeb && !kReleaseMode) _scheduleDebugCapture();
  }

  // Debug-only: simulate tile detection frames and auto-capture without camera.
  void _scheduleDebugCapture() {
    Future.delayed(const Duration(milliseconds: 800), () async {
      if (!mounted) return;
      setState(() {
        _state = _ScanState.detecting;
        _detectionConfidence = 0.97;
        _currentDetection = TileDetection(
          bbox: Rect.fromCenter(
            center: const Offset(320, 480),
            width: 200,
            height: 200,
          ),
          center: const Offset(320, 480),
          confidence: 0.97,
          diameterPixels: 200,
        );
      });
      for (int i = 0; i < _autoCapturFrames; i++) {
        await Future.delayed(const Duration(milliseconds: 120));
        if (!mounted) return;
        _consecutiveHighConfidenceFrames++;
      }
      if (!mounted || _state == _ScanState.captured) return;
      await _capture(autoCapture: true);
    });
  }

  void _buildPipeline() {
    if (kIsWeb) {
      // Web: YOLOv8 tile detection + MiDaS depth + CNN inference via ONNX Runtime Web.
      // YoloWebTileDetector falls back to centred mock when model not loaded or
      // confidence is below threshold, so the app stays functional during first load.
      _pipeline = ScanPipeline(
        detector:        YoloWebTileDetector(),
        depthEstimator:  MidasWebDepthEstimator(),
        macroClassifier: OnnxWebMacroClassifier(),
        wearEstimator:   OnnxWebWearEstimator(),
      );
    } else if (kReleaseMode) {
      // Native release: real ML models via platform channels.
      _pipeline = ScanPipeline(
        detector: YoloTileDetector(),
        depthEstimator: FallbackDepthEstimator([
          ArKitDepthEstimator(),
          MidasDepthEstimator(),
        ]),
        macroClassifier: CnnMacroClassifier(),
        wearEstimator:   CnnWearEstimator(),
      );
    } else {
      // Native debug / simulator: mocks only.
      final testChannels = PatternCodec().encode(0x2DCA3791);
      _pipeline = ScanPipeline(
        detector:        MockTileDetector(),
        depthEstimator:  MockDepthEstimator(channels: testChannels),
        macroClassifier: MockMacroClassifier(2),
        wearEstimator:   MockWearEstimator(),
      );
    }
  }

  Future<void> _bootCamera() async {
    // On web the browser handles camera permission via getUserMedia — no need
    // to pre-request with permission_handler (which has no web implementation).
    if (!kIsWeb) {
      final status = await Permission.camera.request();
      if (!status.isGranted) {
        setState(() {
          _initializing = false;
          _state = _ScanState.error;
          _errorMessage = 'Camera permission required to scan patterns.';
        });
        return;
      }
    }
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _initializing = false;
          _state = _ScanState.error;
          _errorMessage = 'No camera available on this device.';
        });
        return;
      }
      final rear = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final ctrl = CameraController(
        rear,
        ResolutionPreset.high,
        enableAudio: false,
        // yuv420 is not supported on web — omit the format on web so the
        // camera_web plugin uses its default (JPEG/canvas capture).
        imageFormatGroup: kIsWeb ? null : ImageFormatGroup.yuv420,
      );
      await ctrl.initialize();
      if (!mounted) return;
      setState(() {
        _camera = ctrl;
        _initializing = false;
        _state = _ScanState.idle;
      });
      // Frame streaming for real-time detection (native only).
      // On web, startImageStream throws UnimplementedError — the shutter button
      // triggers takePicture() instead.
      if (!kIsWeb) {
        try {
          await _camera!.startImageStream(_onCameraFrame);
        } on UnimplementedError {
          // Older camera_web builds: fall through, button capture still works.
        }
      }
    } catch (e) {
      setState(() {
        _initializing = false;
        _state = _ScanState.error;
        _errorMessage = 'Camera failed to start: $e';
      });
    }
  }

  @override
  void dispose() {
    _reticlePulse.dispose();
    _scanLinePulse.dispose();
    _camera?.dispose();
    _pipeline.dispose();
    super.dispose();
  }

  // -------------------------------------------------------------------------
  // Frame processing — run tile detection on every frame
  // -------------------------------------------------------------------------

  void _onCameraFrame(CameraImage image) {
    if (_processingFrame || _state == _ScanState.captured) return;
    _processingFrame = true;
    _runDetection(image).then((_) {
      _processingFrame = false;
    });
  }

  Future<void> _runDetection(CameraImage image) async {
    try {
      // Build YUV bytes for the detector
      final yuvBytes = _flattenYuv(image);
      final detection = await _pipeline.detector.detect(
        yuvBytes: yuvBytes,
        width: image.width,
        height: image.height,
      );

      if (!mounted) return;

      if (detection == null || detection.confidence < 0.3) {
        setState(() {
          _currentDetection = null;
          _detectionConfidence = 0.0;
          _consecutiveHighConfidenceFrames = 0;
          if (_state == _ScanState.detecting) _state = _ScanState.idle;
        });
        return;
      }

      // Smooth confidence
      final smoothed = _detectionConfidence * 0.6 + detection.confidence * 0.4;

      setState(() {
        _currentDetection = detection;
        _detectionConfidence = smoothed;
        _state = smoothed > 0.4
            ? _ScanState.detecting
            : _ScanState.idle;
      });

      // Auto-capture: require 3 consecutive high-confidence frames
      if (smoothed >= _autoCaptureMsThreshold) {
        _consecutiveHighConfidenceFrames++;
        if (_consecutiveHighConfidenceFrames >= _autoCapturFrames) {
          _consecutiveHighConfidenceFrames = 0;
          await _capture(autoCapture: true);
        }
      } else {
        _consecutiveHighConfidenceFrames = 0;
      }
    } catch (_) {
      _processingFrame = false;
    }
  }

  static Uint8List _flattenYuv(CameraImage image) {
    // Concatenate all plane bytes — detector implementations extract what they need
    int totalSize = 0;
    for (final p in image.planes) totalSize += p.bytes.length;
    final out = Uint8List(totalSize);
    int offset = 0;
    for (final p in image.planes) {
      out.setAll(offset, p.bytes);
      offset += p.bytes.length;
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Capture + decode
  // -------------------------------------------------------------------------

  Future<void> _capture({bool autoCapture = false}) async {
    if (_state == _ScanState.captured) return;
    setState(() {
      _state = _ScanState.captured;
      _scanAttempts++;
    });

    try {
      // Stop frame stream during decode (native only — web never starts it).
      if (!kIsWeb) {
        try { await _camera?.stopImageStream(); } catch (_) {}
      }

      // Capture a real image frame when camera is available.
      // On web: takePicture() works and gives us JPEG bytes for the pipeline.
      // On native release: same. On native debug: skip (uses mock pipeline).
      Uint8List yuvBytes = Uint8List(0);
      if (_camera != null && (kIsWeb || kReleaseMode)) {
        try {
          final file = await _camera!.takePicture()
              .timeout(const Duration(seconds: 5));
          yuvBytes = await file.readAsBytes();
        } catch (_) {
          yuvBytes = Uint8List(0);
        }
      }

      final result = await _pipeline.run(
        yuvBytes,
        _camera?.value.previewSize?.width.toInt() ?? 640,
        _camera?.value.previewSize?.height.toInt() ?? 480,
      );

      _outcomeCounters[result.decode.fallbackLevel] =
          (_outcomeCounters[result.decode.fallbackLevel] ?? 0) + 1;

      // Fire-and-forget telemetry (in-app only by default)
      unawaited(Telemetry.instance.recordScan(
        fallbackLevel: result.decode.fallbackLevel,
        totalMs: result.diagnostics.totalMs,
        depthSource: result.depthSource ?? DepthSource.mock,
      ));

      if (!mounted) return;

      // Derive Channels for the result screen — use encoded channels when available
      final Channels displayChannels;
      if (result.decode.itemId != null) {
        displayChannels = PatternCodec().encode(result.decode.itemId!);
      } else if (result.extraction != null) {
        final ex = result.extraction!;
        displayChannels = Channels(
          macro:  ex.macro  ?? 0,
          count:  ex.count  ?? 0,
          height: ex.height ?? 0,
          angles: ex.angles ?? 0,
          micro:  ex.micro  ?? 0,
          rs:     ex.rs     ?? Uint8List(4),
        );
      } else {
        displayChannels = PatternCodec().encode(0);
      }

      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            decodedId:      result.decode.itemId,
            channels:       displayChannels,
            confidence:     result.decode.confidence,
            fallbackLevel:  result.decode.fallbackLevel,
            estimatedWear:  result.estimatedWear,
            depthSource:    result.depthSource,
            diagnostics:    result.diagnostics.toMap(),
          ),
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Scan error: $e'),
            duration: const Duration(seconds: 15),
            backgroundColor: const Color(0xFF8B0000),
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _state = _ScanState.idle;
          _detectionConfidence = 0.0;
          _currentDetection = null;
          _consecutiveHighConfidenceFrames = 0;
        });
        // Restart frame stream
        try {
          await _camera?.startImageStream(_onCameraFrame);
        } catch (_) {}
      }
    }
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF15110B),
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            _buildCameraLayer(),
            _buildOverlay(),
            if (_currentDetection != null) _buildDetectionBox(),
            _buildHeader(),
            _buildShutterBar(),
            if (_state == _ScanState.captured) _buildCapturingLayer(),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraLayer() {
    if (_initializing) {
      return const Center(
        child: CircularProgressIndicator(color: Color(0xFFB8472B)),
      );
    }
    if (_state == _ScanState.error) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(36),
          child: Text(
            _errorMessage ?? 'Unknown error.',
            style: const TextStyle(
              color: Color(0xFFF4EDE0),
              fontFamily: 'CormorantGaramond',
              fontStyle: FontStyle.italic,
              fontSize: 18,
            ),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    if (_camera == null || !_camera!.value.isInitialized) {
      return const SizedBox.shrink();
    }
    final size = MediaQuery.of(context).size;
    final scale = size.aspectRatio * _camera!.value.aspectRatio;
    return Transform.scale(
      scale: scale < 1 ? 1 / scale : scale,
      child: Center(child: CameraPreview(_camera!)),
    );
  }

  Widget _buildOverlay() {
    return IgnorePointer(
      child: AnimatedBuilder(
        animation: Listenable.merge([_reticlePulse, _scanLinePulse]),
        builder: (context, _) {
          return CustomPaint(
            size: Size.infinite,
            painter: _ReticlePainter(
              pulse:      _reticlePulse.value,
              scanLine:   _scanLinePulse.value,
              confidence: _detectionConfidence,
              state:      _state,
            ),
          );
        },
      ),
    );
  }

  /// Overlay showing the tile detection bounding box
  Widget _buildDetectionBox() {
    final detection = _currentDetection!;
    final size = MediaQuery.of(context).size;
    // Scale bbox from camera coords to screen coords
    final camW = _camera?.value.previewSize?.width ?? size.width;
    final camH = _camera?.value.previewSize?.height ?? size.height;
    final scaleX = size.width  / camH; // note: portrait camera rotates axes
    final scaleY = size.height / camW;

    return IgnorePointer(
      child: CustomPaint(
        size: Size.infinite,
        painter: _DetectionBoxPainter(
          bbox: detection.bbox,
          confidence: detection.confidence,
          scaleX: scaleX,
          scaleY: scaleY,
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Positioned(
      top: 0, left: 0, right: 0,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 16, 24, 0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _Caption(top: 'Second-Life Couture', bottom: 'Atelier Scanner'),
            _Caption(
              top: 'Frame No. 001',
              bottom: kIsWeb ? 'web · yolo + onnx' : 'native · 30 mm',
              alignEnd: true,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildShutterBar() {
    final isCapturing = _state == _ScanState.captured;
    return Positioned(
      left: 0, right: 0, bottom: 0,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 16, 24, 28),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Confidence meter
            _ConfidenceMeter(confidence: _detectionConfidence),
            const SizedBox(height: 10),
            _LabelLine(
              text: _state == _ScanState.detecting
                  ? 'Tile detected · hold steady'
                  : 'Place pattern within reticle · tap to read',
            ),
            const SizedBox(height: 18),
            GestureDetector(
              onTap: isCapturing ? null : () => _capture(),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 220),
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: _state == _ScanState.detecting
                        ? const Color(0xFFB8472B)
                        : const Color(0xFFF4EDE0),
                    width: _state == _ScanState.detecting ? 2.5 : 2.0,
                  ),
                ),
                child: Center(
                  child: Container(
                    width: isCapturing ? 28 : 56,
                    height: isCapturing ? 28 : 56,
                    decoration: BoxDecoration(
                      shape: isCapturing ? BoxShape.rectangle : BoxShape.circle,
                      borderRadius: isCapturing ? BorderRadius.circular(4) : null,
                      color: const Color(0xFFB8472B),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCapturingLayer() {
    return Container(
      color: const Color(0x60000000),
      alignment: Alignment.center,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: const [
          SizedBox(
            width: 32, height: 32,
            child: CircularProgressIndicator(
              color: Color(0xFFB8472B), strokeWidth: 1.5,
            ),
          ),
          SizedBox(height: 18),
          Text(
            'Reading the pattern…',
            style: TextStyle(
              color: Color(0xFFF4EDE0),
              fontFamily: 'CormorantGaramond',
              fontStyle: FontStyle.italic,
              fontSize: 19,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Confidence meter
// ---------------------------------------------------------------------------

class _ConfidenceMeter extends StatelessWidget {
  final double confidence; // 0..1
  const _ConfidenceMeter({required this.confidence});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 2,
      child: LayoutBuilder(builder: (context, constraints) {
        return Stack(
          children: [
            Container(color: const Color(0x33F4EDE0)),
            AnimatedContainer(
              duration: const Duration(milliseconds: 80),
              width: constraints.maxWidth * confidence.clamp(0.0, 1.0),
              color: confidence >= 0.85
                  ? const Color(0xFF2D5A3D)
                  : const Color(0xFFB8472B),
            ),
          ],
        );
      }),
    );
  }
}

// ---------------------------------------------------------------------------
// Detection box overlay
// ---------------------------------------------------------------------------

class _DetectionBoxPainter extends CustomPainter {
  final Rect bbox;
  final double confidence;
  final double scaleX;
  final double scaleY;

  const _DetectionBoxPainter({
    required this.bbox,
    required this.confidence,
    required this.scaleX,
    required this.scaleY,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final scaledRect = Rect.fromLTRB(
      bbox.left * scaleX,
      bbox.top * scaleY,
      bbox.right * scaleX,
      bbox.bottom * scaleY,
    );
    final paint = Paint()
      ..color = Color.fromRGBO(
        184, 71, 43,
        (confidence * 0.9).clamp(0.0, 1.0),
      )
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;
    canvas.drawRect(scaledRect, paint);
  }

  @override
  bool shouldRepaint(covariant _DetectionBoxPainter old) =>
      old.bbox != bbox || old.confidence != confidence;
}

// ---------------------------------------------------------------------------
// Reticle painter (extended from MVP — adds confidence ring tint)
// ---------------------------------------------------------------------------

class _ReticlePainter extends CustomPainter {
  final double pulse;
  final double scanLine;
  final double confidence;
  final _ScanState state;

  const _ReticlePainter({
    required this.pulse,
    required this.scanLine,
    required this.confidence,
    required this.state,
  });

  @override
  void paint(Canvas canvas, Size size) {
    const paper  = Color(0xFFF4EDE0);
    const accent = Color(0xFFB8472B);

    final w = size.width;
    final h = size.height;
    final box = math.min(w, h) * 0.70;
    final cx = w / 2;
    final cy = h / 2;
    final left   = cx - box / 2;
    final top    = cy - box / 2;
    final right  = cx + box / 2;
    final bottom = cy + box / 2;

    // Vignette
    final maskPaint = Paint()..color = const Color(0x66000000);
    canvas.drawRect(Rect.fromLTWH(0, 0, w, top), maskPaint);
    canvas.drawRect(Rect.fromLTWH(0, bottom, w, h - bottom), maskPaint);
    canvas.drawRect(Rect.fromLTWH(0, top, left, box), maskPaint);
    canvas.drawRect(Rect.fromLTWH(right, top, w - right, box), maskPaint);

    // Corner brackets
    final bracketColor = state == _ScanState.detecting
        ? Color.lerp(paper, accent, confidence.clamp(0.0, 1.0))!
        : paper;
    final bracket = Paint()
      ..color = bracketColor
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;
    const armLen = 22.0;
    void drawBracket(Offset a, Offset b1, Offset b2) {
      canvas.drawLine(a, b1, bracket);
      canvas.drawLine(a, b2, bracket);
    }
    drawBracket(Offset(left, top),
        Offset(left + armLen, top), Offset(left, top + armLen));
    drawBracket(Offset(right, top),
        Offset(right - armLen, top), Offset(right, top + armLen));
    drawBracket(Offset(left, bottom),
        Offset(left + armLen, bottom), Offset(left, bottom - armLen));
    drawBracket(Offset(right, bottom),
        Offset(right - armLen, bottom), Offset(right, bottom - armLen));

    // Pulsing inner circle
    final circleR = box * 0.32 * (0.97 + 0.03 * math.sin(pulse * math.pi * 2));
    final circleOpacity = state == _ScanState.detecting
        ? 0.55 + 0.35 * confidence
        : 0.55;
    final circle = Paint()
      ..color = paper.withOpacity(circleOpacity)
      ..strokeWidth = state == _ScanState.detecting ? 1.2 : 0.8
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(Offset(cx, cy), circleR, circle);

    // Accent ticks
    final tick = Paint()
      ..color = accent
      ..strokeWidth = 1.2
      ..style = PaintingStyle.stroke;
    for (int k = 0; k < 8; k++) {
      final ang = k * math.pi / 4;
      final r1 = circleR + 4;
      final r2 = circleR + 10;
      canvas.drawLine(
        Offset(cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)),
        Offset(cx + r2 * math.cos(ang), cy + r2 * math.sin(ang)),
        tick,
      );
    }

    // Crosshair
    final cross = Paint()
      ..color = paper.withOpacity(0.35)
      ..strokeWidth = 0.5;
    canvas.drawLine(Offset(cx - 8, cy), Offset(cx + 8, cy), cross);
    canvas.drawLine(Offset(cx, cy - 8), Offset(cx, cy + 8), cross);

    // Scanning line
    final scanY = top + box * scanLine;
    final lineGrad = Paint()
      ..shader = LinearGradient(
        colors: [
          accent.withOpacity(0),
          accent.withOpacity(0.85),
          accent.withOpacity(0),
        ],
        stops: const [0.0, 0.5, 1.0],
      ).createShader(Rect.fromLTWH(left, scanY - 1, box, 2))
      ..strokeWidth = 1.2;
    canvas.drawLine(Offset(left + 8, scanY), Offset(right - 8, scanY), lineGrad);

    // Numeric watermark
    final tp = TextPainter(
      text: const TextSpan(
        text: 'Ø 30.0 mm',
        style: TextStyle(
          color: Color(0xCCF4EDE0),
          fontFamily: 'JetBrainsMono',
          fontSize: 9,
          letterSpacing: 1.8,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, Offset(right - tp.width - 6, bottom + 6));
  }

  @override
  bool shouldRepaint(covariant _ReticlePainter old) =>
      old.pulse != pulse || old.scanLine != scanLine ||
      old.confidence != confidence || old.state != state;
}

// ---------------------------------------------------------------------------
// Shared subwidgets
// ---------------------------------------------------------------------------

class _Caption extends StatelessWidget {
  final String top;
  final String bottom;
  final bool alignEnd;
  const _Caption({required this.top, required this.bottom, this.alignEnd = false});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          alignEnd ? CrossAxisAlignment.end : CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          top.toUpperCase(),
          style: const TextStyle(
            color: Color(0xFFF4EDE0),
            fontFamily: 'JetBrainsMono',
            fontSize: 9,
            letterSpacing: 2.4,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          bottom.toUpperCase(),
          style: const TextStyle(
            color: Color(0xAAF4EDE0),
            fontFamily: 'JetBrainsMono',
            fontSize: 9,
            letterSpacing: 2.4,
          ),
        ),
      ],
    );
  }
}

class _LabelLine extends StatelessWidget {
  final String text;
  const _LabelLine({required this.text});

  @override
  Widget build(BuildContext context) {
    return Text(
      text.toUpperCase(),
      style: const TextStyle(
        color: Color(0xCCF4EDE0),
        fontFamily: 'JetBrainsMono',
        fontSize: 9,
        letterSpacing: 2.4,
      ),
    );
  }
}
