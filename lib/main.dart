
import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:hand_landmarker/hand_landmarker.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    final cameras = await availableCameras();
    runApp(RehaTrackApp(cameras: cameras));
  } catch (e) {
    runApp(RehaTrackApp(cameras: const [], startupError: '$e'));
  }
}

class RehaTrackApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  final String? startupError;

  const RehaTrackApp({
    super.key,
    required this.cameras,
    this.startupError,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: PoseScreen(
        cameras: cameras,
        startupError: startupError,
      ),
    );
  }
}

class PoseScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  final String? startupError;

  const PoseScreen({
    super.key,
    required this.cameras,
    this.startupError,
  });

  @override
  State<PoseScreen> createState() => _PoseScreenState();
}

class _PoseScreenState extends State<PoseScreen> {
  CameraController? cameraController;
  late final PoseDetector poseDetector;
  HandLandmarkerPlugin? handTracker;

  StreamSubscription<List<Hand>>? handSubscription;

  bool isCameraReady = false;
  bool isProcessingPose = false;
  bool isDisposed = false;
  bool isStreaming = false;

  String? cameraError;
  String message = 'Show your arm and hand to the camera';

  PoseLandmark? shoulder;
  PoseLandmark? elbow;
  PoseLandmark? wrist;

  double? elbowAngle;
  double? wristX;
  double? wristY;
  double? wristMovement;

  double? previousWristX;
  double? previousWristY;

  List<Hand> detectedHands = [];

  static const double confidenceThreshold = 0.60;

  @override
  void initState() {
    super.initState();

    poseDetector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.stream,
        model: PoseDetectionModel.accurate,
      ),
    );

    setupCamera();
  }

  Future<void> setupCamera() async {
    if (widget.cameras.isEmpty) {
      if (mounted) {
        setState(() {
          cameraError = widget.startupError ?? 'No camera found.';
        });
      }
      return;
    }

    final camera = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );

    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    cameraController = controller;

    try {
      // Initialize hand tracking for up to two hands.
      handTracker = HandLandmarkerPlugin.create(
        numHands: 2,
        minHandDetectionConfidence: 0.7,
        delegate: HandLandmarkerDelegate.cpu,
      );

      // Receive hand landmarks asynchronously.
      handSubscription = handTracker!.landmarkStream.listen(
        (hands) {
          if (!mounted || isDisposed) return;

          setState(() {
            detectedHands = hands;
          });
        },
        onError: (Object error) {
          debugPrint('Hand tracking stream error: $error');
        },
      );

      await controller.initialize();

      if (!mounted || isDisposed) return;

      setState(() {
        isCameraReady = true;
        message = 'Camera ready. Show your arm and hand.';
      });

      await controller.startImageStream(processCameraImage);
      isStreaming = true;
    } catch (e) {
      if (mounted && !isDisposed) {
        setState(() {
          cameraError = 'Camera initialization failed: $e';
        });
      }
    }
  }

  // Convert the camera's three YUV420 planes into NV21.
  // The pose detector receives this converted image.
  Uint8List yuv420ToNv21(CameraImage image) {
    if (image.planes.length != 3) {
      throw StateError(
        'Expected three YUV420 planes, got ${image.planes.length}.',
      );
    }

    final width = image.width;
    final height = image.height;

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final uvWidth = (width + 1) ~/ 2;
    final uvHeight = (height + 1) ~/ 2;

    final output = Uint8List(
      width * height + uvWidth * uvHeight * 2,
    );

    var index = 0;

    // Copy the Y (luminance) plane.
    for (var row = 0; row < height; row++) {
      final rowStart = row * yPlane.bytesPerRow;
      final pixelStride = yPlane.bytesPerPixel ?? 1;

      for (var col = 0; col < width; col++) {
        output[index++] =
            yPlane.bytes[rowStart + col * pixelStride];
      }
    }

    // NV21 stores interleaved V, U chroma samples.
    final uPixelStride = uPlane.bytesPerPixel ?? 1;
    final vPixelStride = vPlane.bytesPerPixel ?? 1;

    for (var row = 0; row < uvHeight; row++) {
      for (var col = 0; col < uvWidth; col++) {
        final uIndex =
            row * uPlane.bytesPerRow + col * uPixelStride;
        final vIndex =
            row * vPlane.bytesPerRow + col * vPixelStride;

        output[index++] = vPlane.bytes[vIndex];
        output[index++] = uPlane.bytes[uIndex];
      }
    }

    return output;
  }

  Future<void> processCameraImage(CameraImage image) async {
    if (isDisposed) return;

    final controller = cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    // Hand detection processes the original camera frame.
    // It runs asynchronously inside the plugin.
    try {
      handTracker?.processFrame(
        image,
        controller.description.sensorOrientation,
      );
    } catch (e) {
      debugPrint('Hand frame error: $e');
    }

    // Prevent overlapping pose-detection calls.
    if (isProcessingPose) return;
    isProcessingPose = true;

    try {
      final rotation = InputImageRotationValue.fromRawValue(
        controller.description.sensorOrientation,
      );

      if (rotation == null || image.planes.length != 3) {
        if (mounted && !isDisposed) {
          setState(() {
            message = 'Unsupported camera image format for arm tracking.';
          });
        }
        return;
      }

      final nv21Bytes = yuv420ToNv21(image);

      final inputImage = InputImage.fromBytes(
        bytes: nv21Bytes,
        metadata: InputImageMetadata(
          size: Size(
            image.width.toDouble(),
            image.height.toDouble(),
          ),
          rotation: rotation,
          format: InputImageFormat.nv21,
          bytesPerRow: image.width,
        ),
      );

      final poses = await poseDetector.processImage(inputImage);

      if (!mounted || isDisposed) return;

      PoseLandmark? bestShoulder;
      PoseLandmark? bestElbow;
      PoseLandmark? bestWrist;

      if (poses.isNotEmpty) {
        final pose = poses.first;

        final ls = pose.landmarks[PoseLandmarkType.leftShoulder];
        final le = pose.landmarks[PoseLandmarkType.leftElbow];
        final lw = pose.landmarks[PoseLandmarkType.leftWrist];

        final rs = pose.landmarks[PoseLandmarkType.rightShoulder];
        final re = pose.landmarks[PoseLandmarkType.rightElbow];
        final rw = pose.landmarks[PoseLandmarkType.rightWrist];

        double armScore(
          PoseLandmark? s,
          PoseLandmark? e,
          PoseLandmark? w,
        ) {
          if (s == null || e == null || w == null) return -1;

          if (s.likelihood < confidenceThreshold ||
              e.likelihood < confidenceThreshold ||
              w.likelihood < confidenceThreshold) {
            return -1;
          }

          return s.likelihood + e.likelihood + w.likelihood;
        }

        if (armScore(ls, le, lw) >= armScore(rs, re, rw)) {
          bestShoulder = ls;
          bestElbow = le;
          bestWrist = lw;
        } else {
          bestShoulder = rs;
          bestElbow = re;
          bestWrist = rw;
        }
      }

      if (bestShoulder != null &&
          bestElbow != null &&
          bestWrist != null) {
        final s = bestShoulder;
        final e = bestElbow;
        final w = bestWrist;

        final dx = w.x - e.x;
        final dy = w.y - e.y;

        final movement =
            previousWristX == null || previousWristY == null
                ? 0.0
                : math.sqrt(
                    math.pow(w.x - previousWristX!, 2) +
                        math.pow(w.y - previousWristY!, 2),
                  );

        previousWristX = w.x;
        previousWristY = w.y;

        setState(() {
          shoulder = s;
          elbow = e;
          wrist = w;

          elbowAngle = calculateAngle(s, e, w);

          wristX = dx;
          wristY = dy;
          wristMovement = movement;

          message = 'Arm detected';
        });
      } else {
        setState(() {
          shoulder = null;
          elbow = null;
          wrist = null;

          elbowAngle = null;
          wristX = null;
          wristY = null;
          wristMovement = null;

          previousWristX = null;
          previousWristY = null;

          message = 'Keep your shoulder, elbow and wrist visible';
        });
      }
    } catch (e) {
      debugPrint('Pose detection error: $e');

      if (mounted && !isDisposed) {
        setState(() {
          message = 'Arm detection error. Check the debug output.';
        });
      }
    } finally {
      isProcessingPose = false;
    }
  }

  double calculateAngle(
    PoseLandmark s,
    PoseLandmark e,
    PoseLandmark w,
  ) {
    final a = math.atan2(s.y - e.y, s.x - e.x);
    final b = math.atan2(w.y - e.y, w.x - e.x);

    final angle = (a - b).abs() * 180 / math.pi;

    return angle > 180 ? 360 - angle : angle;
  }

  @override
  void dispose() {
    isDisposed = true;

    unawaited(
      handSubscription?.cancel() ?? Future<void>.value(),
    );

    handTracker?.dispose();

    final controller = cameraController;

    if (controller != null) {
      if (isStreaming) {
        unawaited(controller.stopImageStream());
      }

      unawaited(controller.dispose());
    }

    unawaited(poseDetector.close());

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = cameraController;

    if (cameraError != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('RehaTrack')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Text(
              cameraError!,
              textAlign: TextAlign.center,
            ),
          ),
        ),
      );
    }

    if (!isCameraReady ||
        controller == null ||
        controller.value.previewSize == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('RehaTrack')),
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              Text(message),
            ],
          ),
        ),
      );
    }

    final previewSize = controller.value.previewSize!;

    return Scaffold(
      appBar: AppBar(
        title: const Text('RehaTrack'),
        backgroundColor: Colors.teal,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: AspectRatio(
                aspectRatio: previewSize.height / previewSize.width,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(controller),

                    // Existing shoulder, elbow and wrist overlay.
                    CustomPaint(
                      painter: ArmPainter(
                        shoulder: shoulder,
                        elbow: elbow,
                        wrist: wrist,
                        imageSize: Size(
                          previewSize.height,
                          previewSize.width,
                        ),
                        mirror:
                            controller.description.lensDirection ==
                                CameraLensDirection.front,
                      ),
                    ),

                    // New overlay: all detected hand landmarks.
                    CustomPaint(
                      painter: HandLandmarkPainter(
                        hands: detectedHands,
                        previewSize: previewSize,
                        lensDirection:
                            controller.description.lensDirection,
                        sensorOrientation:
                            controller.description.sensorOrientation,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
            child: Column(
              children: [
                Text(
                  message,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),

                Text(
                  elbowAngle == null
                      ? 'Elbow angle: --'
                      : 'Elbow angle: '
                          '${elbowAngle!.toStringAsFixed(1)}°',
                  style: const TextStyle(fontSize: 20),
                ),

                const SizedBox(height: 4),

                Text(
                  wristX == null
                      ? 'Wrist position: --'
                      : 'Wrist relative to elbow: '
                          'X ${wristX!.toStringAsFixed(0)}, '
                          'Y ${wristY!.toStringAsFixed(0)} px',
                  textAlign: TextAlign.center,
                ),

                const SizedBox(height: 4),

                Text(
                  wristMovement == null
                      ? 'Wrist movement: --'
                      : 'Movement since last frame: '
                          '${wristMovement!.toStringAsFixed(1)} px',
                  textAlign: TextAlign.center,
                ),

                const SizedBox(height: 6),

                Text(
                  'Hands detected: ${detectedHands.length} '
                  '• Landmarks per hand: '
                  '${detectedHands.isEmpty ? 0 : detectedHands.first.landmarks.length}',
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Draw the existing arm skeleton.
class ArmPainter extends CustomPainter {
  final PoseLandmark? shoulder;
  final PoseLandmark? elbow;
  final PoseLandmark? wrist;
  final Size imageSize;
  final bool mirror;

  ArmPainter({
    required this.shoulder,
    required this.elbow,
    required this.wrist,
    required this.imageSize,
    required this.mirror,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (shoulder == null || elbow == null || wrist == null) return;

    final linePaint = Paint()
      ..color = Colors.greenAccent
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;

    final jointPaint = Paint()
      ..color = Colors.orangeAccent
      ..style = PaintingStyle.fill;

    Offset mapPoint(PoseLandmark landmark) {
      var x = landmark.x / imageSize.width * size.width;
      final y = landmark.y / imageSize.height * size.height;

      if (mirror) x = size.width - x;

      return Offset(x, y);
    }

    final s = mapPoint(shoulder!);
    final e = mapPoint(elbow!);
    final w = mapPoint(wrist!);

    canvas.drawLine(s, e, linePaint);
    canvas.drawLine(e, w, linePaint);

    for (final point in [s, e, w]) {
      canvas.drawCircle(point, 7, jointPaint);
    }
  }

  @override
  bool shouldRepaint(covariant ArmPainter oldDelegate) {
    return true;
  }
}

// Draw every landmark and connection for every detected hand.
class HandLandmarkPainter extends CustomPainter {
  final List<Hand> hands;
  final Size previewSize;
  final CameraLensDirection lensDirection;
  final int sensorOrientation;

  HandLandmarkPainter({
    required this.hands,
    required this.previewSize,
    required this.lensDirection,
    required this.sensorOrientation,
  });

  static const List<List<int>> connections = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],

    // Index finger
    [0, 5], [5, 6], [6, 7], [7, 8],

    // Middle finger
    [5, 9], [9, 10], [10, 11], [11, 12],

    // Ring finger
    [9, 13], [13, 14], [14, 15], [15, 16],

    // Little finger and palm
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty || size.isEmpty) return;

    final scale = size.width / previewSize.height;

    final jointPaint = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.fill;

    final linePaint = Paint()
      ..color = Colors.lightBlueAccent
      ..strokeWidth = 3 / scale
      ..strokeCap = StrokeCap.round;

    canvas.save();

    final center = Offset(size.width / 2, size.height / 2);

    canvas.translate(center.dx, center.dy);
    canvas.rotate(sensorOrientation * math.pi / 180);

    if (lensDirection == CameraLensDirection.front) {
      canvas.scale(-1, 1);
      canvas.rotate(math.pi);
    }

    canvas.scale(scale);

    final logicalWidth = previewSize.width;
    final logicalHeight = previewSize.height;

    for (final hand in hands) {
      if (hand.landmarks.length < 21) continue;

      Offset pointFor(int index) {
        final landmark = hand.landmarks[index];

        return Offset(
          (landmark.x - 0.5) * logicalWidth,
          (landmark.y - 0.5) * logicalHeight,
        );
      }

      // Draw the 21 points.
      for (var i = 0; i < 21; i++) {
        canvas.drawCircle(
          pointFor(i),
          5 / scale,
          jointPaint,
        );
      }

      // Draw the finger and palm connections.
      for (final connection in connections) {
        canvas.drawLine(
          pointFor(connection[0]),
          pointFor(connection[1]),
          linePaint,
        );
      }
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant HandLandmarkPainter oldDelegate) {
    return true;
  }
}
