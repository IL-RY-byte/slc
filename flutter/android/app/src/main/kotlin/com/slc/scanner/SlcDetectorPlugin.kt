package com.slc.scanner

// YOLOv8-nano-seg tile detector + ARCore depth via platform channels.
//
// TFLite dependency (add to android/app/build.gradle):
//   implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.4'
//   implementation 'com.google.ar:core:1.41.0'
//
// Required assets:
//   assets/models/tile_detector.tflite   (YOLOv8-nano-seg INT8)
//
// Channels:
//   slc/tile_detector  — method 'detect'
//   slc/depth_estimator — method 'estimate'

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class SlcDetectorPlugin(private val context: Context) {

    companion object {
        const val DETECT_CHANNEL = "slc/tile_detector"
        const val DEPTH_CHANNEL  = "slc/depth_estimator"
        private const val YOLO_INPUT_SIZE = 640
    }

    private var yoloInterpreter: Interpreter? = null

    fun register(flutterEngine: FlutterEngine) {
        val messenger = flutterEngine.dartExecutor.binaryMessenger

        MethodChannel(messenger, DETECT_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "detect" -> {
                    val yuv  = call.argument<ByteArray>("yuv") ?: run {
                        result.error("BAD_ARGS", "yuv required", null); return@setMethodCallHandler
                    }
                    val w = call.argument<Int>("width") ?: run {
                        result.error("BAD_ARGS", "width required", null); return@setMethodCallHandler
                    }
                    val h = call.argument<Int>("height") ?: run {
                        result.error("BAD_ARGS", "height required", null); return@setMethodCallHandler
                    }
                    try {
                        result.success(runYolo(yuv, w, h))
                    } catch (e: Exception) {
                        result.error("INFERENCE_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }

        MethodChannel(messenger, DEPTH_CHANNEL).setMethodCallHandler { call, result ->
            // ARCore depth is obtained from the ARSession managed by the AR plugin.
            // This channel returns a stub until the AR session integration is complete.
            result.error("NOT_IMPLEMENTED",
                "ARCore depth plugin not yet integrated. " +
                "Use MiDaS TFLite fallback via depth_estimator channel.", null)
        }
    }

    private fun ensureYolo() {
        if (yoloInterpreter == null) {
            val afd = context.assets.openFd("models/tile_detector.tflite")
            FileInputStream(afd.fileDescriptor).use { fis ->
                val buf = fis.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    afd.startOffset, afd.declaredLength
                )
                val opts = Interpreter.Options().apply { numThreads = 2 }
                yoloInterpreter = Interpreter(buf, opts)
            }
        }
    }

    // YUV_420_888 -> JPEG -> Bitmap -> resize to 640x640 -> run YOLO
    // Output map: cx, cy, w, h (pixels), confidence, diameter_px
    private fun runYolo(yuvBytes: ByteArray, imgW: Int, imgH: Int): Map<String, Any>? {
        ensureYolo()

        // Decode YUV NV21 to JPEG to Bitmap
        val yuv = YuvImage(yuvBytes, ImageFormat.NV21, imgW, imgH, null)
        val jpegOut = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, imgW, imgH), 90, jpegOut)
        val bitmap = BitmapFactory.decodeByteArray(jpegOut.toByteArray(), 0, jpegOut.size())
        val scaled  = Bitmap.createScaledBitmap(bitmap, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, true)

        // Prepare float input [1, 640, 640, 3] — NHWC, normalised 0..1
        val inputBuf = ByteBuffer.allocateDirect(4 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(YOLO_INPUT_SIZE * YOLO_INPUT_SIZE)
        scaled.getPixels(pixels, 0, YOLO_INPUT_SIZE, 0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
        for (px in pixels) {
            inputBuf.putFloat(((px shr 16) and 0xFF) / 255f)
            inputBuf.putFloat(((px shr 8)  and 0xFF) / 255f)
            inputBuf.putFloat((px          and 0xFF) / 255f)
        }
        inputBuf.rewind()

        // YOLOv8-seg outputs: [1, 116, 8400] detection + [1, 32, 160, 160] proto masks
        val detOut  = Array(1) { Array(116) { FloatArray(8400) } }
        val maskOut = Array(1) { Array(32)  { Array(160) { FloatArray(160) } } }
        val outputs = mapOf(0 to detOut, 1 to maskOut)
        yoloInterpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuf), outputs)

        // Find highest-confidence detection (cx,cy,w,h + conf; skip class logits)
        val dets = detOut[0]
        var bestConf = 0.4f   // confidence threshold
        var bestIdx  = -1
        for (i in 0 until 8400) {
            val conf = dets[4][i]
            if (conf > bestConf) { bestConf = conf; bestIdx = i }
        }
        if (bestIdx < 0) return null

        // Scale coordinates back to original image size
        val scaleX = imgW.toDouble() / YOLO_INPUT_SIZE
        val scaleY = imgH.toDouble() / YOLO_INPUT_SIZE
        val cx  = dets[0][bestIdx] * scaleX
        val cy  = dets[1][bestIdx] * scaleY
        val bw  = dets[2][bestIdx] * scaleX
        val bh  = dets[3][bestIdx] * scaleY
        val dia = (bw + bh) / 2.0

        return mapOf(
            "cx"          to cx,
            "cy"          to cy,
            "w"           to bw,
            "h"           to bh,
            "confidence"  to bestConf.toDouble(),
            "diameter_px" to dia,
        )
    }

    fun dispose() {
        yoloInterpreter?.close()
        yoloInterpreter = null
    }
}
