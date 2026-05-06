package com.slc.scanner

// MiDaS-small monocular depth estimator — non-LiDAR fallback (~120ms on Pixel 6).
//
// Required asset: assets/models/midas_small.tflite
//   Input:  [1, 3, 256, 256] float32 NCHW, ImageNet-normalised
//   Output: [1, 1, 256, 256] float32 inverse relative depth (larger = closer)
//
// Channel: slc/midas_depth -> method 'estimate'
//
// TFLite dependency (add to android/app/build.gradle):
//   implementation 'org.tensorflow:tensorflow-lite:2.14.0'

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
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

class SlcMidasPlugin(private val context: Context) {

    companion object {
        const val CHANNEL      = "slc/midas_depth"
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private var interp: Interpreter? = null

    fun register(flutterEngine: FlutterEngine) {
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "estimate" -> {
                        val yuv    = call.argument<ByteArray>("yuv")
                        val width  = call.argument<Int>("width")
                        val height = call.argument<Int>("height")
                        val cx     = (call.argument<Any>("cx") as? Number)?.toDouble()
                        val cy     = (call.argument<Any>("cy") as? Number)?.toDouble()
                        val bw     = (call.argument<Any>("w")  as? Number)?.toDouble()
                        val bh     = (call.argument<Any>("h")  as? Number)?.toDouble()
                        val diaPx  = (call.argument<Any>("diameter_px") as? Number)?.toDouble()

                        if (yuv == null || width == null || height == null ||
                            cx == null || cy == null || bw == null || bh == null || diaPx == null) {
                            result.error("BAD_ARGS", "missing required args", null)
                            return@setMethodCallHandler
                        }
                        try {
                            result.success(estimate(yuv, width, height, cx, cy, bw, bh, diaPx))
                        } catch (e: Exception) {
                            result.error("MIDAS_ERROR", e.message, null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun ensureModel() {
        if (interp != null) return
        val afd = context.assets.openFd("models/midas_small.tflite")
        FileInputStream(afd.fileDescriptor).use { fis ->
            val buf = fis.channel.map(FileChannel.MapMode.READ_ONLY,
                                      afd.startOffset, afd.declaredLength)
            val opts = Interpreter.Options().apply { numThreads = 2 }
            interp = Interpreter(buf, opts)
        }
    }

    private fun estimate(
        yuvBytes: ByteArray, imgW: Int, imgH: Int,
        cx: Double, cy: Double, bw: Double, bh: Double, diaPx: Double
    ): ByteArray {
        ensureModel()

        // YUV NV21 -> Bitmap -> crop -> resize to 256x256
        val yuv = YuvImage(yuvBytes, ImageFormat.NV21, imgW, imgH, null)
        val jpegOut = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, imgW, imgH), 90, jpegOut)
        val fullBmp = BitmapFactory.decodeByteArray(jpegOut.toByteArray(), 0, jpegOut.size())

        val margin = 0.10
        val x0 = ((cx - bw / 2 * (1 + margin)).toInt()).coerceIn(0, imgW - 1)
        val y0 = ((cy - bh / 2 * (1 + margin)).toInt()).coerceIn(0, imgH - 1)
        val x1 = ((cx + bw / 2 * (1 + margin)).toInt()).coerceIn(x0 + 1, imgW)
        val y1 = ((cy + bh / 2 * (1 + margin)).toInt()).coerceIn(y0 + 1, imgH)
        val cropped = Bitmap.createBitmap(fullBmp, x0, y0, x1 - x0, y1 - y0)
        val scaled  = Bitmap.createScaledBitmap(cropped, 256, 256, true)

        // ImageNet-normalise -> NCHW float buffer [1, 3, 256, 256]
        val pixels = IntArray(256 * 256)
        scaled.getPixels(pixels, 0, 256, 0, 0, 256, 256)
        val inputBuf = ByteBuffer.allocateDirect(4 * 3 * 256 * 256)
            .apply { order(ByteOrder.nativeOrder()) }
        for (c in 0..2) {
            for (px in pixels) {
                val chan = when (c) {
                    0 -> ((px shr 16) and 0xFF)
                    1 -> ((px shr 8)  and 0xFF)
                    else -> (px       and 0xFF)
                }
                inputBuf.putFloat((chan / 255f - MEAN[c]) / STD[c])
            }
        }
        inputBuf.rewind()

        // Run MiDaS: output [1, 1, 256, 256]
        val output = Array(1) { Array(1) { Array(256) { FloatArray(256) } } }
        interp!!.run(inputBuf, output)

        // MiDaS output: larger = closer to camera = taller protrusion.
        // Normalize so relative ordering matches training data (0=flat, 1=tallest).
        val flat = FloatArray(256 * 256) { r -> output[0][0][r / 256][r % 256] }
        val minD = flat.min()
        val maxD = flat.max()
        val range = maxOf(maxD - minD, 1e-6f)
        val relDepth = FloatArray(256 * 256) { i -> (flat[i] - minD) / range }

        // Calibrate + bilinear downsample 256x256 -> 128x128
        val mmPerPx = 30.0 / diaPx
        val field = FloatArray(128 * 128) { idx ->
            val r = idx / 128
            val c = idx % 128
            val sr = r * 256 / 128
            val sc = c * 256 / 128
            (relDepth[sr * 256 + sc] * mmPerPx * 10).toFloat()
        }

        // Pack as raw bytes for Uint8List on Dart side
        val outBuf = ByteBuffer.allocate(128 * 128 * 4).apply {
            order(ByteOrder.LITTLE_ENDIAN)
            field.forEach { putFloat(it) }
        }
        return outBuf.array()
    }

    fun dispose() {
        interp?.close()
        interp = null
    }
}
