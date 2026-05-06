package com.slc.scanner

// TFLite dependency — add to android/app/build.gradle:
//   implementation 'org.tensorflow:tensorflow-lite:2.14.0'
//   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
//
// Required Flutter assets in pubspec.yaml:
//   assets/models/macro_cnn.tflite
//   assets/models/wear_estimator.tflite

import android.content.Context
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class SlcMlPlugin(private val context: Context) {

    companion object {
        const val MACRO_CHANNEL = "slc/macro_classifier"
        const val WEAR_CHANNEL  = "slc/wear_estimator"
        private const val INPUT_SIZE = 64 * 64  // 64x64 float32 crop
        private const val N_CLASSES  = 16
    }

    private var macroInterpreter: Interpreter? = null
    private var wearInterpreter:  Interpreter? = null

    fun register(flutterEngine: FlutterEngine) {
        val messenger = flutterEngine.dartExecutor.binaryMessenger

        MethodChannel(messenger, MACRO_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "classify" -> {
                    val crop = call.argument<List<Double>>("crop")
                    if (crop == null || crop.size != INPUT_SIZE) {
                        result.error("BAD_ARGS",
                            "crop must be a list of ${INPUT_SIZE} floats", null)
                        return@setMethodCallHandler
                    }
                    try {
                        result.success(runMacro(crop))
                    } catch (e: Exception) {
                        result.error("INFERENCE_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }

        MethodChannel(messenger, WEAR_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "estimate" -> {
                    val crop = call.argument<List<Double>>("crop")
                    if (crop == null || crop.size != INPUT_SIZE) {
                        result.error("BAD_ARGS",
                            "crop must be a list of ${INPUT_SIZE} floats", null)
                        return@setMethodCallHandler
                    }
                    try {
                        result.success(runWear(crop))
                    } catch (e: Exception) {
                        result.error("INFERENCE_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun loadModel(assetName: String): Interpreter {
        val afd = context.assets.openFd(assetName)
        FileInputStream(afd.fileDescriptor).use { fis ->
            val buf = fis.channel.map(
                FileChannel.MapMode.READ_ONLY,
                afd.startOffset,
                afd.declaredLength
            )
            return Interpreter(buf)
        }
    }

    private fun ensureMacro() {
        if (macroInterpreter == null)
            macroInterpreter = loadModel("models/macro_cnn.tflite")
    }

    private fun ensureWear() {
        if (wearInterpreter == null)
            wearInterpreter = loadModel("models/wear_estimator.tflite")
    }

    // Input:  [1, 1, 64, 64] float32 (NCHW — matches PyTorch export)
    // Output: [1, 16] macro logits  +  [1, 1] wear sigmoid
    // The shared-backbone ONNX exports two outputs; TFLite conversion preserves them.
    private fun runMacro(crop: List<Double>): Map<String, Any> {
        ensureMacro()
        val input = floatBuffer(crop)
        val logits = Array(1) { FloatArray(N_CLASSES) }
        val wear   = Array(1) { FloatArray(1) }
        val outputs = mapOf(0 to logits, 1 to wear)
        macroInterpreter!!.runForMultipleInputsOutputs(arrayOf(input), outputs)

        var best = 0
        for (i in 1 until N_CLASSES) if (logits[0][i] > logits[0][best]) best = i
        val probs = softmax(logits[0])
        return mapOf("family" to best, "confidence" to probs[best].toDouble())
    }

    private fun runWear(crop: List<Double>): Double {
        ensureWear()
        val input = floatBuffer(crop)
        val wear  = Array(1) { FloatArray(1) }
        val outputs = mapOf(1 to wear)
        wearInterpreter!!.runForMultipleInputsOutputs(arrayOf(input), outputs)
        return wear[0][0].toDouble().coerceIn(0.0, 1.0)
    }

    private fun floatBuffer(values: List<Double>): ByteBuffer =
        ByteBuffer.allocateDirect(4 * values.size).apply {
            order(ByteOrder.nativeOrder())
            values.forEach { putFloat(it.toFloat()) }
            rewind()
        }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exp = logits.map { Math.exp((it - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }

    fun dispose() {
        macroInterpreter?.close(); macroInterpreter = null
        wearInterpreter?.close();  wearInterpreter  = null
    }
}
