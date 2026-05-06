// CoreML plugin for macro classifier and wear estimator.
//
// Required assets in flutter/assets/models/:
//   macro_cnn.mlpackage    (exported by scripts/export_models.py on macOS)
//   wear_estimator.mlpackage
//
// Register in AppDelegate.swift:
//   SlcMlPlugin.register(with: registrar(forPlugin: "SlcMlPlugin")!)
//
// The ONNX model exports two outputs from the shared backbone:
//   macro_logits:  [1, 16] float32
//   wear_estimate: [1, 1]  float32 (sigmoid)

import CoreML
import Flutter
import Foundation

class SlcMlPlugin: NSObject, FlutterPlugin {

    static func register(with registrar: FlutterPluginRegistrar) {
        let plugin = SlcMlPlugin()
        let messenger = registrar.messenger()

        FlutterMethodChannel(name: "slc/macro_classifier",
                             binaryMessenger: messenger)
            .setMethodCallHandler(plugin.handleMacro)

        FlutterMethodChannel(name: "slc/wear_estimator",
                             binaryMessenger: messenger)
            .setMethodCallHandler(plugin.handleWear)
    }

    private var macroModel: MLModel?
    private var wearModel:  MLModel?

    // MARK: - Channel handlers

    private func handleMacro(_ call: FlutterMethodCall,
                              result: @escaping FlutterResult) {
        guard call.method == "classify" else { result(FlutterMethodNotImplemented); return }
        guard let args = call.arguments as? [String: Any],
              let cropList = args["crop"] as? [Double],
              cropList.count == 64 * 64 else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "crop must be [Double] of length 4096", details: nil))
            return
        }
        do {
            result(try runMacro(crop: cropList.map { Float($0) }))
        } catch {
            result(FlutterError(code: "INFERENCE_ERROR",
                                message: error.localizedDescription, details: nil))
        }
    }

    private func handleWear(_ call: FlutterMethodCall,
                             result: @escaping FlutterResult) {
        guard call.method == "estimate" else { result(FlutterMethodNotImplemented); return }
        guard let args = call.arguments as? [String: Any],
              let cropList = args["crop"] as? [Double],
              cropList.count == 64 * 64 else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "crop must be [Double] of length 4096", details: nil))
            return
        }
        do {
            result(try runWear(crop: cropList.map { Float($0) }))
        } catch {
            result(FlutterError(code: "INFERENCE_ERROR",
                                message: error.localizedDescription, details: nil))
        }
    }

    // MARK: - Model loading

    private func ensureMacro() throws {
        guard macroModel == nil else { return }
        guard let url = Bundle.main.url(forResource: "macro_cnn",
                                        withExtension: "mlpackage") else {
            throw SlcMlError.modelNotFound("macro_cnn.mlpackage")
        }
        macroModel = try MLModel(contentsOf: url)
    }

    private func ensureWear() throws {
        guard wearModel == nil else { return }
        guard let url = Bundle.main.url(forResource: "wear_estimator",
                                        withExtension: "mlpackage") else {
            throw SlcMlError.modelNotFound("wear_estimator.mlpackage")
        }
        wearModel = try MLModel(contentsOf: url)
    }

    // MARK: - Inference

    // Input:  height_field [1, 1, 64, 64] float32
    // Outputs: macro_logits [1, 16], wear_estimate [1, 1]
    private func runMacro(crop: [Float]) throws -> [String: Any] {
        try ensureMacro()
        let input = try makeInput(crop: crop)
        let output = try macroModel!.prediction(from: input)

        guard let logits = output.featureValue(for: "macro_logits")?.multiArrayValue
        else { throw SlcMlError.badOutput("macro_logits") }

        var bestIdx = 0
        var bestVal = logits[0].floatValue
        for i in 1..<16 {
            let v = logits[i].floatValue
            if v > bestVal { bestVal = v; bestIdx = i }
        }
        let probs = softmax(logits: logits, count: 16)
        return ["family": bestIdx, "confidence": Double(probs[bestIdx])]
    }

    private func runWear(crop: [Float]) throws -> Double {
        try ensureWear()
        let input = try makeInput(crop: crop)
        let output = try wearModel!.prediction(from: input)

        guard let wearArr = output.featureValue(for: "wear_estimate")?.multiArrayValue
        else { throw SlcMlError.badOutput("wear_estimate") }

        return Double(wearArr[0].floatValue).clamped(to: 0.0...1.0)
    }

    // MARK: - Helpers

    private func makeInput(crop: [Float]) throws -> MLDictionaryFeatureProvider {
        let shape: [NSNumber] = [1, 1, 64, 64]
        guard let arr = try? MLMultiArray(shape: shape, dataType: .float32)
        else { throw SlcMlError.allocationFailed }
        for (i, v) in crop.enumerated() { arr[i] = NSNumber(value: v) }
        return try MLDictionaryFeatureProvider(dictionary: ["height_field": arr])
    }

    private func softmax(logits: MLMultiArray, count: Int) -> [Float] {
        let vals = (0..<count).map { logits[$0].floatValue }
        let maxV = vals.max() ?? 0
        let exps = vals.map { expf($0 - maxV) }
        let sum  = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }

    // MARK: - Error types

    enum SlcMlError: Error {
        case modelNotFound(String)
        case allocationFailed
        case badOutput(String)
    }
}

private extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
