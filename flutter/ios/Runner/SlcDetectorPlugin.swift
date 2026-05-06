// YOLOv8-nano-seg tile detector + ARKit LiDAR depth via platform channels.
//
// Required assets:
//   assets/models/tile_detector.mlpackage   (exported from YOLOv8 via coremltools)
//
// Channels:
//   slc/tile_detector   — method 'detect'  (YUV bytes + width/height -> bbox map)
//   slc/depth_estimator — method 'estimate' (YUV + bbox -> 128x128 Float32 depth mm)
//
// Register in AppDelegate.swift:
//   SlcDetectorPlugin.register(with: registrar(forPlugin: "SlcDetectorPlugin")!)

import ARKit
import CoreML
import Flutter
import UIKit
import VideoToolbox

class SlcDetectorPlugin: NSObject, FlutterPlugin {

    static func register(with registrar: FlutterPluginRegistrar) {
        let plugin = SlcDetectorPlugin()
        let messenger = registrar.messenger()
        FlutterMethodChannel(name: "slc/tile_detector",
                             binaryMessenger: messenger)
            .setMethodCallHandler(plugin.handleDetect)
        FlutterMethodChannel(name: "slc/depth_estimator",
                             binaryMessenger: messenger)
            .setMethodCallHandler(plugin.handleDepth)
    }

    private var yoloModel: MLModel?

    // MARK: - Tile detector

    private func handleDetect(_ call: FlutterMethodCall,
                               result: @escaping FlutterResult) {
        guard call.method == "detect" else { result(FlutterMethodNotImplemented); return }
        guard let args = call.arguments as? [String: Any],
              let yuvData = args["yuv"]   as? FlutterStandardTypedData,
              let width   = args["width"] as? Int,
              let height  = args["height"] as? Int else {
            result(FlutterError(code: "BAD_ARGS", message: "yuv/width/height required", details: nil))
            return
        }
        do {
            let det = try runYolo(yuvBytes: yuvData.data, width: width, height: height)
            result(det)  // nil -> no detection, map -> detection
        } catch {
            result(FlutterError(code: "INFERENCE_ERROR",
                                message: error.localizedDescription, details: nil))
        }
    }

    private func ensureYolo() throws {
        guard yoloModel == nil else { return }
        guard let url = Bundle.main.url(forResource: "tile_detector",
                                        withExtension: "mlpackage") else {
            throw SlcDetectorError.modelNotFound("tile_detector.mlpackage")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        yoloModel = try MLModel(contentsOf: url, configuration: cfg)
    }

    // YUV NV12 -> CGImage -> resize 640x640 -> CoreML -> best detection
    private func runYolo(yuvBytes: Data, width: Int, height: Int) throws -> [String: Any]? {
        try ensureYolo()

        // Decode YUV NV12 to UIImage (camera frame)
        guard let image = yuvToUIImage(yuvBytes: yuvBytes, width: width, height: height),
              let resized = image.resize(to: CGSize(width: 640, height: 640)),
              let pixelBuffer = resized.toPixelBuffer() else {
            throw SlcDetectorError.decodeFailed
        }

        let input  = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
        let output = try yoloModel!.prediction(from: input)

        // YOLOv8 CoreML output: 'var_xxx' shape [1, 116, 8400] — cx,cy,w,h,conf,class
        guard let det = output.featureValue(for: "var_1231")?.multiArrayValue
              ?? output.featureValue(for: "output0")?.multiArrayValue else {
            return nil
        }

        let n = 8400
        var bestConf: Float = 0.4
        var bestI = -1
        for i in 0..<n {
            let conf = det[[0, 4, i] as [NSNumber]].floatValue
            if conf > bestConf { bestConf = conf; bestI = i }
        }
        guard bestI >= 0 else { return nil }

        let scaleX = Double(width)  / 640
        let scaleY = Double(height) / 640
        let cx  = det[[0, 0, bestI] as [NSNumber]].doubleValue * scaleX
        let cy  = det[[0, 1, bestI] as [NSNumber]].doubleValue * scaleY
        let bw  = det[[0, 2, bestI] as [NSNumber]].doubleValue * scaleX
        let bh  = det[[0, 3, bestI] as [NSNumber]].doubleValue * scaleY

        return [
            "cx"          : cx,
            "cy"          : cy,
            "w"           : bw,
            "h"           : bh,
            "confidence"  : Double(bestConf),
            "diameter_px" : (bw + bh) / 2,
        ]
    }

    // MARK: - Depth estimator

    // ARSession is owned by the Flutter camera plugin's AR view.
    // We receive the latest ARFrame via a shared accessor set by the host app.
    static var latestARFrame: ARFrame?

    private func handleDepth(_ call: FlutterMethodCall,
                              result: @escaping FlutterResult) {
        guard call.method == "estimate" else { result(FlutterMethodNotImplemented); return }
        guard let args   = call.arguments as? [String: Any],
              let cx     = (args["cx"]   as? NSNumber)?.doubleValue,
              let cy     = (args["cy"]   as? NSNumber)?.doubleValue,
              let bw     = (args["w"]    as? NSNumber)?.doubleValue,
              let bh     = (args["h"]    as? NSNumber)?.doubleValue,
              let diaPx  = (args["diameter_px"] as? NSNumber)?.doubleValue else {
            result(FlutterError(code: "BAD_ARGS", message: "bbox args required", details: nil))
            return
        }

        guard let frame = SlcDetectorPlugin.latestARFrame,
              let depthMap = frame.sceneDepth?.depthMap else {
            // LiDAR not available or no AR frame — caller should fall back to MiDaS
            result(FlutterError(code: "NO_DEPTH",
                                message: "ARKit LiDAR depth unavailable", details: nil))
            return
        }

        do {
            let heightField = try cropAndCalibrate(
                depthMap: depthMap, cx: cx, cy: cy, bw: bw, bh: bh, diaPx: diaPx)
            result(FlutterStandardTypedData(float32: heightField))
        } catch {
            result(FlutterError(code: "DEPTH_ERROR",
                                message: error.localizedDescription, details: nil))
        }
    }

    // Crop ARKit depth map to tile bbox, resample to 128x128, calibrate to mm.
    private func cropAndCalibrate(depthMap: CVPixelBuffer,
                                  cx: Double, cy: Double,
                                  bw: Double, bh: Double,
                                  diaPx: Double) throws -> Data {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let dW = CVPixelBufferGetWidth(depthMap)
        let dH = CVPixelBufferGetHeight(depthMap)
        guard let base = CVPixelBufferGetBaseAddress(depthMap) else {
            throw SlcDetectorError.decodeFailed
        }
        let depthFloats = base.bindMemory(to: Float32.self,
                                          capacity: dW * dH)

        // Scale bbox from image coords to depth map coords
        let sx = Double(dW) / Double(UIScreen.main.bounds.width * UIScreen.main.scale)
        let sy = Double(dH) / Double(UIScreen.main.bounds.height * UIScreen.main.scale)
        let x0 = max(0, Int((cx - bw / 2) * sx))
        let y0 = max(0, Int((cy - bh / 2) * sy))
        let x1 = min(dW, Int((cx + bw / 2) * sx))
        let y1 = min(dH, Int((cy + bh / 2) * sy))

        let cropW = max(1, x1 - x0)
        let cropH = max(1, y1 - y0)
        // First pass: collect all 128x128 samples and find tile surface depth (median).
        // ARKit depth in metres; smaller = closer to camera = taller protrusion.
        let out = 128 * 128
        var rawMm = [Double](repeating: 0, count: out)
        for r in 0..<128 {
            for c in 0..<128 {
                let srcX = x0 + c * cropW / 128
                let srcY = y0 + r * cropH / 128
                rawMm[r * 128 + c] = Double(depthFloats[srcY * dW + srcX]) * 1000.0
            }
        }
        let sorted = rawMm.sorted()
        let surfaceDepthMm = sorted[out / 2]   // median = tile flat surface reference

        // Second pass: protrusion height = surface_depth - pixel_depth (positive = protruding).
        // Scale to mm using known 30mm tile diameter for the z-axis calibration.
        let mmPerPx = 30.0 / diaPx
        var field = [Float32](repeating: 0, count: out)
        for i in 0..<out {
            let protrusion = max(0.0, surfaceDepthMm - rawMm[i]) * mmPerPx
            field[i] = Float32(protrusion)
        }

        return Data(bytes: field, count: out * 4)
    }

    // MARK: - YUV -> UIImage helpers

    private func yuvToUIImage(yuvBytes: Data, width: Int, height: Int) -> UIImage? {
        // Assume NV12 / YUV_420_888 packed: Y plane then interleaved UV plane
        var image: UIImage?
        yuvBytes.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferCreateWithBytes(
                nil, width, height,
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                UnsafeMutableRawPointer(mutating: base),
                width, nil, nil, nil, &pixelBuffer)
            guard let pb = pixelBuffer else { return }
            let ci = CIImage(cvPixelBuffer: pb)
            let ctx = CIContext()
            if let cg = ctx.createCGImage(ci, from: ci.extent) {
                image = UIImage(cgImage: cg)
            }
        }
        return image
    }

    // MARK: - Error types

    enum SlcDetectorError: Error {
        case modelNotFound(String)
        case decodeFailed
    }
}

// MARK: - UIImage helpers

private extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    func toPixelBuffer() -> CVPixelBuffer? {
        guard let cgImage = cgImage else { return nil }
        let w = cgImage.width, h = cgImage.height
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, w, h,
                            kCVPixelFormatType_32BGRA, nil, &pb)
        guard let pixelBuffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                            width: w, height: h,
                            bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        ctx?.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        return pixelBuffer
    }
}
