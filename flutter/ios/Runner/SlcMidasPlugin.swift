// MiDaS-small monocular depth estimator — non-LiDAR fallback (~100ms).
//
// Required asset: assets/models/midas_small.mlpackage
//   Export from PyTorch with: torch.onnx.export(midas_small, ...) then coremltools.convert(...)
//   Input:  'input'  [1, 3, 256, 256] float32 (RGB, normalised ImageNet mean/std)
//   Output: 'output' [1, 1, 256, 256] float32 (relative inverse depth, larger = closer)
//
// Channel: slc/midas_depth -> method 'estimate'
// Args: yuv (Data), width, height, cx, cy, w, h, diameter_px
// Returns: 128*128*4 bytes (Float32LE height field in mm)
//
// Register in AppDelegate.swift:
//   SlcMidasPlugin.register(with: registrar(forPlugin: "SlcMidasPlugin")!)

import CoreML
import Flutter
import UIKit

class SlcMidasPlugin: NSObject, FlutterPlugin {

    static func register(with registrar: FlutterPluginRegistrar) {
        let plugin = SlcMidasPlugin()
        FlutterMethodChannel(name: "slc/midas_depth",
                             binaryMessenger: registrar.messenger())
            .setMethodCallHandler(plugin.handle)
    }

    private var model: MLModel?

    private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard call.method == "estimate" else { result(FlutterMethodNotImplemented); return }
        guard let args       = call.arguments as? [String: Any],
              let yuvData    = args["yuv"]         as? FlutterStandardTypedData,
              let width      = args["width"]        as? Int,
              let height     = args["height"]       as? Int,
              let cx         = (args["cx"]          as? NSNumber)?.doubleValue,
              let cy         = (args["cy"]          as? NSNumber)?.doubleValue,
              let bw         = (args["w"]           as? NSNumber)?.doubleValue,
              let bh         = (args["h"]           as? NSNumber)?.doubleValue,
              let diaPx      = (args["diameter_px"] as? NSNumber)?.doubleValue else {
            result(FlutterError(code: "BAD_ARGS", message: "missing args", details: nil))
            return
        }
        do {
            let field = try estimate(yuvBytes: yuvData.data,
                                     width: width, height: height,
                                     cx: cx, cy: cy, bw: bw, bh: bh, diaPx: diaPx)
            result(FlutterStandardTypedData(float32: field))
        } catch {
            result(FlutterError(code: "MIDAS_ERROR",
                                message: error.localizedDescription, details: nil))
        }
    }

    private func ensureModel() throws {
        guard model == nil else { return }
        guard let url = Bundle.main.url(forResource: "midas_small",
                                        withExtension: "mlpackage") else {
            throw SlcMidasError.modelNotFound
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: cfg)
    }

    private func estimate(yuvBytes: Data, width: Int, height: Int,
                          cx: Double, cy: Double,
                          bw: Double, bh: Double, diaPx: Double) throws -> Data {
        try ensureModel()

        // Decode YUV to UIImage
        guard let fullImage = yuvToUIImage(yuvBytes: yuvBytes, width: width, height: height)
        else { throw SlcMidasError.decodeFailed }

        // Crop to tile bbox (expanded by 10% for context)
        let margin = 0.1
        let cropRect = CGRect(
            x: (cx - bw / 2 * (1 + margin)).clamped(to: 0...Double(width  - 1)),
            y: (cy - bh / 2 * (1 + margin)).clamped(to: 0...Double(height - 1)),
            width:  min(bw * (1 + 2 * margin), Double(width)),
            height: min(bh * (1 + 2 * margin), Double(height))
        )
        guard let cropped = fullImage.cgImage?.cropping(to: cropRect),
              let cropUI   = UIImage(cgImage: cropped).resize(to: CGSize(width: 256, height: 256)),
              let pb        = cropUI.toNormalisedPixelBuffer() else {
            throw SlcMidasError.decodeFailed
        }

        // Run MiDaS
        let inp = try MLDictionaryFeatureProvider(dictionary: ["input": pb])
        let out = try model!.prediction(from: inp)
        guard let depthArr = out.featureValue(for: "output")?.multiArrayValue
        else { throw SlcMidasError.badOutput }

        // depthArr shape: [1, 1, 256, 256] — inverse relative depth (larger = closer)
        // Flip to get deeper = larger, then bilinear-downsample to 128x128
        // Calibrate to mm: we know the tile subtends diaPx pixels and 30mm diameter.
        // median depth within tile crop = tile surface depth; protrusions are above that.
        let mmPerPx = 30.0 / diaPx

        // Extract depth values from 256x256 array
        let depthSize = 256 * 256
        var depth256 = [Float](repeating: 0, count: depthSize)
        for i in 0..<depthSize {
            depth256[i] = depthArr[[0, 0, i / 256, i % 256] as [NSNumber]].floatValue
        }

        // Invert (MiDaS output is inverse depth: larger = closer)
        let minD = depth256.min() ?? 0
        let maxD = depth256.max() ?? 1
        let range = max(maxD - minD, 1e-6)
        // MiDaS output: larger = closer = taller protrusion (matches training sign).
        let relDepth = depth256.map { ($0 - minD) / range }  // 0=flat, 1=tallest protrusion

        // Bilinear downsample 256x256 -> 128x128
        var field128 = [Float32](repeating: 0, count: 128 * 128)
        for r in 0..<128 {
            for c in 0..<128 {
                let sr = r * 256 / 128
                let sc = c * 256 / 128
                field128[r * 128 + c] = Float32(Double(relDepth[sr * 256 + sc]) * mmPerPx * 10)
            }
        }

        return Data(bytes: field128, count: field128.count * 4)
    }

    // MARK: - Helpers

    private func yuvToUIImage(yuvBytes: Data, width: Int, height: Int) -> UIImage? {
        var image: UIImage?
        yuvBytes.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            var pb: CVPixelBuffer?
            CVPixelBufferCreateWithBytes(nil, width, height,
                                         kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                         UnsafeMutableRawPointer(mutating: base),
                                         width, nil, nil, nil, &pb)
            guard let buf = pb else { return }
            let ci = CIImage(cvPixelBuffer: buf)
            if let cg = CIContext().createCGImage(ci, from: ci.extent) {
                image = UIImage(cgImage: cg)
            }
        }
        return image
    }

    enum SlcMidasError: Error {
        case modelNotFound
        case decodeFailed
        case badOutput
    }
}

private extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}

private extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    // Returns [1, 3, 256, 256] NCHW pixel buffer normalised with ImageNet mean/std
    func toNormalisedPixelBuffer() -> CVPixelBuffer? {
        guard let cgImg = cgImage else { return nil }
        let w = 256, h = 256
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, nil, &pb)
        guard let pixelBuffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                            width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        ctx?.draw(cgImg, in: CGRect(x: 0, y: 0, width: w, height: h))
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        return pixelBuffer
    }
}
