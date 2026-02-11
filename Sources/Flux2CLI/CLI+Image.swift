import CoreGraphics
import Foundation
import Flux2CLICore
import ImageIO
import MLX
import UniformTypeIdentifiers

extension CLI {
  private enum ResizeMode {
    case stretch
    case crop
  }

  struct ConditioningImage {
    let array: MLXArray
    let height: Int
    let width: Int
    let original: CGImage
  }

  private static func loadDataFromSpec(
    _ spec: String,
    remoteTimeout: TimeInterval = 60,
    remoteMaximumBytes: Int = 50 * 1024 * 1024
  ) async throws -> Data {
    try await CLIImageDataLoader.loadDataFromSpec(
      spec,
      remoteTimeout: remoteTimeout,
      remoteMaximumBytes: remoteMaximumBytes
    )
  }

  static func loadImage(spec: String, height: Int, width: Int) async throws -> MLXArray {
    guard height > 0, width > 0 else {
      throw CLIError.invalidOption("Invalid image size \(width)x\(height)")
    }

    let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
    let data = try await loadDataFromSpec(trimmed)
    let cgImage = try decodeCGImage(data: data, spec: trimmed)
    let rgba = try renderRGBA(image: cgImage, width: width, height: height, resizeMode: .crop)

    let pixelCount = width * height
    var floats = [Float32](repeating: 0, count: pixelCount * 3)
    for pixel in 0..<pixelCount {
      let base = pixel * 4
      let r = Float32(rgba[base]) / 255.0
      let g = Float32(rgba[base + 1]) / 255.0
      let b = Float32(rgba[base + 2]) / 255.0

      let valueIndex = pixel
      floats[valueIndex] = (r * 2.0) - 1.0
      floats[pixelCount + valueIndex] = (g * 2.0) - 1.0
      floats[(2 * pixelCount) + valueIndex] = (b * 2.0) - 1.0
    }

    let floatData = floats.withUnsafeBufferPointer { buffer in
      Data(buffer: buffer)
    }
    return MLXArray(floatData, [1, 3, height, width], dtype: .float32)
  }

  static func loadConditioningImage(
    spec: String,
    maxArea: Int = 1024 * 1024,
    multipleOf: Int = 16
  ) async throws -> ConditioningImage {
    guard maxArea > 0 else {
      throw CLIError.invalidOption("Invalid maxArea: \(maxArea)")
    }
    guard multipleOf > 0 else {
      throw CLIError.invalidOption("Invalid multipleOf: \(multipleOf)")
    }

    let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
    let data = try await loadDataFromSpec(trimmed)
    var cgImage = try decodeCGImage(data: data, spec: trimmed)
    let originalImage = cgImage

    // Match diffusers Flux2ImageProcessor behavior:
    // - If pixel area exceeds 1024^2, downscale proportionally to target area.
    // - Floor width/height to multiples of 16 (vae_scale_factor * 2).
    // - Preprocess with resize_mode="crop" and normalize to [-1, 1].
    let originalWidth = max(cgImage.width, 1)
    let originalHeight = max(cgImage.height, 1)
    let pixelArea = originalWidth * originalHeight
    if pixelArea > maxArea {
      let scale = (Double(maxArea) / Double(pixelArea)).squareRoot()
      let resizedWidth = max(Int(Double(originalWidth) * scale), 1)
      let resizedHeight = max(Int(Double(originalHeight) * scale), 1)
      cgImage = try resizeCGImage(image: cgImage, width: resizedWidth, height: resizedHeight)
    }

    var width = (cgImage.width / multipleOf) * multipleOf
    var height = (cgImage.height / multipleOf) * multipleOf
    width = max(width, multipleOf)
    height = max(height, multipleOf)

    let rgba = try renderRGBA(image: cgImage, width: width, height: height, resizeMode: .crop)

    let pixelCount = width * height
    var floats = [Float32](repeating: 0, count: pixelCount * 3)
    for pixel in 0..<pixelCount {
      let base = pixel * 4
      let r = Float32(rgba[base]) / 255.0
      let g = Float32(rgba[base + 1]) / 255.0
      let b = Float32(rgba[base + 2]) / 255.0

      let valueIndex = pixel
      floats[valueIndex] = (r * 2.0) - 1.0
      floats[pixelCount + valueIndex] = (g * 2.0) - 1.0
      floats[(2 * pixelCount) + valueIndex] = (b * 2.0) - 1.0
    }

    let floatData = floats.withUnsafeBufferPointer { buffer in
      Data(buffer: buffer)
    }

    return ConditioningImage(
      array: MLXArray(floatData, [1, 3, height, width], dtype: .float32),
      height: height,
      width: width,
      original: originalImage
    )
  }

  private static func decodeCGImage(data: Data, spec: String) throws -> CGImage {
    let options: [CFString: Any] = [
      kCGImageSourceShouldCache: false,
    ]

    guard let source = CGImageSourceCreateWithData(data as CFData, options as CFDictionary) else {
      throw CLIError.invalidOption("Failed to decode image: \(spec)")
    }
    guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, options as CFDictionary) else {
      throw CLIError.invalidOption("Failed to decode image: \(spec)")
    }
    return cgImage
  }

  private static func resizeCGImage(image: CGImage, width: Int, height: Int) throws -> CGImage {
    let rgba = try renderRGBA(image: image, width: width, height: height, resizeMode: .stretch)
    let data = Data(rgba)
    guard let provider = CGDataProvider(data: data as CFData) else {
      throw CLIError.invalidOption("Failed to create resized image provider")
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    guard let cgImage = CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: provider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent
    ) else {
      throw CLIError.invalidOption("Failed to create resized image")
    }

    return cgImage
  }

  private static func renderRGBA(image: CGImage, width: Int, height: Int, resizeMode: ResizeMode) throws -> [UInt8] {
    var rgba = [UInt8](repeating: 0, count: width * height * 4)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    let bytesPerRow = width * 4

    try rgba.withUnsafeMutableBytes { buffer in
      guard let context = CGContext(
        data: buffer.baseAddress,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
      ) else {
        throw CLIError.invalidOption("Failed to create image context")
      }

      context.interpolationQuality = .high

      let drawRect: CGRect
      switch resizeMode {
      case .stretch:
        drawRect = CGRect(x: 0, y: 0, width: width, height: height)
      case .crop:
        // Match diffusers VaeImageProcessor._resize_and_crop integer math.
        let targetRatio = Double(width) / Double(height)
        let srcRatio = Double(image.width) / Double(image.height)

        let srcW: Int
        if targetRatio > srcRatio {
          srcW = width
        } else {
          srcW = (image.width * height) / max(image.height, 1)
        }

        let srcH: Int
        if targetRatio <= srcRatio {
          srcH = height
        } else {
          srcH = (image.height * width) / max(image.width, 1)
        }

        let originX = (width / 2) - (srcW / 2)
        let originY = (height / 2) - (srcH / 2)
        drawRect = CGRect(x: originX, y: originY, width: srcW, height: srcH)
      }

      context.draw(image, in: drawRect)
    }

    return rgba
  }

  static func resolveOutputURL(_ path: String) -> URL {
    let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
    let url = URL(fileURLWithPath: trimmed.isEmpty ? "flux2.png" : trimmed)
    if url.pathExtension.isEmpty {
      return url.appendingPathExtension("png")
    }
    return url
  }

  static func writeImage(image: MLXArray, url: URL) throws {
    let outputURL = url.standardizedFileURL
    try FileManager.default.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true,
      attributes: nil
    )

    guard image.ndim == 4 else {
      throw CLIError.invalidOption("decoded image must be NCHW")
    }

    let batch = image.dim(0)
    guard batch >= 1 else {
      throw CLIError.invalidOption("decoded image batch is empty")
    }

    let channels = image.dim(1)
    guard channels == 3 else {
      throw CLIError.invalidOption("decoded image must have 3 channels, got \(channels)")
    }

    let height = image.dim(2)
    let width = image.dim(3)

    var rgb = image[0].asType(.float32)
    rgb = (rgb / MLXArray(2.0)) + MLXArray(0.5)
    rgb = MLX.clip(rgb, min: 0.0, max: 1.0)
    rgb = rgb * MLXArray(255.0)
    rgb = rgb.transposed(1, 2, 0)
    let flat = rgb.reshaped(-1).asType(.uint8).asArray(UInt8.self)

    var rgba = [UInt8](repeating: 255, count: width * height * 4)
    var rgbIndex = 0
    for pixel in 0..<(width * height) {
      rgba[pixel * 4] = flat[rgbIndex]
      rgba[pixel * 4 + 1] = flat[rgbIndex + 1]
      rgba[pixel * 4 + 2] = flat[rgbIndex + 2]
      rgbIndex += 3
    }

    let data = Data(rgba)
    guard let provider = CGDataProvider(data: data as CFData) else {
      throw CLIError.invalidOption("Failed to create image provider")
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
    guard let cgImage = CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: provider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent
    ) else {
      throw CLIError.invalidOption("Failed to create image")
    }

    let ext = url.pathExtension.lowercased()
    let type: UTType
    let properties: CFDictionary?
    switch ext {
    case "png":
      type = .png
      properties = nil
    case "jpg", "jpeg":
      type = .jpeg
      properties = [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary
    default:
      throw CLIError.invalidOption("Unsupported output format: \(ext). Supported: png, jpg, jpeg")
    }

    guard let destination = CGImageDestinationCreateWithURL(
      outputURL as CFURL,
      type.identifier as CFString,
      1,
      nil
    ) else {
      throw CLIError.invalidOption("Failed to create image destination")
    }

    CGImageDestinationAddImage(destination, cgImage, properties)
    guard CGImageDestinationFinalize(destination) else {
      throw CLIError.invalidOption("Failed to write image")
    }
  }
}
