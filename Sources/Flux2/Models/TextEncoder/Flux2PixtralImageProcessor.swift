import CoreGraphics
import Foundation
import MLX

public enum Flux2PixtralImageProcessorError: Error {
  case emptyImageList
  case invalidPatchSize(Int)
  case invalidLongestEdge(Int)
  case failedToRenderImage
}

public struct Flux2PixtralImageProcessorConfiguration: Sendable {
  public let longestEdge: Int
  public let doResize: Bool
  public let doRescale: Bool
  public let rescaleFactor: Float
  public let doNormalize: Bool
  public let imageMean: [Float]
  public let imageStd: [Float]

  public init(
    longestEdge: Int = 1540,
    doResize: Bool = true,
    doRescale: Bool = true,
    rescaleFactor: Float = 1.0 / 255.0,
    doNormalize: Bool = true,
    imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073],
    imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]
  ) {
    self.longestEdge = longestEdge
    self.doResize = doResize
    self.doRescale = doRescale
    self.rescaleFactor = rescaleFactor
    self.doNormalize = doNormalize
    self.imageMean = imageMean
    self.imageStd = imageStd
  }
}

public struct Flux2PixtralImageBatch {
  public let pixelValues: MLXArray
  public let imageSizes: [(height: Int, width: Int)]

  public init(pixelValues: MLXArray, imageSizes: [(height: Int, width: Int)]) {
    self.pixelValues = pixelValues
    self.imageSizes = imageSizes
  }
}

public final class Flux2PixtralImageProcessor {
  public let configuration: Flux2PixtralImageProcessorConfiguration

  public init(configuration: Flux2PixtralImageProcessorConfiguration = .init()) {
    self.configuration = configuration
  }

  public func preprocess(
    images: [CGImage],
    patchSize: Int
  ) throws -> Flux2PixtralImageBatch {
    guard !images.isEmpty else {
      throw Flux2PixtralImageProcessorError.emptyImageList
    }
    guard patchSize > 0 else {
      throw Flux2PixtralImageProcessorError.invalidPatchSize(patchSize)
    }
    guard configuration.longestEdge > 0 else {
      throw Flux2PixtralImageProcessorError.invalidLongestEdge(configuration.longestEdge)
    }

    var resizedSizes: [(height: Int, width: Int)] = []
    resizedSizes.reserveCapacity(images.count)

    var imageData: [[Float32]] = []
    imageData.reserveCapacity(images.count)

    var maxHeight = 0
    var maxWidth = 0

    for image in images {
      let size = configuration.doResize
        ? resizeOutputSize(image: image, longestEdge: configuration.longestEdge, patchSize: patchSize)
        : (height: image.height, width: image.width)

      let rgba = try renderRGBA(image: image, width: size.width, height: size.height)
      let chw = convertToCHW(rgba: rgba, width: size.width, height: size.height)

      resizedSizes.append(size)
      imageData.append(chw)
      maxHeight = max(maxHeight, size.height)
      maxWidth = max(maxWidth, size.width)
    }

    let count = images.count
    var padded = [Float32](repeating: 0, count: count * 3 * maxHeight * maxWidth)

    for (index, size) in resizedSizes.enumerated() {
      let src = imageData[index]
      for channel in 0..<3 {
        for y in 0..<size.height {
          let srcRowStart = channel * size.height * size.width + y * size.width
          let dstRowStart = index * 3 * maxHeight * maxWidth + channel * maxHeight * maxWidth + y * maxWidth
          padded[dstRowStart..<(dstRowStart + size.width)] = src[srcRowStart..<(srcRowStart + size.width)]
        }
      }
    }

    let pixelData = padded.withUnsafeBufferPointer { buffer in
      Data(buffer: buffer)
    }
    let pixelValues = MLXArray(pixelData, [count, 3, maxHeight, maxWidth], dtype: .float32)
    return Flux2PixtralImageBatch(pixelValues: pixelValues, imageSizes: resizedSizes)
  }

  private func resizeOutputSize(
    image: CGImage,
    longestEdge: Int,
    patchSize: Int
  ) -> (height: Int, width: Int) {
    let maxEdge = max(longestEdge, 1)

    var height = max(image.height, 1)
    var width = max(image.width, 1)

    let ratio = max(Double(height) / Double(maxEdge), Double(width) / Double(maxEdge))
    if ratio > 1 {
      height = max(Int(floor(Double(height) / ratio)), 1)
      width = max(Int(floor(Double(width) / ratio)), 1)
    }

    let numHeightTokens = ((height - 1) / patchSize) + 1
    let numWidthTokens = ((width - 1) / patchSize) + 1

    return (height: numHeightTokens * patchSize, width: numWidthTokens * patchSize)
  }

  private func renderRGBA(image: CGImage, width: Int, height: Int) throws -> [UInt8] {
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
        throw Flux2PixtralImageProcessorError.failedToRenderImage
      }

      context.interpolationQuality = .high
      context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
      context.fill(CGRect(x: 0, y: 0, width: width, height: height))
      context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    }

    return rgba
  }

  private func convertToCHW(rgba: [UInt8], width: Int, height: Int) -> [Float32] {
    let pixelCount = width * height
    var chw = [Float32](repeating: 0, count: pixelCount * 3)

    let scale: Float32 = configuration.doRescale ? Float32(configuration.rescaleFactor) : 1.0
    let mean = configuration.imageMean.map { Float32($0) }
    let std = configuration.imageStd.map { Float32($0) }

    for pixel in 0..<pixelCount {
      let base = pixel * 4
      var r = Float32(rgba[base]) * scale
      var g = Float32(rgba[base + 1]) * scale
      var b = Float32(rgba[base + 2]) * scale

      if configuration.doNormalize, mean.count >= 3, std.count >= 3 {
        r = (r - mean[0]) / std[0]
        g = (g - mean[1]) / std[1]
        b = (b - mean[2]) / std[2]
      }

      chw[pixel] = r
      chw[pixelCount + pixel] = g
      chw[(2 * pixelCount) + pixel] = b
    }

    return chw
  }
}
