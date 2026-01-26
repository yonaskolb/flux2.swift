import Foundation
import CoreGraphics
import XCTest

@testable import Flux2

final class Flux2PixtralUpsamplingTests: XCTestCase {
  func testPixtralProcessorWithImageLongestEdgeOverridesConfig() throws {
    let tempRoot = try makeTemporaryTokenizerRoot(withMultimodalConfig: true)
    defer { try? FileManager.default.removeItem(at: tempRoot) }

    let processor = try Flux2PixtralProcessor.load(from: tempRoot)
    guard let originalMultimodal = processor.multimodal else {
      XCTFail("Expected multimodal config to load")
      return
    }
    XCTAssertEqual(originalMultimodal.imageProcessor.configuration.longestEdge, 1540)

    let overridden = try processor.withImageLongestEdge(768)
    guard let overriddenMultimodal = overridden.multimodal else {
      XCTFail("Expected multimodal config after override")
      return
    }

    XCTAssertEqual(originalMultimodal.visionPatchSize, overriddenMultimodal.visionPatchSize)
    XCTAssertEqual(originalMultimodal.spatialMergeSize, overriddenMultimodal.spatialMergeSize)
    XCTAssertEqual(originalMultimodal.imageTokenId, overriddenMultimodal.imageTokenId)
    XCTAssertEqual(originalMultimodal.imageBreakTokenId, overriddenMultimodal.imageBreakTokenId)
    XCTAssertEqual(originalMultimodal.imageEndTokenId, overriddenMultimodal.imageEndTokenId)
    XCTAssertEqual(originalMultimodal.imageProcessor.configuration.longestEdge, 1540)
    XCTAssertEqual(overriddenMultimodal.imageProcessor.configuration.longestEdge, 768)
    XCTAssertEqual(processor.maxLength, overridden.maxLength)
  }

  func testPixtralProcessorWithImageLongestEdgeThrowsWithoutMultimodalConfig() throws {
    let tokenizerRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_klein4b")
      .appendingPathComponent("tokenizer")
    let processor = try Flux2PixtralProcessor.load(from: tokenizerRoot)
    XCTAssertNil(processor.multimodal)

    XCTAssertThrowsError(try processor.withImageLongestEdge(768)) { error in
      guard case Flux2PixtralProcessorError.multimodalConfigurationMissing = error else {
        XCTFail("Unexpected error: \(error)")
        return
      }
    }
  }

  func testPixtralImageProcessorRespectsLongestEdgeConfiguration() throws {
    let image = try makeSolidImage(width: 1510, height: 816)
    let processor = Flux2PixtralImageProcessor(configuration: .init(longestEdge: 768))

    let batch = try processor.preprocess(images: [image], patchSize: 28)
    XCTAssertEqual(batch.imageSizes.count, 1)
    XCTAssertEqual(batch.imageSizes[0].width, 784)
    XCTAssertEqual(batch.imageSizes[0].height, 420)
    XCTAssertEqual(batch.pixelValues.shape, [1, 3, 420, 784])
  }

  func testExpandImageTokensMatchesExpectedCounts() throws {
    let imageTokenId = 10
    let imageBreakTokenId = 12
    let imageEndTokenId = 13
    let patchSize = 28
    let imageSize = (height: 840, width: 1512)

    let tokens = [999, imageTokenId, 888]
    let expanded = try Flux2PixtralProcessor.expandImageTokens(
      tokens: tokens,
      imageSizes: [imageSize],
      patchSize: patchSize,
      imageTokenId: imageTokenId,
      imageBreakTokenId: imageBreakTokenId,
      imageEndTokenId: imageEndTokenId
    )

    XCTAssertEqual(expanded.first, 999)
    XCTAssertEqual(expanded.last, 888)

    let inner = Array(expanded.dropFirst().dropLast())
    XCTAssertEqual(inner.count, (imageSize.height / patchSize) * ((imageSize.width / patchSize) + 1))
    XCTAssertEqual(inner.filter { $0 == imageTokenId }.count, (imageSize.height / patchSize) * (imageSize.width / patchSize))
    XCTAssertEqual(inner.filter { $0 == imageBreakTokenId }.count, (imageSize.height / patchSize) - 1)
    XCTAssertEqual(inner.filter { $0 == imageEndTokenId }.count, 1)
    XCTAssertEqual(inner.last, imageEndTokenId)
  }

  func testPixtralImageProcessorRoundsToPatchMultiple() throws {
    let image = try makeSolidImage(width: 1510, height: 816)
    let processor = Flux2PixtralImageProcessor()

    let batch = try processor.preprocess(images: [image], patchSize: 28)
    XCTAssertEqual(batch.imageSizes.count, 1)
    XCTAssertEqual(batch.imageSizes[0].width, 1512)
    XCTAssertEqual(batch.imageSizes[0].height, 840)
    XCTAssertEqual(batch.pixelValues.shape, [1, 3, 840, 1512])
  }

  private func makeSolidImage(width: Int, height: Int) throws -> CGImage {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    let bytesPerRow = width * 4

    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo.rawValue
    ) else {
      XCTFail("Failed to create CGContext")
      throw NSError(domain: "Flux2Tests", code: -1)
    }

    context.setFillColor(CGColor(red: 0.4, green: 0.5, blue: 0.6, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    guard let image = context.makeImage() else {
      XCTFail("Failed to create CGImage")
      throw NSError(domain: "Flux2Tests", code: -2)
    }

    return image
  }

  private func makeTemporaryTokenizerRoot(withMultimodalConfig: Bool) throws -> URL {
    let fixtureTokenizerDir = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_klein4b/tokenizer", isDirectory: true)

    let tempRoot = FileManager.default.temporaryDirectory
      .appendingPathComponent("flux2_pixtral_processor_test_\(UUID().uuidString)", isDirectory: true)
    let tempTokenizerDir = tempRoot.appendingPathComponent("tokenizer", isDirectory: true)
    try FileManager.default.createDirectory(at: tempTokenizerDir, withIntermediateDirectories: true)

    for name in ["tokenizer_config.json", "tokenizer.json"] {
      let src = fixtureTokenizerDir.appendingPathComponent(name)
      let dst = tempTokenizerDir.appendingPathComponent(name)
      try FileManager.default.copyItem(at: src, to: dst)
    }

    if withMultimodalConfig {
      let processorConfig = """
        {
          "patch_size": 28,
          "spatial_merge_size": 2,
          "image_token": "a",
          "image_break_token": "b",
          "image_end_token": "c"
        }
        """
      try processorConfig.write(
        to: tempTokenizerDir.appendingPathComponent("processor_config.json"),
        atomically: true,
        encoding: .utf8
      )

      let preprocessorConfig = """
        {
          "size": {
            "longest_edge": 1540
          }
        }
        """
      try preprocessorConfig.write(
        to: tempTokenizerDir.appendingPathComponent("preprocessor_config.json"),
        atomically: true,
        encoding: .utf8
      )
    }

    return tempRoot
  }
}
