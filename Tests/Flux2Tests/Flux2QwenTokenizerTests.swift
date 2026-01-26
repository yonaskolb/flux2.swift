import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2QwenTokenizerTests: XCTestCase {
  func testTokenizerLoadsAndEncodes() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_klein4b")
    let tokenizerDir = fixtureRoot.appendingPathComponent("tokenizer")

    XCTAssertTrue(FileManager.default.fileExists(atPath: tokenizerDir.path))

    let maxLength = 128
    let tokenizer = try Flux2QwenTokenizer.load(from: tokenizerDir, maxLengthOverride: maxLength)
    let prompt = "A fluffy orange cat sitting on a windowsill"
    let batch = try tokenizer.encode(
      prompts: [prompt],
      maxLength: maxLength,
      addGenerationPrompt: true,
      enableThinking: false
    )

    XCTAssertEqual(batch.inputIds.shape, [1, maxLength])
    XCTAssertEqual(batch.attentionMask.shape, [1, maxLength])
    XCTAssertEqual(batch.inputIds.dtype, .int32)
    XCTAssertEqual(batch.attentionMask.dtype, .int32)

    let maskValues = batch.attentionMask.asArray(Int32.self)
    XCTAssertEqual(maskValues.count, maxLength)
    XCTAssertGreaterThan(maskValues.reduce(0, +), 0)
  }
}
