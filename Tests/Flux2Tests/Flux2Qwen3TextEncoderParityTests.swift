import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2Qwen3TextEncoderParityTests: XCTestCase {
  func testTinyQwen3PromptEmbedsMatchesFixtures() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny_text_encoder")
    let fixtureFile = fixtureRoot.appendingPathComponent("prompt_embeds.safetensors")

    XCTAssertTrue(FileManager.default.fileExists(atPath: fixtureFile.path))

    let reader = try SafeTensorsReader(fileURL: fixtureFile)
    let inputIds = try reader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try reader.tensor(named: "attention_mask").asType(.int32)
    let expectedEmbeds = try reader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try reader.tensor(named: "text_ids").asType(.int32)

    let encoder = try Flux2Qwen3TextEncoder.load(from: fixtureRoot, dtype: .float32)
    let actualEmbeds = try encoder.promptEmbeds(
      inputIds: inputIds,
      attentionMask: attentionMask,
      hiddenStateLayers: [0, 1, 2]
    )

    TestHelpers.assertAllClose(actualEmbeds, expectedEmbeds, atol: 1e-4, rtol: 1e-3)

    let actualTextIds = try Flux2PositionIds.prepareTextIds(actualEmbeds)
    XCTAssertEqual(actualTextIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))
  }
}

