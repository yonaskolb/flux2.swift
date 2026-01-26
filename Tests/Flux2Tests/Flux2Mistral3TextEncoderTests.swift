import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2Mistral3TextEncoderTests: XCTestCase {
  func testPromptEmbedsSmoke() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny_mistral3_text_encoder")
    let fixtureFile = fixtureRoot.appendingPathComponent("prompt_embeds.safetensors")

    XCTAssertTrue(FileManager.default.fileExists(atPath: fixtureFile.path))

    let reader = try SafeTensorsReader(fileURL: fixtureFile)
    let inputIds = try reader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try reader.tensor(named: "attention_mask").asType(.int32)
    let expectedEmbeds = try reader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try reader.tensor(named: "text_ids").asType(.int32)

    let encoder = try Flux2Mistral3TextEncoder.load(from: fixtureRoot, dtype: .float32)
    let actualEmbeds = try encoder.promptEmbeds(
      inputIds: inputIds,
      attentionMask: attentionMask,
      hiddenStateLayers: [0, 1, 2]
    )

    TestHelpers.assertAllClose(actualEmbeds, expectedEmbeds, atol: 1e-4, rtol: 1e-3)

    let batch = inputIds.dim(0)
    XCTAssertEqual(actualEmbeds.dim(0), batch)
    XCTAssertEqual(actualEmbeds.ndim, 3)

    let actualTextIds = try Flux2PositionIds.prepareTextIds(actualEmbeds)
    XCTAssertEqual(actualTextIds.dim(0), batch)
    XCTAssertEqual(actualTextIds.ndim, 3)
    XCTAssertEqual(actualTextIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))

    let embedValues = actualEmbeds.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(embedValues.isEmpty)
    XCTAssertFalse(embedValues.contains { !$0.isFinite })
  }

  func testLoadsLanguageModelModelPrefixWeights() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny_mistral3_text_encoder")
    let fixtureFile = fixtureRoot.appendingPathComponent("prompt_embeds.safetensors")
    let sourceWeightsURL = fixtureRoot.appendingPathComponent("text_encoder/model.safetensors")
    let sourceConfigURL = fixtureRoot.appendingPathComponent("text_encoder/config.json")

    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let tempTextEncoderDir = tempDir.appendingPathComponent("text_encoder")
    try FileManager.default.createDirectory(at: tempTextEncoderDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    try FileManager.default.copyItem(
      at: sourceConfigURL,
      to: tempTextEncoderDir.appendingPathComponent("config.json")
    )

    let weightsReader = try SafeTensorsReader(fileURL: sourceWeightsURL)
    let sourceWeights = try weightsReader.loadAllTensors(as: .float32)
    var renamedWeights: [String: MLXArray] = [:]
    for (name, tensor) in sourceWeights {
      guard name.hasPrefix("model.language_model.") else { continue }
      let suffix = name.dropFirst("model.language_model.".count)
      renamedWeights["language_model.model.\(suffix)"] = tensor
    }

    try MLX.save(
      arrays: renamedWeights,
      metadata: [:],
      url: tempTextEncoderDir.appendingPathComponent("model.safetensors")
    )

    let reader = try SafeTensorsReader(fileURL: fixtureFile)
    let inputIds = try reader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try reader.tensor(named: "attention_mask").asType(.int32)
    let expectedEmbeds = try reader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try reader.tensor(named: "text_ids").asType(.int32)

    let encoder = try Flux2Mistral3TextEncoder.load(from: tempDir, dtype: .float32)
    let actualEmbeds = try encoder.promptEmbeds(
      inputIds: inputIds,
      attentionMask: attentionMask,
      hiddenStateLayers: [0, 1, 2]
    )

    TestHelpers.assertAllClose(actualEmbeds, expectedEmbeds, atol: 1e-4, rtol: 1e-3)

    let batch = inputIds.dim(0)
    XCTAssertEqual(actualEmbeds.dim(0), batch)
    XCTAssertEqual(actualEmbeds.ndim, 3)

    let actualTextIds = try Flux2PositionIds.prepareTextIds(actualEmbeds)
    XCTAssertEqual(actualTextIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))

    let values = actualEmbeds.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(values.isEmpty)
    XCTAssertFalse(values.contains { !$0.isFinite })
  }
}
