import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2DevPipelineTests: XCTestCase {
  func testTinyDevPipelineRuns() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny_dev_pipeline")
    let inputsURL = fixtureRoot.appendingPathComponent("dev_inputs.safetensors")

    XCTAssertTrue(FileManager.default.fileExists(atPath: inputsURL.path))

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let inputIds = try inputsReader.tensor(named: "input_ids").asType(.int32)
    let attentionMask = try inputsReader.tensor(named: "attention_mask").asType(.int32)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)

    let pipeline = try Flux2DevPipeline(
      snapshot: fixtureRoot,
      dtype: .float32,
      hiddenStateLayers: [2],
      loadProcessor: false
    )

    let height = 8
    let width = 8
    let output = try pipeline.generateTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      height: height,
      width: width,
      numInferenceSteps: 3,
      latents: latents,
      guidanceScale: 4.0
    )

    let expectedURL = fixtureRoot.appendingPathComponent("dev_expected.safetensors")
    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))
    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)

    let expectedPacked = try expectedReader.tensor(named: "packed_latents").asType(.float32)
    let expectedDecoded = try expectedReader.tensor(named: "decoded").asType(.float32)
    let expectedPromptEmbeds = try expectedReader.tensor(named: "prompt_embeds").asType(.float32)
    let expectedTextIds = try expectedReader.tensor(named: "text_ids").asType(.int32)
    let expectedLatentIds = try expectedReader.tensor(named: "latent_ids").asType(.int32)

    TestHelpers.assertAllClose(output.packedLatents, expectedPacked, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.decoded, expectedDecoded, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.promptEmbeds, expectedPromptEmbeds, atol: 1e-4, rtol: 1e-3)
    XCTAssertEqual(output.textIds.asType(.int32).asArray(Int32.self), expectedTextIds.asArray(Int32.self))
    XCTAssertEqual(output.latentIds.asType(.int32).asArray(Int32.self), expectedLatentIds.asArray(Int32.self))

    XCTAssertEqual(output.decoded.ndim, 4)
    let batch = output.decoded.dim(0)
    XCTAssertEqual(output.decoded.dim(1), 3)
    XCTAssertEqual(output.decoded.dim(2), height)
    XCTAssertEqual(output.decoded.dim(3), width)

    XCTAssertEqual(output.promptEmbeds.dim(0), batch)
    XCTAssertEqual(output.textIds.dim(0), batch)
    XCTAssertEqual(output.latentIds.dim(0), batch)
    XCTAssertEqual(output.packedLatents.dim(0), batch)

    let decodedValues = output.decoded.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(decodedValues.isEmpty)
    XCTAssertFalse(decodedValues.contains { !$0.isFinite })

    let minValue = decodedValues.min() ?? 0
    let maxValue = decodedValues.max() ?? 0
    XCTAssertGreaterThanOrEqual(minValue, -1.05)
    XCTAssertLessThanOrEqual(maxValue, 1.05)
    XCTAssertGreaterThan(maxValue - minValue, 1e-4)
  }
}
