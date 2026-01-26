import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2PipelineGuidanceTests: XCTestCase {
  func testTinyPipelineGuidanceRuns() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny_pipeline_guidance")
    let inputsURL = fixtureRoot.appendingPathComponent("pipeline_inputs.safetensors")

    XCTAssertTrue(FileManager.default.fileExists(atPath: inputsURL.path))

    let transformer = try Flux2Transformer2DModel.load(from: fixtureRoot, dtype: .float32)
    let scheduler = try FlowMatchEulerDiscreteScheduler.load(from: fixtureRoot)
    let vae = try Flux2AutoencoderKL.load(from: fixtureRoot, dtype: .float32)

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let latents = try inputsReader.tensor(named: "latents").asType(.float32)
    let encoderHiddenStates = try inputsReader.tensor(named: "encoder_hidden_states").asType(.float32)
    let timesteps = try inputsReader.tensor(named: "timesteps")
    let txtIds = try inputsReader.tensor(named: "txt_ids").asType(.int32)
    let latentIds = try inputsReader.tensor(named: "latent_ids").asType(.int32)
    let guidance = try inputsReader.tensor(named: "guidance").asType(.float32)

    try scheduler.setTimesteps(numInferenceSteps: timesteps.dim(0))
    scheduler.setBeginIndex(0)

    let pipeline = Flux2Pipeline(transformer: transformer, scheduler: scheduler, vae: vae)
    let output = try pipeline.generate(
      latents: latents,
      encoderHiddenStates: encoderHiddenStates,
      latentIds: latentIds,
      txtIds: txtIds,
      guidance: guidance
    )

    let expectedURL = fixtureRoot.appendingPathComponent("pipeline_expected.safetensors")
    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))

    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expectedPacked = try expectedReader.tensor(named: "packed_latents").asType(.float32)
    let expectedDecoded = try expectedReader.tensor(named: "decoded").asType(.float32)
    TestHelpers.assertAllClose(output.packedLatents, expectedPacked, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(output.decoded, expectedDecoded, atol: 1e-4, rtol: 1e-3)

    XCTAssertEqual(output.decoded.ndim, 4)
    XCTAssertEqual(output.decoded.dim(0), 1)
    XCTAssertEqual(output.decoded.dim(1), 3)

    let decodedValues = output.decoded.asType(.float32).asArray(Float32.self)
    XCTAssertFalse(decodedValues.isEmpty)
    XCTAssertFalse(decodedValues.contains { !$0.isFinite })
  }
}
