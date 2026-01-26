import Foundation
import MLX
import XCTest
@testable import Flux2

final class FlowMatchEulerDiscreteSchedulerParityTests: XCTestCase {
  func testTinySchedulerMatchesFixtures() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny")

    let expectedURL = fixtureRoot.appendingPathComponent("scheduler_expected.safetensors")
    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))

    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expectedTimesteps = try expectedReader.tensor(named: "timesteps").asType(.float32)
    let expectedSigmas = try expectedReader.tensor(named: "sigmas").asType(.float32)
    let timestep = try expectedReader.tensor(named: "timestep").asType(.float32)
    let sample = try expectedReader.tensor(named: "sample").asType(.float32)
    let noise = try expectedReader.tensor(named: "noise").asType(.float32)
    let modelOutput = try expectedReader.tensor(named: "model_output").asType(.float32)
    let expectedScaled = try expectedReader.tensor(named: "scaled_sample").asType(.float32)
    let expectedPrev = try expectedReader.tensor(named: "prev_sample").asType(.float32)

    let scheduler = try FlowMatchEulerDiscreteScheduler.load(from: fixtureRoot)
    try scheduler.setTimesteps(numInferenceSteps: expectedTimesteps.dim(0))

    TestHelpers.assertAllClose(scheduler.timesteps, expectedTimesteps, atol: 1e-6, rtol: 1e-6)
    TestHelpers.assertAllClose(scheduler.sigmas, expectedSigmas, atol: 1e-6, rtol: 1e-6)

    let actualScaled = try scheduler.scaleNoise(sample: sample, timestep: timestep, noise: noise).asType(.float32)
    TestHelpers.assertAllClose(actualScaled, expectedScaled, atol: 1e-6, rtol: 1e-6)

    let actualPrev = try scheduler.step(modelOutput: modelOutput, timestep: timestep, sample: sample).prevSample.asType(.float32)
    TestHelpers.assertAllClose(actualPrev, expectedPrev, atol: 1e-6, rtol: 1e-6)
  }
}
