import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2VAEParityTests: XCTestCase {
  func testTinyVaeEncodeDecodeMatchesFixtures() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny")

    let inputsURL = fixtureRoot.appendingPathComponent("vae_inputs.safetensors")
    let expectedURL = fixtureRoot.appendingPathComponent("vae_expected.safetensors")
    XCTAssertTrue(FileManager.default.fileExists(atPath: inputsURL.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let image = try inputsReader.tensor(named: "image").asType(.float32)

    let vae = try Flux2AutoencoderKL.load(from: fixtureRoot, dtype: .float32)
    let posterior = vae.encode(image)
    let moments = posterior.parameters.asType(.float32)
    let latents = posterior.mode().asType(.float32)
    let decoded = vae.decode(latents).asType(.float32)

    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expectedMoments = try expectedReader.tensor(named: "moments").asType(.float32)
    let expectedLatents = try expectedReader.tensor(named: "latents").asType(.float32)
    let expectedDecoded = try expectedReader.tensor(named: "decoded").asType(.float32)

    TestHelpers.assertAllClose(moments, expectedMoments, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(latents, expectedLatents, atol: 1e-4, rtol: 1e-3)
    TestHelpers.assertAllClose(decoded, expectedDecoded, atol: 1e-4, rtol: 1e-3)
  }
}

