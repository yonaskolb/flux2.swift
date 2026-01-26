import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2TransformerParityTests: XCTestCase {
  func testTinyTransformerMatchesFixtures() throws {
    let fixtureRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("fixtures/flux2_tiny")

    let inputsURL = fixtureRoot.appendingPathComponent("transformer_inputs.safetensors")
    let expectedURL = fixtureRoot.appendingPathComponent("transformer_expected.safetensors")
    XCTAssertTrue(FileManager.default.fileExists(atPath: inputsURL.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: expectedURL.path))

    let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
    let hiddenStates = try inputsReader.tensor(named: "hidden_states").asType(.float32)
    let encoderHiddenStates = try inputsReader.tensor(named: "encoder_hidden_states").asType(.float32)
    let timestep = try inputsReader.tensor(named: "timestep").asType(.float32)
    let imgIds = try inputsReader.tensor(named: "img_ids").asType(.int32)
    let txtIds = try inputsReader.tensor(named: "txt_ids").asType(.int32)

    let transformer = try Flux2Transformer2DModel.load(from: fixtureRoot, dtype: .float32)
    let actual = transformer(
      hiddenStates,
      encoderHiddenStates: encoderHiddenStates,
      timestep: timestep,
      imgIds: imgIds,
      txtIds: txtIds
    ).asType(.float32)

    let expectedReader = try SafeTensorsReader(fileURL: expectedURL)
    let expected = try expectedReader.tensor(named: "output").asType(.float32)
    TestHelpers.assertAllClose(actual, expected, atol: 1e-4, rtol: 1e-3)
  }
}

