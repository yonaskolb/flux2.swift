import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2WeightsLoaderTests: XCTestCase {
  func testLoadsComponentWeights() throws {
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let componentDir = tempDir.appendingPathComponent("transformer")
    try FileManager.default.createDirectory(at: componentDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let fileURL = componentDir.appendingPathComponent("model.safetensors")
    let values: [Float32] = [5, 6, 7, 8]
    try TestHelpers.writeSafeTensor(
      url: fileURL,
      tensorName: "proj.weight",
      values: values,
      shape: [2, 2]
    )

    let loader = Flux2WeightsLoader(snapshot: tempDir)
    let tensors = try loader.load(component: .transformer, dtype: .float32)
    let tensor = try XCTUnwrap(tensors["proj.weight"])

    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.asArray(Float32.self), values)
  }

  func testLoadFiltersTensorNames() throws {
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let componentDir = tempDir.appendingPathComponent("transformer")
    try FileManager.default.createDirectory(at: componentDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    try TestHelpers.writeSafeTensor(
      url: componentDir.appendingPathComponent("model-a.safetensors"),
      tensorName: "a.weight",
      values: [1, 2, 3, 4],
      shape: [2, 2]
    )

    try TestHelpers.writeSafeTensor(
      url: componentDir.appendingPathComponent("model-b.safetensors"),
      tensorName: "b.weight",
      values: [9, 10, 11, 12],
      shape: [2, 2]
    )

    let loader = Flux2WeightsLoader(snapshot: tempDir)
    let tensors = try loader.load(component: .transformer, dtype: .float32) { name in
      name == "a.weight"
    }

    XCTAssertEqual(tensors.keys.sorted(), ["a.weight"])
    XCTAssertEqual(tensors["a.weight"]?.asArray(Float32.self), [1, 2, 3, 4])
  }
}
