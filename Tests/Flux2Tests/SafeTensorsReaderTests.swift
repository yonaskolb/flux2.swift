import Foundation
import MLX
import XCTest
@testable import Flux2

final class SafeTensorsReaderTests: XCTestCase {
  private func withTempDir(_ body: (URL) throws -> Void) throws {
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }
    try body(tempDir)
  }

  func testReadsTensorValues() throws {
    try withTempDir { tempDir in
      let fileURL = tempDir.appendingPathComponent("weights.safetensors")
      let values: [Float32] = [1, 2, 3, 4]
      try TestHelpers.writeSafeTensor(
        url: fileURL,
        tensorName: "linear.weight",
        values: values,
        shape: [2, 2]
      )

      let reader = try SafeTensorsReader(fileURL: fileURL)
      let tensor = try reader.tensor(named: "linear.weight")

      XCTAssertEqual(tensor.shape, [2, 2])
      XCTAssertEqual(tensor.dtype, .float32)

      let loaded = tensor.asArray(Float32.self)
      XCTAssertEqual(loaded, values)
    }
  }

  func testReadsIntScalars() throws {
    try withTempDir { tempDir in
      let tests: [(dtype: String, value: Int)] = [
        ("I32", 42),
        ("I64", 43),
        ("U32", 44),
        ("U64", 45),
      ]

      for (dtype, value) in tests {
        let fileURL = tempDir.appendingPathComponent("scalar_\(dtype).safetensors")

        let data: Data = {
          switch dtype {
          case "I32":
            var raw = Int32(value).littleEndian
            return withUnsafeBytes(of: &raw) { Data($0) }
          case "I64":
            var raw = Int64(value).littleEndian
            return withUnsafeBytes(of: &raw) { Data($0) }
          case "U32":
            var raw = UInt32(value).littleEndian
            return withUnsafeBytes(of: &raw) { Data($0) }
          case "U64":
            var raw = UInt64(value).littleEndian
            return withUnsafeBytes(of: &raw) { Data($0) }
          default:
            XCTFail("Unhandled dtype \(dtype)")
            return Data()
          }
        }()

        try TestHelpers.writeSafeTensor(
          url: fileURL,
          tensorName: "scalar",
          dtype: dtype,
          shape: [1],
          data: data
        )

        let reader = try SafeTensorsReader(fileURL: fileURL)
        let loaded = try reader.intScalar(named: "scalar")
        XCTAssertEqual(loaded, value)
      }
    }
  }

  func testIntScalarRejectsNonScalarShapes() throws {
    try withTempDir { tempDir in
      let fileURL = tempDir.appendingPathComponent("not_scalar.safetensors")
      let values: [Int32] = [1, 2]
      let data = values.withUnsafeBytes { Data($0) }

      try TestHelpers.writeSafeTensor(
        url: fileURL,
        tensorName: "x",
        dtype: "I32",
        shape: [2],
        data: data
      )

      let reader = try SafeTensorsReader(fileURL: fileURL)
      XCTAssertThrowsError(try reader.intScalar(named: "x")) { error in
        guard case SafeTensorsReaderError.invalidShape(name: "x") = error else {
          XCTFail("Unexpected error \(error)")
          return
        }
      }
    }
  }

  func testIntScalarRejectsUnsupportedDType() throws {
    try withTempDir { tempDir in
      let fileURL = tempDir.appendingPathComponent("float_scalar.safetensors")
      var value: Float32 = 1.0
      let data = withUnsafeBytes(of: &value) { Data($0) }

      try TestHelpers.writeSafeTensor(
        url: fileURL,
        tensorName: "x",
        dtype: "F32",
        shape: [1],
        data: data
      )

      let reader = try SafeTensorsReader(fileURL: fileURL)
      XCTAssertThrowsError(try reader.intScalar(named: "x")) { error in
        guard case SafeTensorsReaderError.unsupportedScalarDType(name: "x", dtype: _) = error else {
          XCTFail("Unexpected error \(error)")
          return
        }
      }
    }
  }
}
