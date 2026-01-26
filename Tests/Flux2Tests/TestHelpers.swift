import Foundation
import MLX
import XCTest

enum TestHelpers {
  static func writeSafeTensor(
    url: URL,
    tensorName: String,
    values: [Float32],
    shape: [Int]
  ) throws {
    let data = values.withUnsafeBytes { Data($0) }
    try writeSafeTensor(url: url, tensorName: tensorName, dtype: "F32", shape: shape, data: data)
  }

  static func writeSafeTensor(
    url: URL,
    tensorName: String,
    dtype: String,
    shape: [Int],
    data: Data
  ) throws {
    let header: [String: Any] = [
      tensorName: [
        "dtype": dtype,
        "shape": shape,
        "data_offsets": [0, data.count]
      ]
    ]
    let headerData = try JSONSerialization.data(withJSONObject: header, options: [])
    var headerLength = UInt64(headerData.count).littleEndian

    var fileData = Data()
    withUnsafeBytes(of: &headerLength) { rawBuffer in
      fileData.append(rawBuffer.bindMemory(to: UInt8.self))
    }
    fileData.append(headerData)
    fileData.append(data)
    try fileData.write(to: url, options: [.atomic])
  }

  static func assertAllClose(
    _ actual: MLXArray,
    _ expected: MLXArray,
    atol: Float = 1e-4,
    rtol: Float = 1e-4,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(actual.shape, expected.shape, "Shape mismatch", file: file, line: line)

    let a = actual.asType(.float32)
    let b = expected.asType(.float32)
    let diff = MLX.abs(a - b)
    let tol = MLXArray(atol) + MLXArray(rtol) * MLX.abs(b)
    let ok = (diff .<= tol).all().item(Bool.self)
    if ok { return }

    let maxDiff = diff.max().item(Float.self)
    let maxTol = tol.max().item(Float.self)
    XCTFail(
      "Arrays not close (atol=\(atol), rtol=\(rtol)). max(|a-b|)=\(maxDiff) max(tol)=\(maxTol)",
      file: file,
      line: line
    )
  }
}
