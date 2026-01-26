import MLX
import XCTest
@testable import Flux2

final class Flux2LayerNormTests: XCTestCase {
  func testLayerNormMatchesReference() {
    MLXRandom.seed(0)

    let eps: Float = 1e-5
    let x = MLXRandom.normal([2, 17, 256]).asType(.float32)

    let layerNorm = Flux2LayerNorm(eps: eps)
    let actual = layerNorm(x)

    let mean = x.mean(axis: -1, keepDims: true)
    let variance = x.variance(axis: -1, keepDims: true)
    let expected = (x - mean) / sqrt(variance + eps)

    TestHelpers.assertAllClose(actual, expected, atol: 1e-4, rtol: 1e-4)
  }
}
