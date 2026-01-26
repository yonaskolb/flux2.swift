import MLX
import XCTest
@testable import Flux2

final class Flux2FusedKernelsTests: XCTestCase {
  func testFusedRoPEMatchesReference() throws {
    try Device.withDefaultDevice(.gpu) {
      MLXRandom.seed(0)

      let batch = 2
      let seq = 13
      let heads = 3
      let headDim = 32

      let halfDim = headDim / 2
      let cosHalf = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)
      let sinHalf = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)

      let cos = MLX.stacked([cosHalf, cosHalf], axis: -1).reshaped(seq, headDim)
      let sin = MLX.stacked([sinHalf, sinHalf], axis: -1).reshaped(seq, headDim)
      let rotary: Flux2RotaryEmbeddings = (cos: cos, sin: sin)

      func runCase(dtype: DType, name: String) {
        let x = MLXRandom.normal([batch, seq, heads, headDim]).asType(dtype)

        guard let fused = Flux2FusedKernels.applyRotaryEmb(x, cos: cos, sin: sin) else {
          XCTFail("Expected fused RoPE kernel to be available on GPU (\(name)).")
          return
        }

        let reference = flux2ApplyRotaryEmbReference(x, rotary, sequenceDim: 1)
        TestHelpers.assertAllClose(fused, reference, atol: 1e-2, rtol: 1e-2)
      }

      runCase(dtype: .bfloat16, name: "bfloat16")
      runCase(dtype: .float16, name: "float16")
      runCase(dtype: .float32, name: "float32")
    }
  }

  func testFusedRoPEMatchesReferenceBatchedRotary() throws {
    try Device.withDefaultDevice(.gpu) {
      MLXRandom.seed(0)

      let batch = 2
      let seq = 13
      let heads = 3
      let headDim = 32

      let halfDim = headDim / 2
      let cosHalf = MLXRandom.uniform(0.0 ..< 1.0, [batch, seq, halfDim]).asType(.float32)
      let sinHalf = MLXRandom.uniform(0.0 ..< 1.0, [batch, seq, halfDim]).asType(.float32)

      let cos = MLX.stacked([cosHalf, cosHalf], axis: -1).reshaped(batch, seq, headDim)
      let sin = MLX.stacked([sinHalf, sinHalf], axis: -1).reshaped(batch, seq, headDim)
      let rotary: Flux2RotaryEmbeddings = (cos: cos, sin: sin)

      func runCase(dtype: DType, name: String) {
        let x = MLXRandom.normal([batch, seq, heads, headDim]).asType(dtype)

        guard let fused = Flux2FusedKernels.applyRotaryEmb(x, cos: cos, sin: sin) else {
          XCTFail("Expected fused RoPE kernel to be available on GPU (\(name)).")
          return
        }

        let reference = flux2ApplyRotaryEmbReference(x, rotary, sequenceDim: 1)
        TestHelpers.assertAllClose(fused, reference, atol: 1e-2, rtol: 1e-2)
      }

      runCase(dtype: .bfloat16, name: "bfloat16")
      runCase(dtype: .float16, name: "float16")
      runCase(dtype: .float32, name: "float32")
    }
  }

  func testFusedRoPEMatchesReferenceBatchedRotaryBatch1() throws {
    try Device.withDefaultDevice(.gpu) {
      MLXRandom.seed(0)

      let batch = 1
      let seq = 13
      let heads = 3
      let headDim = 32

      let halfDim = headDim / 2
      let cosHalf = MLXRandom.uniform(0.0 ..< 1.0, [batch, seq, halfDim]).asType(.float32)
      let sinHalf = MLXRandom.uniform(0.0 ..< 1.0, [batch, seq, halfDim]).asType(.float32)

      let cos = MLX.stacked([cosHalf, cosHalf], axis: -1).reshaped(batch, seq, headDim)
      let sin = MLX.stacked([sinHalf, sinHalf], axis: -1).reshaped(batch, seq, headDim)
      let rotary: Flux2RotaryEmbeddings = (cos: cos, sin: sin)

      func runCase(dtype: DType, name: String) {
        let x = MLXRandom.normal([batch, seq, heads, headDim]).asType(dtype)

        guard let fused = Flux2FusedKernels.applyRotaryEmb(x, cos: cos, sin: sin) else {
          XCTFail("Expected fused RoPE kernel to be available on GPU (\(name)).")
          return
        }

        let reference = flux2ApplyRotaryEmbReference(x, rotary, sequenceDim: 1)
        TestHelpers.assertAllClose(fused, reference, atol: 1e-2, rtol: 1e-2)
      }

      runCase(dtype: .bfloat16, name: "bfloat16")
      runCase(dtype: .float16, name: "float16")
      runCase(dtype: .float32, name: "float32")
    }
  }

  func testFusedRoPEBatchedRotaryUsesPerBatchEmbeddings() throws {
    try Device.withDefaultDevice(.gpu) {
      MLXRandom.seed(0)

      let batch = 2
      let seq = 13
      let heads = 3
      let headDim = 32
      let halfDim = headDim / 2

      let cosHalf0 = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)
      let sinHalf0 = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)
      let cosHalf1 = cosHalf0 + MLXArray(0.5)
      let sinHalf1 = sinHalf0 + MLXArray(0.25)

      let cosHalf = MLX.stacked([cosHalf0, cosHalf1], axis: 0)
      let sinHalf = MLX.stacked([sinHalf0, sinHalf1], axis: 0)

      let cos = MLX.stacked([cosHalf, cosHalf], axis: -1).reshaped(batch, seq, headDim)
      let sin = MLX.stacked([sinHalf, sinHalf], axis: -1).reshaped(batch, seq, headDim)
      let rotary: Flux2RotaryEmbeddings = (cos: cos, sin: sin)

      let x0 = MLXRandom.normal([seq, heads, headDim]).asType(.float16)
      let x = MLX.stacked([x0, x0], axis: 0)

      guard let fused = Flux2FusedKernels.applyRotaryEmb(x, cos: cos, sin: sin) else {
        XCTFail("Expected fused RoPE kernel to be available on GPU.")
        return
      }

      let reference = flux2ApplyRotaryEmbReference(x, rotary, sequenceDim: 1)
      TestHelpers.assertAllClose(fused, reference, atol: 1e-2, rtol: 1e-2)

      let maxDiff = MLX.abs(fused[0] - fused[1]).max().item(Float.self)
      XCTAssertGreaterThan(maxDiff, 1e-3, "Expected per-batch rotary embeddings to change the output.")
    }
  }

  func testRoPECPUFallbackUsesReference() throws {
    try Device.withDefaultDevice(.cpu) {
      MLXRandom.seed(0)

      let batch = 2
      let seq = 13
      let heads = 3
      let headDim = 32

      let x = MLXRandom.normal([batch, seq, heads, headDim]).asType(.float32)

      let halfDim = headDim / 2
      let cosHalf = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)
      let sinHalf = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)
      let cos = MLX.stacked([cosHalf, cosHalf], axis: -1).reshaped(seq, headDim)
      let sin = MLX.stacked([sinHalf, sinHalf], axis: -1).reshaped(seq, headDim)
      let rotary: Flux2RotaryEmbeddings = (cos: cos, sin: sin)

      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(x, cos: cos, sin: sin))

      let actual = flux2ApplyRotaryEmb(x, rotary, sequenceDim: 1)
      let expected = flux2ApplyRotaryEmbReference(x, rotary, sequenceDim: 1)
      TestHelpers.assertAllClose(actual, expected, atol: 1e-5, rtol: 1e-5)
    }
  }

  func testRoPEFusedRejectsInvalidInputs() throws {
    try Device.withDefaultDevice(.gpu) {
      MLXRandom.seed(0)

      let batch = 2
      let seq = 13
      let heads = 3
      let headDim = 32

      let x = MLXRandom.normal([batch, seq, heads, headDim]).asType(.float16)

      let halfDim = headDim / 2
      let cosHalf = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)
      let sinHalf = MLXRandom.uniform(0.0 ..< 1.0, [seq, halfDim]).asType(.float32)
      let cos = MLX.stacked([cosHalf, cosHalf], axis: -1).reshaped(seq, headDim)
      let sin = MLX.stacked([sinHalf, sinHalf], axis: -1).reshaped(seq, headDim)

      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(x, cos: cos.asType(.float16), sin: sin))
      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(x, cos: cos, sin: sin.asType(.float16)))

      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(x, cos: cos[0..<seq - 1], sin: sin[0..<seq - 1]))
      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(x, cos: cos[0..., 0..<headDim - 1], sin: sin[0..., 0..<headDim - 1]))

      let cosBatched = cos.expandedDimensions(axis: 0)
      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(x, cos: cosBatched, sin: sin))

      let headDimOdd = 31
      let xOdd = MLXRandom.normal([batch, seq, heads, headDimOdd]).asType(.float16)
      let cosOdd = MLXRandom.uniform(0.0 ..< 1.0, [seq, headDimOdd]).asType(.float32)
      let sinOdd = MLXRandom.uniform(0.0 ..< 1.0, [seq, headDimOdd]).asType(.float32)
      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(xOdd, cos: cosOdd, sin: sinOdd))

      let xWrongRank = MLXRandom.normal([batch, seq, heads * headDim]).asType(.float16)
      XCTAssertNil(Flux2FusedKernels.applyRotaryEmb(xWrongRank, cos: cos, sin: sin))
    }
  }
}
