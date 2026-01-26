import MLX
import XCTest
@testable import Flux2

final class Flux2FusedKernelsPerfTests: XCTestCase {
  private static func seconds(_ duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }

  func testRoPEAndLayerNormPerf() throws {
    #if !FLUX2_RUN_PERF_TESTS
      guard ProcessInfo.processInfo.environment["FLUX2_RUN_PERF_TESTS"] == "1" else {
        throw XCTSkip(
          "Enable perf tests by building with -DFLUX2_RUN_PERF_TESTS (xcodebuild: SWIFT_ACTIVE_COMPILATION_CONDITIONS='$(inherited) FLUX2_RUN_PERF_TESTS'), or set FLUX2_RUN_PERF_TESTS=1 in the Xcode scheme."
        )
      }
    #endif

    try Device.withDefaultDevice(.gpu) {
      MLXRandom.seed(0)

      let batch = 2
      let seq = 768
      let heads = 24
      let headDim = 128

      let x = MLXRandom.normal([batch, seq, heads, headDim]).asType(.bfloat16)

      let halfDim = headDim / 2

      let cosHalfBatched = MLXRandom.uniform(0.0 ..< 1.0, [batch, seq, halfDim]).asType(.float32)
      let sinHalfBatched = MLXRandom.uniform(0.0 ..< 1.0, [batch, seq, halfDim]).asType(.float32)
      let cosBatched = MLX.stacked([cosHalfBatched, cosHalfBatched], axis: -1).reshaped(batch, seq, headDim)
      let sinBatched = MLX.stacked([sinHalfBatched, sinHalfBatched], axis: -1).reshaped(batch, seq, headDim)
      let rotaryBatched: Flux2RotaryEmbeddings = (cos: cosBatched, sin: sinBatched)

      let lnHidden = 3072
      let lnInput = MLXRandom.normal([batch, seq, lnHidden]).asType(.bfloat16)
      let lnEps: Float = 1e-6

      func ropeFusedBatched(_ x: MLXArray) -> MLXArray {
        Flux2FusedKernels.applyRotaryEmbBf16(x, cos: cosBatched, sin: sinBatched)
          ?? flux2ApplyRotaryEmbReference(x, rotaryBatched)
      }

      func ropeRefBatched(_ x: MLXArray) -> MLXArray {
        flux2ApplyRotaryEmbReference(x, rotaryBatched)
      }

      func layerNormFast(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, eps: lnEps)
      }

      func layerNormRef(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        return (x - mean) / sqrt(variance + lnEps)
      }

      MLX.eval(ropeFusedBatched(x))
      MLX.eval(ropeRefBatched(x))
      MLX.eval(layerNormFast(lnInput))
      MLX.eval(layerNormRef(lnInput))

      let warmIters = 5
      let iters = 20
      for _ in 0..<warmIters {
        MLX.eval(ropeFusedBatched(x))
        MLX.eval(ropeRefBatched(x))
        MLX.eval(layerNormFast(lnInput))
        MLX.eval(layerNormRef(lnInput))
      }

      let clock = ContinuousClock()

      func time(_ name: String, _ fn: () -> MLXArray) -> Double {
        let start = clock.now
        for _ in 0..<iters {
          MLX.eval(fn())
        }
        let elapsed = Self.seconds(clock.now - start)
        let perIter = elapsed / Double(iters)
        print("[perf] \(name): \(perIter * 1000.0) ms/iter (\(iters) iters)")
        return perIter
      }

      let ropeFusedBatchedS = time("rope_fused_batched", { ropeFusedBatched(x) })
      let ropeRefBatchedS = time("rope_ref_batched", { ropeRefBatched(x) })
      print("[perf] rope batched speedup: \(ropeRefBatchedS / ropeFusedBatchedS)x")

      let lnFastS = time("layer_norm_fast", { layerNormFast(lnInput) })
      let lnRefS = time("layer_norm_ref", { layerNormRef(lnInput) })
      print("[perf] layer_norm speedup: \(lnRefS / lnFastS)x")
    }
  }
}
