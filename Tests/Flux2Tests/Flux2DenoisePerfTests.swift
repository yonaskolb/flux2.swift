import Foundation
import MLX
import XCTest
@testable import Flux2

final class Flux2DenoisePerfTests: XCTestCase {
  private static func seconds(_ duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
  }

  func testTinyDenoiseLoopPerf() throws {
    #if !FLUX2_RUN_PERF_TESTS
      guard ProcessInfo.processInfo.environment["FLUX2_RUN_PERF_TESTS"] == "1" else {
        throw XCTSkip(
          "Enable perf tests by setting FLUX2_RUN_PERF_TESTS=1, or build with -DFLUX2_RUN_PERF_TESTS."
        )
      }
    #endif

    let repoRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()

    let fixtureRoot = repoRoot.appendingPathComponent("fixtures/flux2_tiny_klein_pipeline")
    let inputsURL = fixtureRoot.appendingPathComponent("klein_inputs.safetensors")

    try Device.withDefaultDevice(.gpu) {
      let inputsReader = try SafeTensorsReader(fileURL: inputsURL)
      let latentInput = try inputsReader.tensor(named: "latents").asType(.float32)
      let promptEmbeds = try inputsReader.tensor(named: "prompt_embeds").asType(.float32)
      let txtIds = try inputsReader.tensor(named: "text_ids").asType(.int32)
      let latentIds = try inputsReader.tensor(named: "latent_ids").asType(.int32)

      let batch = latentInput.dim(0)
      let latents = try Flux2LatentUtils.packLatents(latentInput)
      let tokenCount = latents.dim(1)

      let transformer = try Flux2Transformer2DModel.load(from: fixtureRoot, dtype: .float32)
      let scheduler = try FlowMatchEulerDiscreteScheduler.load(from: fixtureRoot)
      let vae = try Flux2AutoencoderKL.load(from: fixtureRoot, dtype: .float32)
      let pipeline = Flux2Pipeline(transformer: transformer, scheduler: scheduler, vae: vae)

      let modelTimestepScale: Float = 0.001
      let steps = 28
      let sigmas = scheduler.flux2DefaultSigmas(numInferenceSteps: steps)
      let mu = scheduler.config.useDynamicShifting
        ? FlowMatchEulerDiscreteScheduler.flux2EmpiricalMu(imageSeqLen: tokenCount, numSteps: steps)
        : nil

      MLX.eval(latents, promptEmbeds, txtIds, latentIds)

      func configureScheduler() throws -> [Float] {
        try scheduler.setTimesteps(numInferenceSteps: steps, sigmas: sigmas, mu: mu)
        scheduler.setBeginIndex(0)
        return scheduler.timestepsValues
      }

      func baseline(stepValues: [Float]) throws -> MLXArray {
        var current = latents

        for step in stepValues {
          let timestep = MLX.full([batch], values: step).asType(current.dtype)

          var transformerTimestep = timestep
          let scale = MLXArray(modelTimestepScale).asType(transformerTimestep.dtype)
          transformerTimestep = transformerTimestep * scale

          let noisePredAll = transformer(
            current,
            encoderHiddenStates: promptEmbeds,
            timestep: transformerTimestep,
            imgIds: latentIds,
            txtIds: txtIds,
            guidance: nil,
            attentionMask: .none
          )

          let noisePred = noisePredAll[0..., 0..<tokenCount, 0...]
          current = try scheduler.step(
            modelOutput: noisePred,
            timestep: timestep,
            sample: current
          ).prevSample
        }

        MLX.eval(current)
        return current
      }

      func optimized(stepValues: [Float]) throws -> MLXArray {
        let out = try pipeline.denoiseLoop(
          latents: latents,
          encoderHiddenStates: promptEmbeds,
          timestepValues: stepValues,
          latentIds: latentIds,
          txtIds: txtIds,
          modelTimestepScale: modelTimestepScale
        )
        MLX.eval(out)
        return out
      }

      _ = try baseline(stepValues: configureScheduler())
      _ = try optimized(stepValues: configureScheduler())

      let clock = ContinuousClock()

      func time(_ name: String, _ fn: () throws -> MLXArray) throws -> Double {
        let iters = 5
        let start = clock.now
        for _ in 0..<iters {
          _ = try fn()
        }
        let elapsed = Self.seconds(clock.now - start)
        let perIter = elapsed / Double(iters)
        let msPerStep = perIter / Double(steps) * 1000.0
        print("[perf] \(name): \(msPerStep) ms/step (\(steps) steps, \(iters) runs)")
        return perIter
      }

      let baselineS = try time("denoise_baseline_lazy_graph", {
        try baseline(stepValues: configureScheduler())
      })

      let optimizedS = try time("denoise_optimized_compiled", {
        try optimized(stepValues: configureScheduler())
      })

      print("[perf] denoise speedup: \(baselineS / optimizedS)x")
    }
  }
}
