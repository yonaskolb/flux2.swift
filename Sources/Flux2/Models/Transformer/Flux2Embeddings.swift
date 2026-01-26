import Foundation
import MLX
import MLXNN

final class Flux2Timesteps: Module, UnaryLayer {
  let numChannels: Int
  let flipSinToCos: Bool
  let downscaleFreqShift: Float
  let scale: Float
  let maxPeriod: Float

  init(
    numChannels: Int,
    flipSinToCos: Bool = false,
    downscaleFreqShift: Float = 1,
    scale: Float = 1,
    maxPeriod: Float = 10_000
  ) {
    self.numChannels = numChannels
    self.flipSinToCos = flipSinToCos
    self.downscaleFreqShift = downscaleFreqShift
    self.scale = scale
    self.maxPeriod = maxPeriod
  }

  func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
    precondition(timesteps.ndim == 1, "Timesteps should be a 1D array.")

    let halfDim = numChannels / 2
    let steps = timesteps.asType(.float32)

    var exponent = MLXArray(0..<halfDim).asType(.float32)
    exponent = exponent * (-log(maxPeriod))
    exponent = exponent / MLXArray(Float(halfDim) - downscaleFreqShift)

    var emb = MLX.exp(exponent)
    emb = steps.reshaped(-1, 1) * emb.reshaped(1, -1)

    if scale != 1 {
      emb = emb * scale
    }

    let sinEmb = MLX.sin(emb)
    let cosEmb = MLX.cos(emb)

    var out = flipSinToCos
      ? MLX.concatenated([cosEmb, sinEmb], axis: -1)
      : MLX.concatenated([sinEmb, cosEmb], axis: -1)

    if numChannels % 2 == 1 {
      let padding = MLX.zeros([out.dim(0), 1], dtype: out.dtype)
      out = MLX.concatenated([out, padding], axis: -1)
    }

    return out
  }
}

final class Flux2TimestepEmbedding: Module, UnaryLayer {
  @ModuleInfo(key: "linear_1") private var linear1: Linear
  @ModuleInfo(key: "linear_2") private var linear2: Linear

  init(inputChannels: Int, timeEmbedDim: Int, bias: Bool = true) {
    _linear1.wrappedValue = Linear(inputChannels, timeEmbedDim, bias: bias)
    _linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim, bias: bias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = linear1(x)
    out = silu(out)
    out = linear2(out)
    return out
  }
}

final class Flux2TimestepGuidanceEmbeddings: Module {
  private let timeProj: Flux2Timesteps
  @ModuleInfo(key: "timestep_embedder") private var timestepEmbedder: Flux2TimestepEmbedding
  @ModuleInfo(key: "guidance_embedder") private var guidanceEmbedder: Flux2TimestepEmbedding?

  init(
    inChannels: Int = 256,
    embeddingDim: Int,
    bias: Bool = false,
    guidanceEmbeds: Bool = true
  ) {
    timeProj = Flux2Timesteps(
      numChannels: inChannels,
      flipSinToCos: true,
      downscaleFreqShift: 0
    )
    _timestepEmbedder.wrappedValue = Flux2TimestepEmbedding(
      inputChannels: inChannels,
      timeEmbedDim: embeddingDim,
      bias: bias
    )

    if guidanceEmbeds {
      _guidanceEmbedder.wrappedValue = Flux2TimestepEmbedding(
        inputChannels: inChannels,
        timeEmbedDim: embeddingDim,
        bias: bias
      )
    }
  }

  func callAsFunction(_ timestep: MLXArray, guidance: MLXArray?) -> MLXArray {
    let timestepsProj = timeProj(timestep).asType(timestep.dtype)
    var timestepsEmb = timestepEmbedder(timestepsProj)

    if let guidance, let guidanceEmbedder {
      let guidanceProj = timeProj(guidance).asType(guidance.dtype)
      let guidanceEmb = guidanceEmbedder(guidanceProj)
      timestepsEmb = timestepsEmb + guidanceEmb
    }

    return timestepsEmb
  }
}
