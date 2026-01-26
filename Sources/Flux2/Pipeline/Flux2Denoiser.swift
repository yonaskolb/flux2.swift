import Foundation
import MLX
import MLXNN

public struct Flux2DenoiserOutput {
  public let noisePred: MLXArray
  public let prevLatents: MLXArray
}

public final class Flux2Denoiser {
  public let transformer: Flux2Transformer2DModel
  public let scheduler: FlowMatchEulerDiscreteScheduler

  public init(
    transformer: Flux2Transformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler
  ) {
    self.transformer = transformer
    self.scheduler = scheduler
  }

  public func step(
    latents: MLXArray,
    encoderHiddenStates: MLXArray,
    timestep: MLXArray,
    imgIds: MLXArray,
    txtIds: MLXArray,
    imageLatents: MLXArray? = nil,
    guidance: MLXArray? = nil,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    modelTimestepScale: Float? = nil
  ) throws -> Flux2DenoiserOutput {
    var modelInput = latents
    if let imageLatents {
      modelInput = MLX.concatenated([latents, imageLatents], axis: 1)
    }

    var transformerTimestep = timestep
    if let modelTimestepScale {
      let scale = MLXArray(modelTimestepScale).asType(transformerTimestep.dtype)
      transformerTimestep = transformerTimestep * scale
    }

    let noisePredAll = transformer(
      modelInput,
      encoderHiddenStates: encoderHiddenStates,
      timestep: transformerTimestep,
      imgIds: imgIds,
      txtIds: txtIds,
      guidance: guidance,
      attentionMask: attentionMask
    )

    let tokenCount = latents.dim(1)
    let noisePred = noisePredAll[0..., 0..<tokenCount, 0...]

    let prevLatents = try scheduler.step(
      modelOutput: noisePred,
      timestep: timestep,
      sample: latents
    ).prevSample

    return Flux2DenoiserOutput(noisePred: noisePred, prevLatents: prevLatents)
  }
}
