import Foundation
import MLX
import MLXNN

public struct Flux2PipelineOutput {
  public let packedLatents: MLXArray
  public let decoded: MLXArray
}

public final class Flux2Pipeline {
  public let transformer: Flux2Transformer2DModel
  public let scheduler: FlowMatchEulerDiscreteScheduler
  public let vae: Flux2AutoencoderKL

  private let denoiser: Flux2Denoiser

  public init(
    transformer: Flux2Transformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: Flux2AutoencoderKL
  ) {
    self.transformer = transformer
    self.scheduler = scheduler
    self.vae = vae
    self.denoiser = Flux2Denoiser(transformer: transformer, scheduler: scheduler)
  }

  public func denoiseLoop(
    latents: MLXArray,
    encoderHiddenStates: MLXArray,
    timestepValues: [Float]? = nil,
    latentIds: MLXArray,
    txtIds: MLXArray,
    imageConditioning: (latents: MLXArray, ids: MLXArray)? = nil,
    guidance: MLXArray? = nil,
    modelTimestepScale: Float = 0.001
  ) throws -> MLXArray {
    let stepValues = timestepValues ?? scheduler.timestepsValues
    let batch = latents.dim(0)

    var current = latents
    let combinedIds: MLXArray
    let conditioningImageLatents: MLXArray?

    if let imageConditioning {
      conditioningImageLatents = imageConditioning.latents
      combinedIds = MLX.concatenated([latentIds, imageConditioning.ids], axis: 1)
    } else {
      conditioningImageLatents = nil
      combinedIds = latentIds
    }

    for (stepIndex, step) in stepValues.enumerated() {
      let timestep = MLX.full([batch], values: step).asType(current.dtype)
      let output = try denoiser.step(
        latents: current,
        encoderHiddenStates: encoderHiddenStates,
        timestep: timestep,
        imgIds: combinedIds,
        txtIds: txtIds,
        imageLatents: conditioningImageLatents,
        guidance: guidance,
        modelTimestepScale: modelTimestepScale
      )
      current = output.prevLatents
    }
    return current
  }

  public func decodeLatents(
    _ packedLatents: MLXArray,
    latentIds: MLXArray
  ) throws -> MLXArray {
    let unpacked = try Flux2LatentUtils.unpackLatentsWithIds(packedLatents, ids: latentIds)
    let unpatchified = try Flux2LatentUtils.denormalizeAndUnpatchify(unpacked, vae: vae)
    return vae.decode(unpatchified)
  }

  public func generate(
    latents: MLXArray,
    encoderHiddenStates: MLXArray,
    timestepValues: [Float]? = nil,
    latentIds: MLXArray,
    txtIds: MLXArray,
    imageConditioning: (latents: MLXArray, ids: MLXArray)? = nil,
    guidance: MLXArray? = nil,
    modelTimestepScale: Float = 0.001
  ) throws -> Flux2PipelineOutput {
    let packed = try denoiseLoop(
      latents: latents,
      encoderHiddenStates: encoderHiddenStates,
      timestepValues: timestepValues,
      latentIds: latentIds,
      txtIds: txtIds,
      imageConditioning: imageConditioning,
      guidance: guidance,
      modelTimestepScale: modelTimestepScale
    )
    let decoded = try decodeLatents(packed, latentIds: latentIds)
    MLX.eval(packed, decoded)
    return Flux2PipelineOutput(packedLatents: packed, decoded: decoded)
  }
}
