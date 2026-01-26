import Foundation
import MLX

public struct Flux2ImageConditioningOutput {
  public let packedLatents: MLXArray
  public let decoded: MLXArray
  public let imageLatents: MLXArray
  public let imageLatentIds: MLXArray
}

public final class Flux2ImageConditioningPipeline {
  public let pipeline: Flux2Pipeline
  public let vae: Flux2AutoencoderKL

  public init(pipeline: Flux2Pipeline) {
    self.pipeline = pipeline
    self.vae = pipeline.vae
  }

  public func prepareImageLatents(
    images: [MLXArray],
    batchSize: Int,
    dtype: DType,
    imageIdScale: Int = 10
  ) throws -> Flux2PreparedImageLatents {
    return try Flux2LatentPreparation.prepareImageLatents(
      images: images,
      batchSize: batchSize,
      vae: vae,
      dtype: dtype,
      imageIdScale: imageIdScale
    )
  }

  public func generate(
    latents: MLXArray,
    encoderHiddenStates: MLXArray,
    timestepValues: [Float]? = nil,
    latentIds: MLXArray,
    txtIds: MLXArray,
    images: [MLXArray],
    batchSize: Int,
    dtype: DType,
    guidance: MLXArray? = nil,
    imageIdScale: Int = 10
  ) throws -> Flux2ImageConditioningOutput {
    let prepared = try prepareImageLatents(
      images: images,
      batchSize: batchSize,
      dtype: dtype,
      imageIdScale: imageIdScale
    )

    let output = try pipeline.generate(
      latents: latents,
      encoderHiddenStates: encoderHiddenStates,
      timestepValues: timestepValues,
      latentIds: latentIds,
      txtIds: txtIds,
      imageConditioning: (latents: prepared.latents, ids: prepared.ids),
      guidance: guidance
    )

    return Flux2ImageConditioningOutput(
      packedLatents: output.packedLatents,
      decoded: output.decoded,
      imageLatents: prepared.latents,
      imageLatentIds: prepared.ids
    )
  }
}
