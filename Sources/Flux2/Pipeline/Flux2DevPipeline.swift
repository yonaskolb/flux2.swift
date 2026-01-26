import Foundation
import MLX
import Hub

public struct Flux2DevPipelineOutput {
  public let packedLatents: MLXArray
  public let decoded: MLXArray
  public let promptEmbeds: MLXArray
  public let textIds: MLXArray
  public let latentIds: MLXArray
  public let imageLatents: MLXArray?
  public let imageLatentIds: MLXArray?
}

public enum Flux2DevPipelineError: Error {
  case promptEncoderReleased
  case missingProcessor
  case invalidLatentChannels(Int, Int)
  case invalidNumInferenceSteps(Int)
  case invalidImageCount(Int)
}

public final class Flux2DevPipeline {
  public let transformer: Flux2Transformer2DModel
  public let scheduler: FlowMatchEulerDiscreteScheduler
  public let vae: Flux2AutoencoderKL
  public private(set) var promptEncoder: Flux2DevPromptEncoder?

  private let pipeline: Flux2Pipeline

  public init(
    transformer: Flux2Transformer2DModel,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: Flux2AutoencoderKL,
    promptEncoder: Flux2DevPromptEncoder
  ) {
    self.transformer = transformer
    self.scheduler = scheduler
    self.vae = vae
    self.promptEncoder = promptEncoder
    self.pipeline = Flux2Pipeline(transformer: transformer, scheduler: scheduler, vae: vae)
  }

  public convenience init(
    snapshot: URL,
    dtype: DType = .bfloat16,
    hiddenStateLayers: [Int] = [10, 20, 30],
    maxLengthOverride: Int? = nil,
    systemMessage: String = Flux2SystemMessages.defaultMessage,
    hubApi: HubApi = .shared,
    loadProcessor: Bool = true
  ) throws {
    let transformer = try Flux2Transformer2DModel.load(from: snapshot, dtype: dtype)
    let scheduler = try FlowMatchEulerDiscreteScheduler.load(from: snapshot)
    let vae = try Flux2AutoencoderKL.load(from: snapshot, dtype: dtype)

    let promptEncoder: Flux2DevPromptEncoder
    if loadProcessor {
      promptEncoder = try Flux2DevPromptEncoder(
        snapshot: snapshot,
        dtype: dtype,
        hiddenStateLayers: hiddenStateLayers,
        maxLengthOverride: maxLengthOverride,
        systemMessage: systemMessage,
        hubApi: hubApi
      )
    } else {
      let encoder = try Flux2Mistral3TextEncoder.load(from: snapshot, dtype: dtype)
      promptEncoder = Flux2DevPromptEncoder(
        textEncoder: encoder,
        processor: nil,
        hiddenStateLayers: hiddenStateLayers,
        systemMessage: systemMessage
      )
    }

    self.init(transformer: transformer, scheduler: scheduler, vae: vae, promptEncoder: promptEncoder)
  }

  public func generate(
    prompts: [String],
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    numImagesPerPrompt: Int = 1,
    latents: MLXArray? = nil,
    guidanceScale: Float = 4.0,
    modelTimestepScale: Float = 0.001,
    images: [MLXArray]? = nil,
    imageIdScale: Int = 10,
    maxLength: Int? = nil
  ) throws -> Flux2DevPipelineOutput {
    guard let promptEncoder = promptEncoder else {
      throw Flux2DevPipelineError.promptEncoderReleased
    }
    guard promptEncoder.processor != nil else {
      throw Flux2DevPipelineError.missingProcessor
    }

    let encoding = try promptEncoder.encodePrompts(
      prompts,
      maxLength: maxLength,
      numImagesPerPrompt: numImagesPerPrompt
    )

    evalAndReleasePromptEncoder(promptEncoding: encoding)

    return try generate(
      promptEncoding: encoding,
      height: height,
      width: width,
      numInferenceSteps: numInferenceSteps,
      latents: latents,
      guidanceScale: guidanceScale,
      modelTimestepScale: modelTimestepScale,
      images: images,
      imageIdScale: imageIdScale
    )
  }

  public func generateTokens(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    numImagesPerPrompt: Int = 1,
    latents: MLXArray? = nil,
    guidanceScale: Float = 4.0,
    modelTimestepScale: Float = 0.001,
    images: [MLXArray]? = nil,
    imageIdScale: Int = 10
  ) throws -> Flux2DevPipelineOutput {
    guard let promptEncoder = promptEncoder else {
      throw Flux2DevPipelineError.promptEncoderReleased
    }

    let encoding = try promptEncoder.encodeTokens(
      inputIds: inputIds,
      attentionMask: attentionMask,
      numImagesPerPrompt: numImagesPerPrompt
    )

    evalAndReleasePromptEncoder(promptEncoding: encoding)

    return try generate(
      promptEncoding: encoding,
      height: height,
      width: width,
      numInferenceSteps: numInferenceSteps,
      latents: latents,
      guidanceScale: guidanceScale,
      modelTimestepScale: modelTimestepScale,
      images: images,
      imageIdScale: imageIdScale
    )
  }

  private func evalAndReleasePromptEncoder(
    promptEncoding: Flux2PromptEncoding
  ) {
    MLX.eval(promptEncoding.promptEmbeds, promptEncoding.textIds)
    promptEncoder = nil
  }

  private func generate(
    promptEncoding: Flux2PromptEncoding,
    height: Int,
    width: Int,
    numInferenceSteps: Int,
    latents: MLXArray?,
    guidanceScale: Float,
    modelTimestepScale: Float,
    images: [MLXArray]?,
    imageIdScale: Int
  ) throws -> Flux2DevPipelineOutput {
    guard numInferenceSteps > 0 else {
      throw Flux2DevPipelineError.invalidNumInferenceSteps(numInferenceSteps)
    }

    let patchArea = vae.configuration.patchSizeArea
    let inChannels = transformer.configuration.inChannels
    guard inChannels % patchArea == 0 else {
      throw Flux2DevPipelineError.invalidLatentChannels(inChannels, patchArea)
    }
    let numLatentChannels = inChannels / patchArea

    let batchSize = promptEncoding.promptEmbeds.dim(0)
    let prepared = try Flux2LatentPreparation.prepareLatents(
      batchSize: batchSize,
      numLatentChannels: numLatentChannels,
      height: height,
      width: width,
      vae: vae,
      dtype: promptEncoding.promptEmbeds.dtype,
      latents: latents
    )

    let preparedImages: Flux2PreparedImageLatents?
    if let images {
      if images.isEmpty {
        throw Flux2DevPipelineError.invalidImageCount(images.count)
      }
      preparedImages = try Flux2LatentPreparation.prepareImageLatents(
        images: images,
        batchSize: batchSize,
        vae: vae,
        dtype: promptEncoding.promptEmbeds.dtype,
        imageIdScale: imageIdScale
      )
    } else {
      preparedImages = nil
    }

    let sigmas = scheduler.flux2DefaultSigmas(numInferenceSteps: numInferenceSteps)
    let mu = scheduler.config.useDynamicShifting
      ? FlowMatchEulerDiscreteScheduler.flux2EmpiricalMu(
        imageSeqLen: prepared.latents.dim(1),
        numSteps: numInferenceSteps
      )
      : nil

    try scheduler.setTimesteps(numInferenceSteps: numInferenceSteps, sigmas: sigmas, mu: mu)
    scheduler.setBeginIndex(0)

    let guidance = MLXArray(Array(repeating: guidanceScale, count: batchSize)).asType(.float32)

    let denoised = try pipeline.denoiseLoop(
      latents: prepared.latents,
      encoderHiddenStates: promptEncoding.promptEmbeds,
      latentIds: prepared.ids,
      txtIds: promptEncoding.textIds,
      imageConditioning: preparedImages.map { (latents: $0.latents, ids: $0.ids) },
      guidance: guidance,
      modelTimestepScale: modelTimestepScale
    )

    let decoded = try pipeline.decodeLatents(denoised, latentIds: prepared.ids)

    var toEval: [MLXArray] = [
      denoised,
      decoded,
      promptEncoding.promptEmbeds,
      promptEncoding.textIds,
      prepared.ids,
    ]
    if let preparedImages {
      toEval.append(preparedImages.latents)
      toEval.append(preparedImages.ids)
    }
    MLX.eval(toEval)

    return Flux2DevPipelineOutput(
      packedLatents: denoised,
      decoded: decoded,
      promptEmbeds: promptEncoding.promptEmbeds,
      textIds: promptEncoding.textIds,
      latentIds: prepared.ids,
      imageLatents: preparedImages?.latents,
      imageLatentIds: preparedImages?.ids
    )
  }

}
