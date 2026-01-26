import CoreGraphics
import Foundation
import MLX
import Hub

public enum Flux2DevPromptUpsamplerError: Error {
  case emptyPromptList
}

public final class Flux2DevPromptUpsampler {
  public let textEncoder: Flux2Mistral3TextEncoder
  public let processor: Flux2PixtralProcessor

  public init(
    textEncoder: Flux2Mistral3TextEncoder,
    processor: Flux2PixtralProcessor
  ) {
    self.textEncoder = textEncoder
    self.processor = processor
  }

  public convenience init(
    snapshot: URL,
    dtype: DType = .bfloat16,
    maxLength: Int = 2048,
    hubApi: HubApi = .shared
  ) throws {
    let processor = try Flux2PixtralProcessor.load(from: snapshot, maxLengthOverride: maxLength, hubApi: hubApi)
    let textEncoder = try Flux2Mistral3TextEncoder.load(from: snapshot, dtype: dtype)
    self.init(textEncoder: textEncoder, processor: processor)
  }

  public func upsample(
    prompts: [String],
    images: [CGImage]? = nil,
    temperature: Float = 0.15,
    maxNewTokens: Int = 512,
    seed: UInt64? = nil,
    prefillChunkSize: Int = 256,
    evaluationPolicy: Flux2EvaluationPolicy = .deferred,
    throwOnError: Bool = false,
    onError: ((Int, Error) -> Void)? = nil
  ) throws -> [String] {
    guard !prompts.isEmpty else {
      throw Flux2DevPromptUpsamplerError.emptyPromptList
    }

    let systemMessage: String
    if let images, !images.isEmpty {
      systemMessage = Flux2SystemMessages.upsamplingImageToImage
    } else {
      systemMessage = Flux2SystemMessages.upsamplingTextToImage
    }

    let imageBatch: Flux2PixtralImageBatch?
    if let images, !images.isEmpty {
      imageBatch = try processor.preprocessImages(images)
    } else {
      imageBatch = nil
    }

    let tokens: Flux2TokenBatch
    let imageTokenPositions: [[Int32]]?
    if let imageBatch {
      let tokenized = try processor.encodeMultimodalWithImageTokenPositions(
        prompts: prompts,
        systemMessage: systemMessage,
        imageSizes: imageBatch.imageSizes,
        maxLength: processor.maxLength,
        addGenerationPrompt: true
      )
      tokens = tokenized.tokens
      imageTokenPositions = tokenized.imageTokenPositions
    } else {
      tokens = try processor.encode(
        prompts: prompts,
        systemMessage: systemMessage,
        maxLength: processor.maxLength,
        addGenerationPrompt: true
      )
      imageTokenPositions = nil
    }

    var outputs: [String] = []
    outputs.reserveCapacity(prompts.count)

    for index in 0..<prompts.count {
      let prompt = prompts[index]
      let attention = tokens.attentionMask[index].asType(.int32)
      let promptLength = Int(attention.sum().item(Int32.self))

      if promptLength <= 0 {
        outputs.append(prompt)
        continue
      }

      let inputIds = tokens.inputIds[index, 0..<promptLength].reshaped(1, promptLength)
      let elementSeed = seed.map { $0 &+ UInt64(index) }

      do {
        let positions = imageTokenPositions?[index]
        let generated = try textEncoder.generateTokenIds(
          inputIds: inputIds,
          maxNewTokens: maxNewTokens,
          temperature: temperature,
          eosTokenId: processor.eosTokenId,
          pixelValues: imageBatch?.pixelValues,
          imageSizes: imageBatch?.imageSizes,
          imageTokenId: processor.multimodal?.imageTokenId,
          imageTokenPositions: positions,
          seed: elementSeed,
          prefillChunkSize: prefillChunkSize,
          evaluationPolicy: evaluationPolicy
        )
        let decoded = processor.decode(tokens: generated, skipSpecialTokens: true)
          .trimmingCharacters(in: .whitespacesAndNewlines)
        outputs.append(decoded.isEmpty ? prompt : decoded)
      } catch {
        onError?(index, error)
        if throwOnError {
          throw error
        }
        outputs.append(prompt)
      }
    }

    return outputs
  }
}
