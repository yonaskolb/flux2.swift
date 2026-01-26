import Foundation
import MLX
import Hub

public final class Flux2DevPromptEncoder {
  public let textEncoder: Flux2Mistral3TextEncoder
  public let processor: Flux2PixtralProcessor?
  public let hiddenStateLayers: [Int]
  public let systemMessage: String

  public init(
    textEncoder: Flux2Mistral3TextEncoder,
    processor: Flux2PixtralProcessor? = nil,
    hiddenStateLayers: [Int] = [10, 20, 30],
    systemMessage: String = Flux2SystemMessages.defaultMessage
  ) {
    self.textEncoder = textEncoder
    self.processor = processor
    self.hiddenStateLayers = hiddenStateLayers
    self.systemMessage = systemMessage
  }

  public convenience init(
    snapshot: URL,
    dtype: DType = .bfloat16,
    hiddenStateLayers: [Int] = [10, 20, 30],
    maxLengthOverride: Int? = nil,
    systemMessage: String = Flux2SystemMessages.defaultMessage,
    hubApi: HubApi = .shared
  ) throws {
    let processor = try Flux2PixtralProcessor.load(
      from: snapshot,
      maxLengthOverride: maxLengthOverride,
      hubApi: hubApi
    )
    let textEncoder = try Flux2Mistral3TextEncoder.load(from: snapshot, dtype: dtype)
    self.init(
      textEncoder: textEncoder,
      processor: processor,
      hiddenStateLayers: hiddenStateLayers,
      systemMessage: systemMessage
    )
  }

  public func encodePrompts(
    _ prompts: [String],
    maxLength: Int? = nil,
    addGenerationPrompt: Bool = false,
    numImagesPerPrompt: Int = 1,
    tCoord: MLXArray? = nil
  ) throws -> Flux2PromptEncoding {
    guard let processor else {
      throw Flux2PromptEncoderError.tokenizerUnavailable
    }
    let tokens = try processor.encode(
      prompts: prompts,
      systemMessage: systemMessage,
      maxLength: maxLength,
      addGenerationPrompt: addGenerationPrompt
    )
    return try encodeTokens(
      inputIds: tokens.inputIds,
      attentionMask: tokens.attentionMask,
      numImagesPerPrompt: numImagesPerPrompt,
      tCoord: tCoord
    )
  }

  public func encodeTokens(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    numImagesPerPrompt: Int = 1,
    tCoord: MLXArray? = nil
  ) throws -> Flux2PromptEncoding {
    guard numImagesPerPrompt >= 1 else {
      throw Flux2PromptEncoderError.invalidRepeatCount(numImagesPerPrompt)
    }

    var ids = inputIds
    if ids.dtype != .int32 {
      ids = ids.asType(.int32)
    }
    var mask = attentionMask
    if mask.dtype != .int32 {
      mask = mask.asType(.int32)
    }

    var promptEmbeds = try textEncoder.promptEmbeds(
      inputIds: ids,
      attentionMask: mask,
      hiddenStateLayers: hiddenStateLayers
    )

    var repeatedIds = ids
    var repeatedMask = mask
    var repeatedTCoord = tCoord

    if numImagesPerPrompt > 1 {
      promptEmbeds = repeatBatch(promptEmbeds, repeats: numImagesPerPrompt)
      repeatedIds = repeatBatch(ids, repeats: numImagesPerPrompt)
      repeatedMask = repeatBatch(mask, repeats: numImagesPerPrompt)
      if let tCoord {
        repeatedTCoord = repeatBatch(tCoord, repeats: numImagesPerPrompt)
      }
    }

    let textIds = try Flux2PositionIds.prepareTextIds(promptEmbeds, tCoord: repeatedTCoord)
    return Flux2PromptEncoding(
      promptEmbeds: promptEmbeds,
      textIds: textIds,
      inputIds: repeatedIds,
      attentionMask: repeatedMask
    )
  }

  private func repeatBatch(_ array: MLXArray, repeats: Int) -> MLXArray {
    let batch = array.dim(0)
    let tailShape = Array(array.shape.dropFirst())

    let expanded = array.expandedDimensions(axis: 1)
    var broadcastShape = [batch, repeats]
    broadcastShape.append(contentsOf: tailShape)

    let broadcasted = MLX.broadcast(expanded, to: broadcastShape)
    var reshaped = [batch * repeats]
    reshaped.append(contentsOf: tailShape)
    return broadcasted.reshaped(reshaped)
  }
}
