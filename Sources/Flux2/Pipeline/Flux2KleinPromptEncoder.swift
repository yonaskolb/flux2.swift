import Foundation
import MLX
import Hub

public struct Flux2PromptEncoding {
  public let promptEmbeds: MLXArray
  public let textIds: MLXArray
  public let inputIds: MLXArray
  public let attentionMask: MLXArray
}

public enum Flux2PromptEncoderError: Error {
  case tokenizerUnavailable
  case invalidRepeatCount(Int)
}

public final class Flux2KleinPromptEncoder {
  public let textEncoder: Flux2Qwen3TextEncoder
  public let tokenizer: Flux2QwenTokenizer?
  public let hiddenStateLayers: [Int]

  public init(
    textEncoder: Flux2Qwen3TextEncoder,
    tokenizer: Flux2QwenTokenizer? = nil,
    hiddenStateLayers: [Int] = [9, 18, 27]
  ) {
    self.textEncoder = textEncoder
    self.tokenizer = tokenizer
    self.hiddenStateLayers = hiddenStateLayers
  }

  public convenience init(
    snapshot: URL,
    dtype: DType = .bfloat16,
    hiddenStateLayers: [Int] = [9, 18, 27],
    maxLengthOverride: Int? = nil,
    hubApi: HubApi = .shared
  ) throws {
    let tokenizer = try Flux2QwenTokenizer.load(
      from: snapshot,
      maxLengthOverride: maxLengthOverride,
      hubApi: hubApi
    )
    let textEncoder = try Flux2Qwen3TextEncoder.load(from: snapshot, dtype: dtype)
    self.init(textEncoder: textEncoder, tokenizer: tokenizer, hiddenStateLayers: hiddenStateLayers)
  }

  public func encodePrompts(
    _ prompts: [String],
    maxLength: Int? = nil,
    addGenerationPrompt: Bool = true,
    enableThinking: Bool = false,
    numImagesPerPrompt: Int = 1,
    tCoord: MLXArray? = nil
  ) throws -> Flux2PromptEncoding {
    guard let tokenizer else {
      throw Flux2PromptEncoderError.tokenizerUnavailable
    }
    let tokens = try tokenizer.encode(
      prompts: prompts,
      maxLength: maxLength,
      addGenerationPrompt: addGenerationPrompt,
      enableThinking: enableThinking
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
