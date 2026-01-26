import CoreGraphics
import Foundation
import MLX
import Tokenizers
import Hub

public enum Flux2PixtralProcessorError: Error {
  case multimodalConfigurationMissing
  case emptyPromptList
  case invalidMaxLength(Int)
  case multimodalTokenMissing(String)
  case tokenExpansionMismatch(expected: Int, actual: Int)
  case tokenExpansionOverflow(expanded: Int, maxLength: Int)
}

public struct Flux2PixtralMultimodalConfiguration {
  public let visionPatchSize: Int
  public let spatialMergeSize: Int
  public let imageTokenId: Int
  public let imageBreakTokenId: Int
  public let imageEndTokenId: Int
  public let imageProcessor: Flux2PixtralImageProcessor

  public var downsamplePatchSize: Int { max(visionPatchSize, 1) * max(spatialMergeSize, 1) }

  public init(
    visionPatchSize: Int,
    spatialMergeSize: Int,
    imageTokenId: Int,
    imageBreakTokenId: Int,
    imageEndTokenId: Int,
    imageProcessor: Flux2PixtralImageProcessor
  ) {
    self.visionPatchSize = visionPatchSize
    self.spatialMergeSize = spatialMergeSize
    self.imageTokenId = imageTokenId
    self.imageBreakTokenId = imageBreakTokenId
    self.imageEndTokenId = imageEndTokenId
    self.imageProcessor = imageProcessor
  }
}

public struct Flux2PixtralMultimodalTokenBatch {
  public let tokens: Flux2TokenBatch
  public let imageTokenPositions: [[Int32]]

  public init(tokens: Flux2TokenBatch, imageTokenPositions: [[Int32]]) {
    self.tokens = tokens
    self.imageTokenPositions = imageTokenPositions
  }
}

public final class Flux2PixtralProcessor {
  private let tokenizer: Tokenizer
  private let padTokenId: Int
  private let chatTemplate: ChatTemplateArgument?
  public let maxLength: Int
  public let multimodal: Flux2PixtralMultimodalConfiguration?

  public var eosTokenId: Int? {
    tokenizer.eosTokenId
  }

  public init(
    tokenizer: Tokenizer,
    padTokenId: Int,
    maxLength: Int,
    chatTemplate: ChatTemplateArgument? = nil,
    multimodal: Flux2PixtralMultimodalConfiguration? = nil
  ) {
    self.tokenizer = tokenizer
    self.padTokenId = padTokenId
    self.maxLength = maxLength
    self.chatTemplate = chatTemplate
    self.multimodal = multimodal
  }

  public func withImageLongestEdge(_ longestEdge: Int) throws -> Flux2PixtralProcessor {
    guard let multimodal else {
      throw Flux2PixtralProcessorError.multimodalConfigurationMissing
    }
    guard longestEdge > 0 else {
      throw Flux2PixtralImageProcessorError.invalidLongestEdge(longestEdge)
    }

    let existing = multimodal.imageProcessor.configuration
    let configuration = Flux2PixtralImageProcessorConfiguration(
      longestEdge: longestEdge,
      doResize: existing.doResize,
      doRescale: existing.doRescale,
      rescaleFactor: existing.rescaleFactor,
      doNormalize: existing.doNormalize,
      imageMean: existing.imageMean,
      imageStd: existing.imageStd
    )

    return Flux2PixtralProcessor(
      tokenizer: tokenizer,
      padTokenId: padTokenId,
      maxLength: maxLength,
      chatTemplate: chatTemplate,
      multimodal: Flux2PixtralMultimodalConfiguration(
        visionPatchSize: multimodal.visionPatchSize,
        spatialMergeSize: multimodal.spatialMergeSize,
        imageTokenId: multimodal.imageTokenId,
        imageBreakTokenId: multimodal.imageBreakTokenId,
        imageEndTokenId: multimodal.imageEndTokenId,
        imageProcessor: Flux2PixtralImageProcessor(configuration: configuration)
      )
    )
  }

  public static func load(
    from directory: URL,
    maxLengthOverride: Int? = nil,
    hubApi: HubApi = .shared
  ) throws -> Flux2PixtralProcessor {
    let tokenizerDirectory = resolveTokenizerDirectory(directory)
    let tokenizerConfigURL = tokenizerDirectory.appending(path: "tokenizer_config.json")
    let tokenizerDataURL = tokenizerDirectory.appending(path: "tokenizer.json")

    guard FileManager.default.fileExists(atPath: tokenizerDirectory.path) else {
      throw Flux2TokenizerError.directoryNotFound(tokenizerDirectory)
    }
    guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
      throw Flux2TokenizerError.fileNotFound(tokenizerConfigURL)
    }

    let tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
    let tokenizer: Tokenizer
    if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
      let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
      tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    } else {
      let vocabURL = tokenizerDirectory.appending(path: "vocab.json")
      let mergesURL = tokenizerDirectory.appending(path: "merges.txt")
      guard FileManager.default.fileExists(atPath: vocabURL.path),
            FileManager.default.fileExists(atPath: mergesURL.path) else {
        throw Flux2TokenizerError.fileNotFound(tokenizerDataURL)
      }
      let tokenizerData = try Flux2QwenTokenizer.makeBPETokenizerData(vocabURL: vocabURL, mergesURL: mergesURL)
      tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    let padTokenNode = tokenizerConfig["pad_token"]
    let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
    guard let padToken = padTokenString else {
      throw Flux2TokenizerError.padTokenMissing
    }

    guard let padId = tokenizer.convertTokenToId(padToken) ??
      tokenizer.eosTokenId ??
      tokenizer.bosTokenId
    else {
      throw Flux2TokenizerError.padTokenNotInVocabulary(padToken)
    }

    let configMaxLength = tokenizerConfig["model_max_length"].integer(or: 512)
    let resolvedMaxLength = maxLengthOverride ?? configMaxLength

    let chatTemplateURL = tokenizerDirectory.appending(path: "chat_template.jinja")
    let chatTemplate: ChatTemplateArgument?
    if FileManager.default.fileExists(atPath: chatTemplateURL.path) {
      let template = try String(contentsOf: chatTemplateURL, encoding: .utf8)
      chatTemplate = .literal(template)
    } else {
      chatTemplate = nil
    }

    let multimodal = try loadMultimodalConfiguration(tokenizer: tokenizer, tokenizerDirectory: tokenizerDirectory)

    return Flux2PixtralProcessor(
      tokenizer: tokenizer,
      padTokenId: padId,
      maxLength: resolvedMaxLength,
      chatTemplate: chatTemplate,
      multimodal: multimodal
    )
  }

  public func preprocessImages(_ images: [CGImage]) throws -> Flux2PixtralImageBatch {
    guard let multimodal else {
      throw Flux2PixtralProcessorError.multimodalConfigurationMissing
    }
    return try multimodal.imageProcessor.preprocess(images: images, patchSize: multimodal.downsamplePatchSize)
  }

  public func encode(
    prompts: [String],
    systemMessage: String,
    maxLength: Int? = nil,
    addGenerationPrompt: Bool = false
  ) throws -> Flux2TokenBatch {
    guard !prompts.isEmpty else {
      throw Flux2PixtralProcessorError.emptyPromptList
    }
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    guard targetLength > 0 else {
      throw Flux2PixtralProcessorError.invalidMaxLength(targetLength)
    }

    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      let cleaned = prompt.replacingOccurrences(of: "[IMG]", with: "")
      let messages: [Message] = [
        [
          "role": "system",
          "content": [
            ["type": "text", "text": systemMessage]
          ],
        ],
        [
          "role": "user",
          "content": [
            ["type": "text", "text": cleaned]
          ],
        ],
      ]

      let tokens = try tokenizer.applyChatTemplate(
        messages: messages,
        chatTemplate: chatTemplate,
        addGenerationPrompt: addGenerationPrompt,
        truncation: true,
        maxLength: targetLength,
        tools: nil,
        additionalContext: nil
      )

      let (padded, attention) = padTokens(tokens: tokens, padTokenId: padTokenId, maxLength: targetLength)
      inputSequences.append(padded)
      attentionSequences.append(attention)
    }

    let flatInput = inputSequences.flatMap { $0 }.map { Int32($0) }
    let flatAttention = attentionSequences.flatMap { $0 }.map { Int32($0) }
    let batch = prompts.count
    let shape = [batch, targetLength]

    let inputIds = MLXArray(flatInput, shape).asType(.int32)
    let attentionMask = MLXArray(flatAttention, shape).asType(.int32)
    return Flux2TokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  public func encodeMultimodal(
    prompts: [String],
    systemMessage: String,
    imageSizes: [(height: Int, width: Int)],
    maxLength: Int? = nil,
    addGenerationPrompt: Bool = false
  ) throws -> Flux2TokenBatch {
    return try encodeMultimodalWithImageTokenPositions(
      prompts: prompts,
      systemMessage: systemMessage,
      imageSizes: imageSizes,
      maxLength: maxLength,
      addGenerationPrompt: addGenerationPrompt
    ).tokens
  }

  public func encodeMultimodalWithImageTokenPositions(
    prompts: [String],
    systemMessage: String,
    imageSizes: [(height: Int, width: Int)],
    maxLength: Int? = nil,
    addGenerationPrompt: Bool = false
  ) throws -> Flux2PixtralMultimodalTokenBatch {
    guard let multimodal else {
      throw Flux2PixtralProcessorError.multimodalConfigurationMissing
    }

    guard !prompts.isEmpty else {
      throw Flux2PixtralProcessorError.emptyPromptList
    }
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    guard targetLength > 0 else {
      throw Flux2PixtralProcessorError.invalidMaxLength(targetLength)
    }

    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    var imageTokenPositions: [[Int32]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)
    imageTokenPositions.reserveCapacity(prompts.count)

    for prompt in prompts {
      let cleaned = prompt.replacingOccurrences(of: "[IMG]", with: "")
      var userBlocks: [[String: String]] = imageSizes.map { _ in ["type": "image"] }
      userBlocks.append(["type": "text", "text": cleaned])

      let messages: [Message] = [
        [
          "role": "system",
          "content": [
            ["type": "text", "text": systemMessage]
          ],
        ],
        [
          "role": "user",
          "content": userBlocks,
        ],
      ]

      let tokens = try tokenizer.applyChatTemplate(
        messages: messages,
        chatTemplate: chatTemplate,
        addGenerationPrompt: addGenerationPrompt,
        truncation: false,
        maxLength: nil,
        tools: nil,
        additionalContext: nil
      )

      let expanded = try Self.expandImageTokens(
        tokens: tokens,
        imageSizes: imageSizes,
        patchSize: multimodal.downsamplePatchSize,
        imageTokenId: multimodal.imageTokenId,
        imageBreakTokenId: multimodal.imageBreakTokenId,
        imageEndTokenId: multimodal.imageEndTokenId
      )

      if expanded.count > targetLength {
        throw Flux2PixtralProcessorError.tokenExpansionOverflow(expanded: expanded.count, maxLength: targetLength)
      }

      let positions = expanded.enumerated().compactMap { index, token in
        token == multimodal.imageTokenId ? Int32(index) : nil
      }

      let (padded, attention) = padTokens(tokens: expanded, padTokenId: padTokenId, maxLength: targetLength)
      inputSequences.append(padded)
      attentionSequences.append(attention)
      imageTokenPositions.append(positions)
    }

    let flatInput = inputSequences.flatMap { $0 }.map { Int32($0) }
    let flatAttention = attentionSequences.flatMap { $0 }.map { Int32($0) }
    let batch = prompts.count
    let shape = [batch, targetLength]

    let inputIds = MLXArray(flatInput, shape).asType(.int32)
    let attentionMask = MLXArray(flatAttention, shape).asType(.int32)
    return Flux2PixtralMultimodalTokenBatch(
      tokens: Flux2TokenBatch(inputIds: inputIds, attentionMask: attentionMask),
      imageTokenPositions: imageTokenPositions
    )
  }

  public func decode(tokens: [Int], skipSpecialTokens: Bool = true) -> String {
    tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
  }

  private func padTokens(
    tokens: [Int],
    padTokenId: Int,
    maxLength: Int
  ) -> ([Int], [Int]) {
    let truncated = Array(tokens.prefix(maxLength))
    let paddingCount = max(0, maxLength - truncated.count)

    var padded = truncated
    if paddingCount > 0 {
      padded.append(contentsOf: Array(repeating: padTokenId, count: paddingCount))
    }

    var attention = Array(repeating: 1, count: truncated.count)
    if paddingCount > 0 {
      attention.append(contentsOf: Array(repeating: 0, count: paddingCount))
    }

    return (padded, attention)
  }

  private static func resolveTokenizerDirectory(_ directory: URL) -> URL {
    let tokenizerPath = directory.appending(path: "tokenizer", directoryHint: .isDirectory)
    if FileManager.default.fileExists(atPath: tokenizerPath.path) {
      return tokenizerPath
    }
    return directory
  }

  private struct ProcessorConfig: Decodable {
    let patchSize: Int
    let spatialMergeSize: Int
    let imageToken: String
    let imageBreakToken: String
    let imageEndToken: String

    enum CodingKeys: String, CodingKey {
      case patchSize = "patch_size"
      case spatialMergeSize = "spatial_merge_size"
      case imageToken = "image_token"
      case imageBreakToken = "image_break_token"
      case imageEndToken = "image_end_token"
    }
  }

  private struct PreprocessorConfig: Decodable {
    struct Size: Decodable {
      let longestEdge: Int

      enum CodingKeys: String, CodingKey {
        case longestEdge = "longest_edge"
      }
    }

    let size: Size?
    let doResize: Bool?
    let doRescale: Bool?
    let rescaleFactor: Float?
    let doNormalize: Bool?
    let imageMean: [Float]?
    let imageStd: [Float]?

    enum CodingKeys: String, CodingKey {
      case size
      case doResize = "do_resize"
      case doRescale = "do_rescale"
      case rescaleFactor = "rescale_factor"
      case doNormalize = "do_normalize"
      case imageMean = "image_mean"
      case imageStd = "image_std"
    }
  }

  private static func loadMultimodalConfiguration(
    tokenizer: Tokenizer,
    tokenizerDirectory: URL
  ) throws -> Flux2PixtralMultimodalConfiguration? {
    let processorURL = tokenizerDirectory.appending(path: "processor_config.json")
    let preprocessorURL = tokenizerDirectory.appending(path: "preprocessor_config.json")

    guard FileManager.default.fileExists(atPath: processorURL.path) else {
      return nil
    }

    let processorData = try Data(contentsOf: processorURL)
    let processorConfig = try JSONDecoder().decode(ProcessorConfig.self, from: processorData)

    guard let imageTokenId = tokenizer.convertTokenToId(processorConfig.imageToken) else {
      throw Flux2PixtralProcessorError.multimodalTokenMissing(processorConfig.imageToken)
    }
    guard let imageBreakTokenId = tokenizer.convertTokenToId(processorConfig.imageBreakToken) else {
      throw Flux2PixtralProcessorError.multimodalTokenMissing(processorConfig.imageBreakToken)
    }
    guard let imageEndTokenId = tokenizer.convertTokenToId(processorConfig.imageEndToken) else {
      throw Flux2PixtralProcessorError.multimodalTokenMissing(processorConfig.imageEndToken)
    }

    var imageProcessorConfiguration = Flux2PixtralImageProcessorConfiguration()
    if FileManager.default.fileExists(atPath: preprocessorURL.path) {
      let preprocessorData = try Data(contentsOf: preprocessorURL)
      let preprocessorConfig = try JSONDecoder().decode(PreprocessorConfig.self, from: preprocessorData)

      imageProcessorConfiguration = Flux2PixtralImageProcessorConfiguration(
        longestEdge: preprocessorConfig.size?.longestEdge ?? imageProcessorConfiguration.longestEdge,
        doResize: preprocessorConfig.doResize ?? imageProcessorConfiguration.doResize,
        doRescale: preprocessorConfig.doRescale ?? imageProcessorConfiguration.doRescale,
        rescaleFactor: preprocessorConfig.rescaleFactor ?? imageProcessorConfiguration.rescaleFactor,
        doNormalize: preprocessorConfig.doNormalize ?? imageProcessorConfiguration.doNormalize,
        imageMean: preprocessorConfig.imageMean ?? imageProcessorConfiguration.imageMean,
        imageStd: preprocessorConfig.imageStd ?? imageProcessorConfiguration.imageStd
      )
    }

    return Flux2PixtralMultimodalConfiguration(
      visionPatchSize: processorConfig.patchSize,
      spatialMergeSize: processorConfig.spatialMergeSize,
      imageTokenId: imageTokenId,
      imageBreakTokenId: imageBreakTokenId,
      imageEndTokenId: imageEndTokenId,
      imageProcessor: Flux2PixtralImageProcessor(configuration: imageProcessorConfiguration)
    )
  }

  static func expandImageTokens(
    tokens: [Int],
    imageSizes: [(height: Int, width: Int)],
    patchSize: Int,
    imageTokenId: Int,
    imageBreakTokenId: Int,
    imageEndTokenId: Int
  ) throws -> [Int] {
    guard patchSize > 0 else {
      return tokens
    }

    var expanded: [Int] = []
    expanded.reserveCapacity(tokens.count)

    var imageIndex = 0

    for token in tokens {
      if token != imageTokenId {
        expanded.append(token)
        continue
      }

      guard imageIndex < imageSizes.count else {
        throw Flux2PixtralProcessorError.tokenExpansionMismatch(expected: imageSizes.count, actual: imageIndex)
      }

      let size = imageSizes[imageIndex]
      imageIndex += 1

      let heightTokens = max(size.height / patchSize, 1)
      let widthTokens = max(size.width / patchSize, 1)

      for row in 0..<heightTokens {
        expanded.append(contentsOf: Array(repeating: imageTokenId, count: widthTokens))
        expanded.append(row == heightTokens - 1 ? imageEndTokenId : imageBreakTokenId)
      }
    }

    if imageIndex != imageSizes.count {
      throw Flux2PixtralProcessorError.tokenExpansionMismatch(expected: imageSizes.count, actual: imageIndex)
    }

    return expanded
  }
}
