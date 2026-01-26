import Foundation
import MLX
import Tokenizers
import Hub

public enum Flux2TokenizerError: Error {
  case directoryNotFound(URL)
  case fileNotFound(URL)
  case padTokenMissing
  case padTokenNotInVocabulary(String)
  case emptyPromptList
  case invalidMaxLength(Int)
}

public struct Flux2TokenBatch {
  public let inputIds: MLXArray
  public let attentionMask: MLXArray

  public init(inputIds: MLXArray, attentionMask: MLXArray) {
    self.inputIds = inputIds
    self.attentionMask = attentionMask
  }
}

public final class Flux2QwenTokenizer {
  private let tokenizer: Tokenizer
  private let padTokenId: Int
  private let chatTemplate: ChatTemplateArgument?
  public let maxLength: Int

  public init(
    tokenizer: Tokenizer,
    padTokenId: Int,
    maxLength: Int,
    chatTemplate: ChatTemplateArgument? = nil
  ) {
    self.tokenizer = tokenizer
    self.padTokenId = padTokenId
    self.maxLength = maxLength
    self.chatTemplate = chatTemplate
  }

  public static func load(
    from directory: URL,
    maxLengthOverride: Int? = nil,
    hubApi: HubApi = .shared
  ) throws -> Flux2QwenTokenizer {
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
      let tokenizerData = try makeBPETokenizerData(vocabURL: vocabURL, mergesURL: mergesURL)
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

    let resolvedMaxLength = maxLengthOverride ?? tokenizerConfig["model_max_length"].integer(or: 512)

    let chatTemplateURL = tokenizerDirectory.appending(path: "chat_template.jinja")
    let chatTemplate: ChatTemplateArgument?
    if FileManager.default.fileExists(atPath: chatTemplateURL.path) {
      let template = try String(contentsOf: chatTemplateURL, encoding: .utf8)
      chatTemplate = .literal(template)
    } else {
      chatTemplate = nil
    }

    return Flux2QwenTokenizer(
      tokenizer: tokenizer,
      padTokenId: padId,
      maxLength: resolvedMaxLength,
      chatTemplate: chatTemplate
    )
  }

  public func encode(
    prompts: [String],
    maxLength: Int? = nil,
    addGenerationPrompt: Bool = true,
    enableThinking: Bool = false
  ) throws -> Flux2TokenBatch {
    guard !prompts.isEmpty else {
      throw Flux2TokenizerError.emptyPromptList
    }
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    guard targetLength > 0 else {
      throw Flux2TokenizerError.invalidMaxLength(targetLength)
    }

    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      let messages: [Message] = [
        ["role": "user", "content": prompt]
      ]
      let tokens = try tokenizer.applyChatTemplate(
        messages: messages,
        chatTemplate: chatTemplate,
        addGenerationPrompt: addGenerationPrompt,
        truncation: true,
        maxLength: targetLength,
        tools: nil,
        additionalContext: ["enable_thinking": enableThinking]
      )

      let (padded, attention) = Self.padTokens(tokens: tokens, padTokenId: padTokenId, maxLength: targetLength)
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

  private static func padTokens(
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

  static func makeBPETokenizerData(vocabURL: URL, mergesURL: URL) throws -> Config {
    let vocabData = try Data(contentsOf: vocabURL)
    guard let vocabObject = try JSONSerialization.jsonObject(with: vocabData, options: []) as? [String: Any] else {
      throw Flux2TokenizerError.fileNotFound(vocabURL)
    }
    var vocab: [String: Int] = [:]
    vocab.reserveCapacity(vocabObject.count)
    for (k, v) in vocabObject {
      if let i = v as? Int { vocab[k] = i }
    }

    let tokenizerDir = vocabURL.deletingLastPathComponent()
    let addedTokensURL = tokenizerDir.appending(path: "added_tokens.json")
    var addedTokensMap: [String: Int] = [:]
    if FileManager.default.fileExists(atPath: addedTokensURL.path) {
      if let addedData = try? Data(contentsOf: addedTokensURL),
         let added = try? JSONSerialization.jsonObject(with: addedData, options: []) as? [String: Any] {
        for (k, v) in added {
          if let i = v as? Int {
            vocab[k] = i
            addedTokensMap[k] = i
          }
        }
      }
    }

    let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
    let merges: [String] = mergesText
      .components(separatedBy: .newlines)
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty && !$0.hasPrefix("#") }

    var tokenizerDict: [String: Any] = [
      "model": [
        "vocab": vocab,
        "merges": merges
      ],
      "preTokenizer": [
        "type": "ByteLevel",
        "addPrefixSpace": false,
        "trimOffsets": true,
        "useRegex": true
      ],
      "decoder": [
        "type": "ByteLevel"
      ]
    ]

    if !addedTokensMap.isEmpty {
      var addedList: [[String: Any]] = []
      addedList.reserveCapacity(addedTokensMap.count)
      for (tok, id) in addedTokensMap {
        addedList.append([
          "id": id,
          "content": tok,
          "lstrip": false,
          "rstrip": false,
          "special": true
        ])
      }
      tokenizerDict["addedTokens"] = addedList
    }
    let data = try JSONSerialization.data(withJSONObject: tokenizerDict, options: [])
    let tokenizerData = try JSONDecoder().decode(Config.self, from: data)
    return tokenizerData
  }
}
