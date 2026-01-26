import Foundation

public struct Flux2Mistral3Configuration: Codable, Sendable {
  public let modelType: String
  public let hiddenSize: Int
  public let hiddenLayers: Int
  public let intermediateSize: Int
  public let attentionHeads: Int
  public let rmsNormEps: Float
  public let vocabularySize: Int
  public let headDim: Int?
  public let maxPositionEmbeddings: Int?
  public let kvHeads: Int
  public let ropeTheta: Float
  public let ropeParameters: [String: Flux2StringOrNumber]?
  public let tieWordEmbeddings: Bool
  public let layerTypes: [String]
  public let slidingWindow: Int?

  public var resolvedHeadDimensions: Int {
    headDim ?? (hiddenSize / max(attentionHeads, 1))
  }

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case hiddenSize = "hidden_size"
    case hiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case attentionHeads = "num_attention_heads"
    case rmsNormEps = "rms_norm_eps"
    case vocabularySize = "vocab_size"
    case headDim = "head_dim"
    case maxPositionEmbeddings = "max_position_embeddings"
    case kvHeads = "num_key_value_heads"
    case ropeTheta = "rope_theta"
    case ropeParameters = "rope_parameters"
    case tieWordEmbeddings = "tie_word_embeddings"
    case layerTypes = "layer_types"
    case slidingWindow = "sliding_window"
  }

  enum VLMCodingKeys: String, CodingKey {
    case textConfig = "text_config"
  }

  public init(from decoder: Decoder) throws {
    let topLevel = try decoder.container(keyedBy: CodingKeys.self)
    let vlm = try decoder.container(keyedBy: VLMCodingKeys.self)

    let textContainer: KeyedDecodingContainer<CodingKeys>
    if vlm.contains(.textConfig) {
      textContainer = try vlm.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
    } else {
      textContainer = topLevel
    }

    modelType = try topLevel.decodeIfPresent(String.self, forKey: .modelType)
      ?? textContainer.decodeIfPresent(String.self, forKey: .modelType)
      ?? "mistral3"

    hiddenSize = try textContainer.decode(Int.self, forKey: .hiddenSize)
    hiddenLayers = try textContainer.decode(Int.self, forKey: .hiddenLayers)
    intermediateSize = try textContainer.decode(Int.self, forKey: .intermediateSize)
    attentionHeads = try textContainer.decode(Int.self, forKey: .attentionHeads)
    rmsNormEps = try textContainer.decode(Float.self, forKey: .rmsNormEps)
    vocabularySize = try textContainer.decode(Int.self, forKey: .vocabularySize)

    headDim = try textContainer.decodeIfPresent(Int.self, forKey: .headDim)
    maxPositionEmbeddings = try textContainer.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
    slidingWindow = try textContainer.decodeIfPresent(Int.self, forKey: .slidingWindow)

    let decodedKvHeads = try textContainer.decodeIfPresent(Int.self, forKey: .kvHeads)
    kvHeads = decodedKvHeads ?? attentionHeads

    let decodedRopeParams = try topLevel.decodeIfPresent([String: Flux2StringOrNumber].self, forKey: .ropeParameters)
      ?? textContainer.decodeIfPresent([String: Flux2StringOrNumber].self, forKey: .ropeParameters)
    ropeParameters = decodedRopeParams

    let decodedRopeTheta = try textContainer.decodeIfPresent(Float.self, forKey: .ropeTheta)
    ropeTheta = decodedRopeParams?["rope_theta"]?.asFloat()
      ?? decodedRopeTheta
      ?? 10_000

    tieWordEmbeddings = try textContainer.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
      ?? topLevel.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
      ?? false

    layerTypes = try topLevel.decodeIfPresent([String].self, forKey: .layerTypes)
      ?? textContainer.decodeIfPresent([String].self, forKey: .layerTypes)
      ?? Array(repeating: "full_attention", count: hiddenLayers)
  }
}
