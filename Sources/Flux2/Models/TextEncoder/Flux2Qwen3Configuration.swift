import Foundation

public struct Flux2Qwen3Configuration: Codable {
  public let hiddenSize: Int
  public let hiddenLayers: Int
  public let intermediateSize: Int
  public let attentionHeads: Int
  public let rmsNormEps: Float
  public let vocabularySize: Int
  public let kvHeads: Int
  public let ropeTheta: Float
  public let headDim: Int
  public let ropeScaling: [String: Flux2StringOrNumber]?
  public let tieWordEmbeddings: Bool
  public let maxPositionEmbeddings: Int

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case hiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case attentionHeads = "num_attention_heads"
    case rmsNormEps = "rms_norm_eps"
    case vocabularySize = "vocab_size"
    case kvHeads = "num_key_value_heads"
    case ropeTheta = "rope_theta"
    case headDim = "head_dim"
    case ropeScaling = "rope_scaling"
    case tieWordEmbeddings = "tie_word_embeddings"
    case maxPositionEmbeddings = "max_position_embeddings"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
    hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
    intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
    attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
    rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
    vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
    kvHeads = try container.decode(Int.self, forKey: .kvHeads)
    ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
    headDim = try container.decode(Int.self, forKey: .headDim)
    ropeScaling = try container.decodeIfPresent([String: Flux2StringOrNumber].self, forKey: .ropeScaling)
    tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
  }
}
