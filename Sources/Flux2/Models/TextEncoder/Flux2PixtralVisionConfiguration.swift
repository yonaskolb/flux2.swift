import Foundation

public struct Flux2PixtralVisionConfiguration: Codable, Sendable {
  public let modelType: String
  public let hiddenSize: Int
  public let hiddenLayers: Int
  public let intermediateSize: Int
  public let attentionHeads: Int
  public let headDim: Int?
  public let numChannels: Int
  public let patchSize: Int
  public let imageSize: Int
  public let ropeTheta: Float
  public let hiddenAct: String?

  public var resolvedHeadDimensions: Int {
    headDim ?? (hiddenSize / max(attentionHeads, 1))
  }

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case hiddenSize = "hidden_size"
    case hiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case attentionHeads = "num_attention_heads"
    case headDim = "head_dim"
    case numChannels = "num_channels"
    case patchSize = "patch_size"
    case imageSize = "image_size"
    case ropeTheta = "rope_theta"
    case hiddenAct = "hidden_act"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "pixtral"
    hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
    hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
    intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
    attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
    headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
    numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
    patchSize = try container.decode(Int.self, forKey: .patchSize)
    imageSize = try container.decode(Int.self, forKey: .imageSize)
    ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
    hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct)
  }
}

