import Foundation

public struct Flux2TransformerConfiguration: Codable {
  public let patchSize: Int
  public let inChannels: Int
  public let outChannels: Int?
  public let numLayers: Int
  public let numSingleLayers: Int
  public let attentionHeadDim: Int
  public let numAttentionHeads: Int
  public let jointAttentionDim: Int
  public let timestepGuidanceChannels: Int
  public let mlpRatio: Float
  public let axesDimsRope: [Int]
  public let ropeTheta: Int
  public let eps: Float
  public let guidanceEmbeds: Bool

  enum CodingKeys: String, CodingKey {
    case patchSize = "patch_size"
    case inChannels = "in_channels"
    case outChannels = "out_channels"
    case numLayers = "num_layers"
    case numSingleLayers = "num_single_layers"
    case attentionHeadDim = "attention_head_dim"
    case numAttentionHeads = "num_attention_heads"
    case jointAttentionDim = "joint_attention_dim"
    case timestepGuidanceChannels = "timestep_guidance_channels"
    case mlpRatio = "mlp_ratio"
    case axesDimsRope = "axes_dims_rope"
    case ropeTheta = "rope_theta"
    case eps
    case guidanceEmbeds = "guidance_embeds"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 1
    inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 128
    outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels)
    numLayers = try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? 8
    numSingleLayers = try container.decodeIfPresent(Int.self, forKey: .numSingleLayers) ?? 48
    attentionHeadDim = try container.decodeIfPresent(Int.self, forKey: .attentionHeadDim) ?? 128
    numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 48
    jointAttentionDim = try container.decodeIfPresent(Int.self, forKey: .jointAttentionDim) ?? 15360
    timestepGuidanceChannels = try container.decodeIfPresent(Int.self, forKey: .timestepGuidanceChannels) ?? 256
    mlpRatio = try container.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 3.0
    axesDimsRope = try container.decodeIfPresent([Int].self, forKey: .axesDimsRope) ?? [32, 32, 32, 32]
    ropeTheta = try container.decodeIfPresent(Int.self, forKey: .ropeTheta) ?? 2000
    eps = try container.decodeIfPresent(Float.self, forKey: .eps) ?? 1e-6
    guidanceEmbeds = try container.decodeIfPresent(Bool.self, forKey: .guidanceEmbeds) ?? true
  }

  public var innerDim: Int {
    numAttentionHeads * attentionHeadDim
  }

  public var resolvedOutChannels: Int {
    outChannels ?? inChannels
  }
}
