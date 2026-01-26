import Foundation

public struct Flux2Mistral3VLMConfiguration: Codable, Sendable {
  public let modelType: String
  public let textConfig: Flux2Mistral3Configuration
  public let visionConfig: Flux2PixtralVisionConfiguration
  public let spatialMergeSize: Int
  public let imageTokenIndex: Int
  public let visionFeatureLayer: Flux2StringOrNumber
  public let projectorHiddenAct: String
  public let multimodalProjectorBias: Bool

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case textConfig = "text_config"
    case visionConfig = "vision_config"
    case spatialMergeSize = "spatial_merge_size"
    case imageTokenIndex = "image_token_index"
    case visionFeatureLayer = "vision_feature_layer"
    case projectorHiddenAct = "projector_hidden_act"
    case multimodalProjectorBias = "multimodal_projector_bias"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral3"
    textConfig = try container.decode(Flux2Mistral3Configuration.self, forKey: .textConfig)
    visionConfig = try container.decode(Flux2PixtralVisionConfiguration.self, forKey: .visionConfig)
    spatialMergeSize = try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 1
    imageTokenIndex = try container.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 0
    visionFeatureLayer = try container.decodeIfPresent(Flux2StringOrNumber.self, forKey: .visionFeatureLayer) ?? .int(-1)
    projectorHiddenAct = try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
    multimodalProjectorBias = try container.decodeIfPresent(Bool.self, forKey: .multimodalProjectorBias) ?? false
  }
}

