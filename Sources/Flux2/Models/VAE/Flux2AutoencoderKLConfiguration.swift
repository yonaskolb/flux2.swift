import Foundation

public struct Flux2AutoencoderKLConfiguration: Decodable, Sendable {
  public let inChannels: Int
  public let outChannels: Int
  public let downBlockTypes: [String]
  public let upBlockTypes: [String]
  public let blockOutChannels: [Int]
  public let layersPerBlock: Int
  public let actFn: String
  public let latentChannels: Int
  public let normNumGroups: Int
  public let sampleSize: [Int]
  public let forceUpcast: Bool
  public let useQuantConv: Bool
  public let usePostQuantConv: Bool
  public let midBlockAddAttention: Bool
  public let batchNormEps: Float
  public let batchNormMomentum: Float
  public let patchSize: (Int, Int)

  public var patchSizeArea: Int {
    patchSize.0 * patchSize.1
  }

  enum CodingKeys: String, CodingKey {
    case inChannels = "in_channels"
    case outChannels = "out_channels"
    case downBlockTypes = "down_block_types"
    case upBlockTypes = "up_block_types"
    case blockOutChannels = "block_out_channels"
    case layersPerBlock = "layers_per_block"
    case actFn = "act_fn"
    case latentChannels = "latent_channels"
    case normNumGroups = "norm_num_groups"
    case sampleSize = "sample_size"
    case forceUpcast = "force_upcast"
    case useQuantConv = "use_quant_conv"
    case usePostQuantConv = "use_post_quant_conv"
    case midBlockAddAttention = "mid_block_add_attention"
    case batchNormEps = "batch_norm_eps"
    case batchNormMomentum = "batch_norm_momentum"
    case patchSize = "patch_size"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
    outChannels = try container.decodeIfPresent(Int.self, forKey: .outChannels) ?? 3
    downBlockTypes = try container.decodeIfPresent([String].self, forKey: .downBlockTypes) ?? Array(
      repeating: "DownEncoderBlock2D",
      count: 4
    )
    upBlockTypes = try container.decodeIfPresent([String].self, forKey: .upBlockTypes) ?? Array(
      repeating: "UpDecoderBlock2D",
      count: 4
    )
    blockOutChannels = try container.decodeIfPresent([Int].self, forKey: .blockOutChannels)
      ?? [128, 256, 512, 512]
    layersPerBlock = try container.decodeIfPresent(Int.self, forKey: .layersPerBlock) ?? 2
    actFn = try container.decodeIfPresent(String.self, forKey: .actFn) ?? "silu"
    latentChannels = try container.decodeIfPresent(Int.self, forKey: .latentChannels) ?? 32
    normNumGroups = try container.decodeIfPresent(Int.self, forKey: .normNumGroups) ?? 32

    let sampleSizeValue = try container.decodeIfPresent(Flux2IntOrArray.self, forKey: .sampleSize)
    sampleSize = sampleSizeValue?.values ?? [1024]

    forceUpcast = try container.decodeIfPresent(Bool.self, forKey: .forceUpcast) ?? true
    useQuantConv = try container.decodeIfPresent(Bool.self, forKey: .useQuantConv) ?? true
    usePostQuantConv = try container.decodeIfPresent(Bool.self, forKey: .usePostQuantConv) ?? true
    midBlockAddAttention = try container.decodeIfPresent(Bool.self, forKey: .midBlockAddAttention) ?? true
    batchNormEps = try container.decodeIfPresent(Float.self, forKey: .batchNormEps) ?? 1e-4
    batchNormMomentum = try container.decodeIfPresent(Float.self, forKey: .batchNormMomentum) ?? 0.1

    let patchValue = try container.decodeIfPresent(Flux2IntOrArray.self, forKey: .patchSize)
    let patchValues = patchValue?.values ?? [2, 2]
    let patch0 = patchValues.first ?? 2
    let patch1 = patchValues.count > 1 ? patchValues[1] : patch0
    patchSize = (patch0, patch1)
  }
}

private enum Flux2IntOrArray: Decodable {
  case int(Int)
  case array([Int])

  init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if let value = try? container.decode(Int.self) {
      self = .int(value)
      return
    }
    if let value = try? container.decode([Int].self) {
      self = .array(value)
      return
    }
    throw DecodingError.typeMismatch(
      Flux2IntOrArray.self,
      DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected int or array")
    )
  }

  var values: [Int] {
    switch self {
    case .int(let value):
      return [value]
    case .array(let values):
      return values
    }
  }
}
