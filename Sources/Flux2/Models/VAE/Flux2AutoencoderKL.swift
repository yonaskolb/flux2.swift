import Foundation
import MLX
import MLXNN

public struct Flux2DiagonalGaussianDistribution {
  public let parameters: MLXArray
  public let mean: MLXArray
  public let logvar: MLXArray
  public let std: MLXArray
  public let variance: MLXArray
  public let deterministic: Bool

  public init(_ parameters: MLXArray, deterministic: Bool = false) {
    self.parameters = parameters
    let channels = parameters.dim(1) / 2
    let splits = split(parameters, indices: [channels], axis: 1)
    mean = splits[0]
    logvar = clip(splits[1], min: -30.0, max: 20.0)
    self.deterministic = deterministic

    if deterministic {
      std = MLXArray.zeros(like: mean)
      variance = MLXArray.zeros(like: mean)
    } else {
      std = exp(0.5 * logvar)
      variance = exp(logvar)
    }
  }

  public func sample() -> MLXArray {
    if deterministic {
      return mean
    }
    let noise = MLXRandom.normal(mean.shape, dtype: mean.dtype)
    return mean + std * noise
  }

  public func mode() -> MLXArray {
    mean
  }
}

public enum Flux2AutoencoderKLError: Error {
  case configNotFound(URL)
}

final class Flux2VAEEncoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "down_blocks") var downBlocks: [Flux2VAEDownBlock]
  @ModuleInfo(key: "mid_block") var midBlock: Flux2VAEMidBlock
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  init(configuration: Flux2AutoencoderKLConfiguration) {
    let channels = configuration.blockOutChannels
    let eps: Float = 1e-6

    _convIn.wrappedValue = Conv2d(
      inputChannels: configuration.inChannels,
      outputChannels: channels.first ?? 128,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    var downs: [Flux2VAEDownBlock] = []
    for (index, channel) in channels.enumerated() {
      let isLast = index == channels.count - 1
      let inCh = index == 0 ? channels.first ?? channel : channels[index - 1]
      downs.append(
        Flux2VAEDownBlock(
          inChannels: inCh,
          outChannels: channel,
          blockCount: configuration.layersPerBlock,
          hasDownsampler: !isLast,
          normGroups: configuration.normNumGroups,
          eps: eps
        )
      )
    }
    _downBlocks.wrappedValue = downs

    _midBlock.wrappedValue = Flux2VAEMidBlock(
      channels: channels.last ?? 512,
      normGroups: configuration.normNumGroups,
      addAttention: configuration.midBlockAddAttention,
      eps: eps
    )

    _convNormOut.wrappedValue = GroupNorm(
      groupCount: configuration.normNumGroups,
      dimensions: channels.last ?? 512,
      eps: eps,
      affine: true,
      pytorchCompatible: true
    )
    _convOut.wrappedValue = Conv2d(
      inputChannels: channels.last ?? 512,
      outputChannels: configuration.latentChannels * 2,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = convIn(x)
    for block in downBlocks {
      hidden = block(hidden)
    }
    hidden = midBlock(hidden)
    hidden = silu(convNormOut(hidden))
    hidden = convOut(hidden)
    return hidden
  }
}

final class Flux2VAEDecoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") var midBlock: Flux2VAEMidBlock
  @ModuleInfo(key: "up_blocks") var upBlocks: [Flux2VAEUpBlock]
  @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  init(configuration: Flux2AutoencoderKLConfiguration) {
    let channels = configuration.blockOutChannels
    let eps: Float = 1e-6

    _convIn.wrappedValue = Conv2d(
      inputChannels: configuration.latentChannels,
      outputChannels: channels.last ?? 512,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    _midBlock.wrappedValue = Flux2VAEMidBlock(
      channels: channels.last ?? 512,
      normGroups: configuration.normNumGroups,
      addAttention: configuration.midBlockAddAttention,
      eps: eps
    )

    var up: [Flux2VAEUpBlock] = []
    let reversed = channels.reversed()
    for (index, channel) in reversed.enumerated() {
      let isLast = index == channels.count - 1
      let prevOut = index == 0
        ? channel
        : reversed[reversed.index(reversed.startIndex, offsetBy: index - 1)]
      up.append(
        Flux2VAEUpBlock(
          inChannels: prevOut,
          outChannels: channel,
          blockCount: configuration.layersPerBlock + 1,
          hasUpsampler: !isLast,
          normGroups: configuration.normNumGroups,
          eps: eps
        )
      )
    }
    _upBlocks.wrappedValue = up

    _convNormOut.wrappedValue = GroupNorm(
      groupCount: configuration.normNumGroups,
      dimensions: channels.first ?? 128,
      eps: eps,
      affine: true,
      pytorchCompatible: true
    )
    _convOut.wrappedValue = Conv2d(
      inputChannels: channels.first ?? 128,
      outputChannels: configuration.outChannels,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    super.init()
  }

  func callAsFunction(_ latents: MLXArray) -> MLXArray {
    var hidden = convIn(latents)
    hidden = midBlock(hidden)
    for block in upBlocks {
      hidden = block(hidden)
    }
    hidden = silu(convNormOut(hidden))
    hidden = convOut(hidden)
    return hidden
  }
}

public final class Flux2AutoencoderKL: Module {
  public let configuration: Flux2AutoencoderKLConfiguration

  @ModuleInfo(key: "decoder") private var decoder: Flux2VAEDecoder
  @ModuleInfo(key: "encoder") private var encoder: Flux2VAEEncoder
  @ModuleInfo(key: "quant_conv") private var quantConv: Conv2d?
  @ModuleInfo(key: "post_quant_conv") private var postQuantConv: Conv2d?
  @ModuleInfo(key: "bn") private var batchNorm: BatchNorm

  public var bn: BatchNorm {
    batchNorm
  }

  public init(configuration: Flux2AutoencoderKLConfiguration) {
    precondition(configuration.actFn == "silu", "Only silu activation is supported.")
    precondition(
      configuration.downBlockTypes.allSatisfy { $0 == "DownEncoderBlock2D" },
      "Only DownEncoderBlock2D blocks are supported."
    )
    precondition(
      configuration.upBlockTypes.allSatisfy { $0 == "UpDecoderBlock2D" },
      "Only UpDecoderBlock2D blocks are supported."
    )

    self.configuration = configuration
    _decoder.wrappedValue = Flux2VAEDecoder(configuration: configuration)
    _encoder.wrappedValue = Flux2VAEEncoder(configuration: configuration)

    if configuration.useQuantConv {
      _quantConv.wrappedValue = Conv2d(
        inputChannels: configuration.latentChannels * 2,
        outputChannels: configuration.latentChannels * 2,
        kernelSize: 1,
        stride: 1,
        padding: 0
      )
    }

    if configuration.usePostQuantConv {
      _postQuantConv.wrappedValue = Conv2d(
        inputChannels: configuration.latentChannels,
        outputChannels: configuration.latentChannels,
        kernelSize: 1,
        stride: 1,
        padding: 0
      )
    }

    _batchNorm.wrappedValue = BatchNorm(
      featureCount: configuration.patchSizeArea * configuration.latentChannels,
      eps: configuration.batchNormEps,
      momentum: configuration.batchNormMomentum,
      affine: false,
      trackRunningStats: true
    )

    super.init()
  }

  public static func load(from snapshot: URL, dtype: DType = .bfloat16) throws -> Flux2AutoencoderKL {
    let configURL = snapshot
      .appendingPathComponent("vae")
      .appendingPathComponent("config.json")

    guard FileManager.default.fileExists(atPath: configURL.path) else {
      throw Flux2AutoencoderKLError.configNotFound(configURL)
    }

    let configData = try Data(contentsOf: configURL)
    let configuration = try JSONDecoder().decode(Flux2AutoencoderKLConfiguration.self, from: configData)
    let model = Flux2AutoencoderKL(configuration: configuration)

    let loader = Flux2WeightsLoader(snapshot: snapshot)
    let weights = try loader.load(component: .vae, dtype: dtype)
    let mapped = Flux2AutoencoderKL.mapWeights(weights)
    try model.update(parameters: ModuleParameters.unflattened(mapped), verify: .none)
    model.train(false)

    return model
  }

  public func encode(_ images: MLXArray) -> Flux2DiagonalGaussianDistribution {
    var hidden = images.transposed(0, 2, 3, 1)
    hidden = encoder(hidden)
    if let quantConv {
      hidden = quantConv(hidden)
    }
    hidden = hidden.transposed(0, 3, 1, 2)
    return Flux2DiagonalGaussianDistribution(hidden)
  }

  public func decode(_ latents: MLXArray) -> MLXArray {
    var hidden = latents.transposed(0, 2, 3, 1)
    if let postQuantConv {
      hidden = postQuantConv(hidden)
    }
    hidden = decoder(hidden)
    hidden = hidden.transposed(0, 3, 1, 2)
    return hidden
  }

  private static func mapWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    mapped.reserveCapacity(weights.count)
    for (key, value) in weights {
      var tensor = value
      if tensor.ndim == 4 {
        tensor = tensor.transposed(0, 2, 3, 1)
      }
      mapped[key] = tensor
    }
    return mapped
  }
}
