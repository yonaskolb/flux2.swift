import Foundation
import MLX
import MLXNN

public enum Flux2Transformer2DModelError: Error {
  case configNotFound(URL)
}

public final class Flux2Transformer2DModel: Module {
  public let configuration: Flux2TransformerConfiguration
  public let outChannels: Int
  public let innerDim: Int

  private let posEmbed: Flux2PosEmbed

  @ModuleInfo(key: "time_guidance_embed") private var timeGuidanceEmbed: Flux2TimestepGuidanceEmbeddings
  @ModuleInfo(key: "double_stream_modulation_img") private var doubleStreamModulationImg: Flux2Modulation
  @ModuleInfo(key: "double_stream_modulation_txt") private var doubleStreamModulationTxt: Flux2Modulation
  @ModuleInfo(key: "single_stream_modulation") private var singleStreamModulation: Flux2Modulation

  @ModuleInfo(key: "x_embedder") private var xEmbedder: Linear
  @ModuleInfo(key: "context_embedder") private var contextEmbedder: Linear

  @ModuleInfo(key: "transformer_blocks") private var transformerBlocks: [Flux2TransformerBlock]
  @ModuleInfo(key: "single_transformer_blocks") private var singleTransformerBlocks: [Flux2SingleTransformerBlock]

  @ModuleInfo(key: "norm_out") private var normOut: Flux2AdaLayerNormContinuous
  @ModuleInfo(key: "proj_out") private var projOut: Linear

  public init(configuration: Flux2TransformerConfiguration) {
    self.configuration = configuration
    outChannels = configuration.resolvedOutChannels
    innerDim = configuration.innerDim
    let numHeads = configuration.numAttentionHeads
    let headDim = configuration.attentionHeadDim
    let mlpRatio = configuration.mlpRatio
    let eps = configuration.eps
    let numLayers = configuration.numLayers
    let numSingleLayers = configuration.numSingleLayers

    posEmbed = Flux2PosEmbed(theta: configuration.ropeTheta, axesDims: configuration.axesDimsRope)

    _timeGuidanceEmbed.wrappedValue = Flux2TimestepGuidanceEmbeddings(
      inChannels: configuration.timestepGuidanceChannels,
      embeddingDim: innerDim,
      bias: false,
      guidanceEmbeds: configuration.guidanceEmbeds
    )

    _doubleStreamModulationImg.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2, bias: false)
    _doubleStreamModulationTxt.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2, bias: false)
    _singleStreamModulation.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 1, bias: false)

    _xEmbedder.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _contextEmbedder.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)

    var blocks: [Flux2TransformerBlock] = []
    blocks.reserveCapacity(numLayers)
    for _ in 0..<numLayers {
      blocks.append(Flux2TransformerBlock(
        dim: innerDim,
        numAttentionHeads: numHeads,
        attentionHeadDim: headDim,
        mlpRatio: mlpRatio,
        eps: eps,
        bias: false
      ))
    }
    _transformerBlocks.wrappedValue = blocks

    var singleBlocks: [Flux2SingleTransformerBlock] = []
    singleBlocks.reserveCapacity(numSingleLayers)
    for _ in 0..<numSingleLayers {
      singleBlocks.append(Flux2SingleTransformerBlock(
        dim: innerDim,
        numAttentionHeads: numHeads,
        attentionHeadDim: headDim,
        mlpRatio: mlpRatio,
        eps: eps,
        bias: false
      ))
    }
    _singleTransformerBlocks.wrappedValue = singleBlocks

    _normOut.wrappedValue = Flux2AdaLayerNormContinuous(
      embeddingDim: innerDim,
      conditioningEmbeddingDim: innerDim,
      elementwiseAffine: false,
      eps: eps,
      bias: false
    )

    let projOutDim = configuration.patchSize * configuration.patchSize * outChannels
    _projOut.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)

    super.init()
  }

  public static func load(from snapshot: URL, dtype: DType = .bfloat16) throws -> Flux2Transformer2DModel {
    let configURL = snapshot
      .appendingPathComponent("transformer")
      .appendingPathComponent("config.json")

    guard FileManager.default.fileExists(atPath: configURL.path) else {
      throw Flux2Transformer2DModelError.configNotFound(configURL)
    }

    let configData = try Data(contentsOf: configURL)
    let configuration = try JSONDecoder().decode(Flux2TransformerConfiguration.self, from: configData)
    let model = Flux2Transformer2DModel(configuration: configuration)

    let loader = Flux2WeightsLoader(snapshot: snapshot)
    let weights = try loader.load(component: .transformer, dtype: dtype)
    if let manifest = try Flux2Quantizer.loadManifest(from: snapshot) {
      Flux2Quantizer.applyQuantization(to: model, manifest: manifest, weights: weights)
    }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)

    return model
  }

  public func callAsFunction(
    _ hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    timestep: MLXArray,
    imgIds: MLXArray,
    txtIds: MLXArray,
    guidance: MLXArray? = nil,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
  ) -> MLXArray {
    let numTxtTokens = encoderHiddenStates.dim(1)
    let targetDtype = hiddenStates.dtype
    let scale = MLXArray(1000.0).asType(targetDtype)

    let timestepScaled = timestep.asType(targetDtype) * scale
    var guidanceScaled: MLXArray? = nil
    if let guidance {
      guidanceScaled = guidance.asType(targetDtype) * scale
    }

    let temb = timeGuidanceEmbed(timestepScaled, guidance: guidanceScaled)

    let doubleStreamModImg = doubleStreamModulationImg(temb)
    let doubleStreamModTxt = doubleStreamModulationTxt(temb)
    let singleStreamMod = singleStreamModulation(temb)[0]

    var hidden = xEmbedder(hiddenStates)
    var encoder = contextEmbedder(encoderHiddenStates)

    let batch = hiddenStates.dim(0)
    let imgSeq = hiddenStates.dim(1)
    let txtSeq = encoderHiddenStates.dim(1)

    func validateIds(_ ids: MLXArray, name: String, expectedSeq: Int) {
      precondition(ids.ndim == 3, "\(name) must be [B, S, num_axes].")
      precondition(ids.dim(0) == batch, "\(name) must have B=\(batch); got \(ids.dim(0)).")
      precondition(ids.dim(1) == expectedSeq, "\(name) must have S=\(expectedSeq); got \(ids.dim(1)).")
    }

    validateIds(imgIds, name: "imgIds", expectedSeq: imgSeq)
    validateIds(txtIds, name: "txtIds", expectedSeq: txtSeq)

    let imageRotary = posEmbed(imgIds)
    let textRotary = posEmbed(txtIds)
    let concatRotary: Flux2RotaryEmbeddings = (
      cos: MLX.concatenated([textRotary.cos, imageRotary.cos], axis: 1),
      sin: MLX.concatenated([textRotary.sin, imageRotary.sin], axis: 1)
    )

    for block in transformerBlocks {
      let outputs = block(
        hiddenStates: hidden,
        encoderHiddenStates: encoder,
        tembModParamsImg: doubleStreamModImg,
        tembModParamsTxt: doubleStreamModTxt,
        imageRotaryEmb: concatRotary,
        attentionMask: attentionMask
      )
      encoder = outputs.encoderHiddenStates
      hidden = outputs.hiddenStates
    }

    hidden = MLX.concatenated([encoder, hidden], axis: 1)

    for block in singleTransformerBlocks {
      let outputs = block(
        hiddenStates: hidden,
        tembModParams: singleStreamMod,
        imageRotaryEmb: concatRotary,
        attentionMask: attentionMask
      )
      hidden = outputs.hiddenStates
    }

    let splitStates = split(hidden, indices: [numTxtTokens], axis: 1)
    hidden = splitStates[1]

    hidden = normOut(hidden, conditioningEmbedding: temb)
    return projOut(hidden)
  }
}
