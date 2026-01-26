import Foundation
import MLX
import MLXNN

public enum Flux2Mistral3TextEncoderError: Error {
  case configNotFound(URL)
  case invalidInputShape
  case missingHiddenState(Int)
  case invalidGenerationInput
  case multimodalUnavailable
  case invalidMultimodalInput
  case imageTokenMismatch(expected: Int, actual: Int)
}

public final class Flux2Mistral3TextEncoder: Module {
  public let configuration: Flux2Mistral3Configuration
  let vlmConfiguration: Flux2Mistral3VLMConfiguration?

  @ModuleInfo(key: "language_model") private var languageModel: Flux2Mistral3CausalLM
  @ModuleInfo(key: "vision_tower") private var visionTower: Flux2PixtralVisionTower?
  @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector: Flux2Mistral3MultiModalProjector?

  public init(configuration: Flux2Mistral3Configuration, vlmConfiguration: Flux2Mistral3VLMConfiguration? = nil) {
    self.configuration = configuration
    self.vlmConfiguration = vlmConfiguration
    _languageModel.wrappedValue = Flux2Mistral3CausalLM(configuration)
    if let vlmConfiguration {
      _visionTower.wrappedValue = Flux2PixtralVisionTower(configuration: vlmConfiguration.visionConfig)
      _multiModalProjector.wrappedValue = Flux2Mistral3MultiModalProjector(vlmConfiguration: vlmConfiguration)
    } else {
      _visionTower.wrappedValue = nil
      _multiModalProjector.wrappedValue = nil
    }
    super.init()
  }

  public static func load(from snapshot: URL, dtype: DType = .bfloat16) throws -> Flux2Mistral3TextEncoder {
    let configURL = snapshot
      .appendingPathComponent("text_encoder")
      .appendingPathComponent("config.json")

    guard FileManager.default.fileExists(atPath: configURL.path) else {
      throw Flux2Mistral3TextEncoderError.configNotFound(configURL)
    }

    let configData = try Data(contentsOf: configURL)
    let vlmConfiguration = try? JSONDecoder().decode(Flux2Mistral3VLMConfiguration.self, from: configData)
    let configuration: Flux2Mistral3Configuration
    if let vlmConfiguration {
      configuration = vlmConfiguration.textConfig
    } else {
      configuration = try JSONDecoder().decode(Flux2Mistral3Configuration.self, from: configData)
    }
    let encoder = Flux2Mistral3TextEncoder(configuration: configuration, vlmConfiguration: vlmConfiguration)

    let loader = Flux2WeightsLoader(snapshot: snapshot)
    let weights = try loader.load(component: .textEncoder, dtype: dtype) { name in
      name.hasPrefix("language_model.model.") ||
        name.hasPrefix("language_model.lm_head.") ||
        name.hasPrefix("vision_tower.") ||
        name.hasPrefix("multi_modal_projector.") ||
        name.hasPrefix("model.language_model.") ||
        name.hasPrefix("model.vision_tower.") ||
        name.hasPrefix("model.multi_modal_projector.")
    }

    let normalized = normalizeTextEncoderWeights(weights)
    if let manifest = try Flux2Quantizer.loadManifest(from: snapshot) {
      Flux2Quantizer.applyQuantization(to: encoder, manifest: manifest, weights: normalized)
    }
    try encoder.update(parameters: ModuleParameters.unflattened(normalized), verify: .none)

    return encoder
  }

  public func generateTokenIds(
    inputIds: MLXArray,
    maxNewTokens: Int = 512,
    temperature: Float = 0.15,
    eosTokenId: Int? = nil,
    pixelValues: MLXArray? = nil,
    imageSizes: [(height: Int, width: Int)]? = nil,
    imageTokenId: Int? = nil,
    imageTokenPositions: [Int32]? = nil,
    seed: UInt64? = nil,
    prefillChunkSize: Int = 256,
    evaluationPolicy: Flux2EvaluationPolicy = .deferred
  ) throws -> [Int] {
    guard languageModel.isGenerationReady else {
      throw Flux2Mistral3TextEncoderError.invalidGenerationInput
    }
    guard inputIds.ndim == 2 else {
      throw Flux2Mistral3TextEncoderError.invalidGenerationInput
    }
    guard maxNewTokens >= 0, temperature > 0 else {
      throw Flux2Mistral3TextEncoderError.invalidGenerationInput
    }

    var ids = inputIds
    if ids.dtype != .int32 {
      ids = ids.asType(.int32)
    }

    let batch = ids.dim(0)
    guard batch == 1 else {
      throw Flux2Mistral3TextEncoderError.invalidGenerationInput
    }

    let seqLen = ids.dim(1)
    guard seqLen > 0 else {
      throw Flux2Mistral3TextEncoderError.invalidGenerationInput
    }

    let chunkSize = max(prefillChunkSize, 1)
    let caches = (0..<configuration.hiddenLayers).map { _ in Flux2KVCache(step: chunkSize) }

    var inputEmbeds: MLXArray? = nil
    if let pixelValues {
      guard let imageSizes, let imageTokenId else {
        throw Flux2Mistral3TextEncoderError.invalidMultimodalInput
      }

      let imageFeatures = try getImageFeatures(
        pixelValues: pixelValues,
        imageSizes: imageSizes
      )

      let positions: [Int32]
      if let imageTokenPositions {
        positions = imageTokenPositions
      } else {
        let tokenIds = ids.asArray(Int32.self)
        var resolved = [Int32]()
        resolved.reserveCapacity(tokenIds.count)
        for (index, token) in tokenIds.enumerated() where Int(token) == imageTokenId {
          resolved.append(Int32(index))
        }
        positions = resolved
      }

      let expected = imageFeatures.dim(0)
      if positions.count != expected {
        throw Flux2Mistral3TextEncoderError.imageTokenMismatch(expected: expected, actual: positions.count)
      }

      var embeds = languageModel.tokenEmbeddings(inputIds: ids)
      let positionArray = MLXArray(positions, [positions.count]).asType(.int32)
      let typedFeatures = imageFeatures.asType(embeds.dtype)
      embeds[0, positionArray] = typedFeatures
      inputEmbeds = embeds
    }

    var logits: MLXArray? = nil
    var position = 0
    while position < seqLen {
      let end = min(seqLen, position + chunkSize)
      let chunk = ids[.ellipsis, position ..< end]
      let embedChunk: MLXArray?
      if let inputEmbeds {
        embedChunk = inputEmbeds[.ellipsis, position ..< end, 0...]
      } else {
        embedChunk = nil
      }

      logits = languageModel.lastLogits(
        inputIds: chunk,
        inputsEmbeds: embedChunk,
        caches: caches,
        evaluationPolicy: evaluationPolicy
      )
      position = end
    }

    guard var currentLogits = logits else {
      throw Flux2Mistral3TextEncoderError.invalidGenerationInput
    }

    let rng = seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()

    var output: [Int] = []
    output.reserveCapacity(maxNewTokens)

    for _ in 0..<maxNewTokens {
      var logitsFP32 = currentLogits.asType(.float32)
      if temperature != 1.0 {
        logitsFP32 = logitsFP32 / MLXArray(temperature)
      }

      let sampled = MLXRandom.categorical(logitsFP32, key: rng).asType(.int32)
      let tokenId = Int(sampled.item(Int32.self))
      output.append(tokenId)

      if let eosTokenId, tokenId == eosTokenId {
        break
      }

      let nextInput = MLXArray([Int32(tokenId)], [1, 1])
      currentLogits = languageModel.lastLogits(
        inputIds: nextInput,
        inputsEmbeds: nil,
        caches: caches,
        evaluationPolicy: evaluationPolicy
      )
      evaluationPolicy.evalIfNeeded(currentLogits)
    }

    return output
  }

  private func getImageFeatures(
    pixelValues: MLXArray,
    imageSizes: [(height: Int, width: Int)]
  ) throws -> MLXArray {
    guard let vlmConfiguration,
          let visionTower,
          let multiModalProjector
    else {
      throw Flux2Mistral3TextEncoderError.multimodalUnavailable
    }

    let visionLayers: [Int]
    switch vlmConfiguration.visionFeatureLayer {
    case .int(let value):
      visionLayers = [value]
    case .ints(let values):
      visionLayers = values
    default:
      visionLayers = [-1]
    }

    let hiddenStates = visionTower.selectedHiddenStates(
      pixelValues: pixelValues,
      imageSizes: imageSizes,
      featureLayers: visionLayers
    )

    let selected = hiddenStates[0]
    var projected = multiModalProjector(selected, imageSizes: imageSizes)

    return projected
  }

  public func promptEmbeds(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    hiddenStateLayers: [Int] = [10, 20, 30],
    evaluationPolicy: Flux2EvaluationPolicy = .deferred
  ) throws -> MLXArray {
    guard inputIds.ndim == 2, attentionMask.ndim == 2 else {
      throw Flux2Mistral3TextEncoderError.invalidInputShape
    }
    guard inputIds.shape == attentionMask.shape else {
      throw Flux2Mistral3TextEncoderError.invalidInputShape
    }

    let hiddenStates = languageModel.hiddenStates(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputLayerIndices: hiddenStateLayers,
      evaluationPolicy: evaluationPolicy
    )

    var selected: [MLXArray] = []
    selected.reserveCapacity(hiddenStateLayers.count)
    for index in hiddenStateLayers {
      guard let state = hiddenStates[index] else {
        throw Flux2Mistral3TextEncoderError.missingHiddenState(index)
      }
      selected.append(state)
    }

    let stackedStates = stacked(selected, axis: 1)
    let permuted = stackedStates.transposed(0, 2, 1, 3)
    let batch = permuted.dim(0)
    let seqLen = permuted.dim(1)
    let embeds = permuted.reshaped(batch, seqLen, hiddenStateLayers.count * configuration.hiddenSize)
    return embeds
  }
}

private func normalizeTextEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
  var normalized: [String: MLXArray] = [:]
  normalized.reserveCapacity(weights.count)

  for (name, value) in weights {
    var tensor = value
    let key: String
    if name.hasPrefix("model.language_model.model.") {
      let suffix = name.dropFirst("model.language_model.model.".count)
      key = "language_model.model.\(suffix)"
    } else if name.hasPrefix("model.language_model.lm_head.") {
      let suffix = name.dropFirst("model.language_model.lm_head.".count)
      key = "language_model.lm_head.\(suffix)"
    } else if name.hasPrefix("model.language_model.") {
      let suffix = name.dropFirst("model.language_model.".count)
      key = "language_model.model.\(suffix)"
    } else if name.hasPrefix("model.vision_tower.") {
      let suffix = name.dropFirst("model.vision_tower.".count)
      key = "vision_tower.\(suffix)"
    } else if name.hasPrefix("model.multi_modal_projector.") {
      let suffix = name.dropFirst("model.multi_modal_projector.".count)
      key = "multi_modal_projector.\(suffix)"
    } else if name.hasPrefix("model.") {
      let suffix = name.dropFirst("model.".count)
      key = String(suffix)
    } else {
      key = name
    }

    if key == "vision_tower.patch_conv.weight", tensor.ndim == 4 {
      tensor = tensor.transposed(0, 2, 3, 1)
    }

    normalized[key] = tensor
  }

  return normalized
}

private enum Flux2Mistral3AttentionMask {
  static func make(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    slidingWindow: Int? = nil
  ) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let seqLen = hiddenStates.dim(1)
    if seqLen <= 1 {
      return .none
    }

    let dtype = hiddenStates.dtype
    let negInf = MLXArray(-Float.infinity).asType(dtype)

    var additivePadding: MLXArray? = nil
    if let attentionMask {
      precondition(attentionMask.ndim == 2, "attentionMask must be [batch, seq_len].")
      precondition(attentionMask.dim(1) == seqLen, "attentionMask must match sequence length.")

      let mask = attentionMask.asType(dtype)
      let zeros = MLX.zeros(mask.shape, dtype: dtype)
      let keepMask = mask .== MLXArray(1).asType(dtype)
      additivePadding = MLX.where(keepMask, zeros, zeros + negInf).reshaped(mask.dim(0), 1, 1, seqLen)
    }

    let causalAdditive = Flux2AttentionMaskCache.causalAdditive(
      seqLen: seqLen,
      dtype: dtype,
      slidingWindow: slidingWindow
    )

    if let additivePadding {
      return .array(causalAdditive + additivePadding)
    }
    return .array(causalAdditive)
  }
}

private func llama4AttentionScale(
  sequenceLength: Int,
  offset: Int,
  beta: Float?,
  maxPositionEmbeddings: Int?,
  dtype: DType
) -> MLXArray {
  guard let beta, let maxPositionEmbeddings, sequenceLength > 0 else {
    return MLX.ones([max(sequenceLength, 1), 1], dtype: dtype)
  }

  let start = Int32(max(offset, 0))
  let end = start + Int32(sequenceLength)
  let positions = MLXArray(start..<end).asType(.float32)
  let divisor = MLXArray(Float(maxPositionEmbeddings))
  let scaling = MLXArray(1.0) + MLXArray(beta) * MLX.log(MLXArray(1.0) + MLX.floor(positions / divisor))
  return scaling.asType(dtype).reshaped(sequenceLength, 1)
}

private final class Flux2Mistral3CausalLM: Module {
  private let configuration: Flux2Mistral3Configuration
  @ModuleInfo(key: "model") private var model: Flux2Mistral3LanguageModel
  @ModuleInfo(key: "lm_head") private var lmHead: Linear

  init(_ configuration: Flux2Mistral3Configuration) {
    self.configuration = configuration
    _model.wrappedValue = Flux2Mistral3LanguageModel(configuration)
    _lmHead.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    super.init()
  }

  var isGenerationReady: Bool {
    lmHead.weight.ndim == 2 &&
      lmHead.weight.dim(0) == configuration.vocabularySize &&
      lmHead.weight.dim(1) == configuration.hiddenSize
  }

  func hiddenStates(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    outputLayerIndices: [Int],
    evaluationPolicy: Flux2EvaluationPolicy
  ) -> [Int: MLXArray] {
    model.hiddenStates(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputLayerIndices: outputLayerIndices,
      evaluationPolicy: evaluationPolicy
    )
  }

  func tokenEmbeddings(inputIds: MLXArray) -> MLXArray {
    model.tokenEmbeddings(inputIds: inputIds)
  }

  func lastLogits(
    inputIds: MLXArray,
    inputsEmbeds: MLXArray?,
    caches: [Flux2KVCache],
    evaluationPolicy: Flux2EvaluationPolicy
  ) -> MLXArray {
    let hidden = model.forwardWithCache(
      inputIds: inputIds,
      inputsEmbeds: inputsEmbeds,
      caches: caches,
      evaluationPolicy: evaluationPolicy
    )
    let lastHidden = hidden[.ellipsis, -1, 0...]
    let logits = lmHead(lastHidden)
    evaluationPolicy.evalIfNeeded(logits)
    return logits
  }

  func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
    model(inputIds, attentionMask: attentionMask)
  }
}

private final class Flux2Mistral3LanguageModel: Module {
  @ModuleInfo(key: "embed_tokens") private var embedTokens: Embedding

  private let layers: [Flux2Mistral3TransformerBlock]
  private let norm: RMSNorm
  private let configuration: Flux2Mistral3Configuration

  init(_ configuration: Flux2Mistral3Configuration) {
    self.configuration = configuration
    _embedTokens.wrappedValue = Flux2ModulePlaceholders.embedding()

    let layerTypes = configuration.layerTypes.count == configuration.hiddenLayers
      ? configuration.layerTypes
      : Array(repeating: "full_attention", count: configuration.hiddenLayers)

    layers = layerTypes.map { Flux2Mistral3TransformerBlock(configuration, useSliding: $0 == "sliding_attention") }
    norm = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  func hiddenStates(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    outputLayerIndices: [Int],
    evaluationPolicy: Flux2EvaluationPolicy
  ) -> [Int: MLXArray] {
    let wanted = Set(outputLayerIndices)
    var results: [Int: MLXArray] = [:]

    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    var hidden = embedTokens(tokenIds)
    evaluationPolicy.evalIfNeeded(hidden)
    if wanted.contains(0) {
      results[0] = hidden
    }

    let fullMask = Flux2Mistral3AttentionMask.make(hiddenStates: hidden, attentionMask: attentionMask, slidingWindow: nil)
    let slidingMask = Flux2Mistral3AttentionMask.make(
      hiddenStates: hidden,
      attentionMask: attentionMask,
      slidingWindow: configuration.slidingWindow
    )

    let beta = configuration.ropeParameters?["llama_4_scaling_beta"]?.asFloat()
    let originalMax = configuration.ropeParameters?["original_max_position_embeddings"]?.asInt()
    let attnScale = llama4AttentionScale(
      sequenceLength: hidden.dim(1),
      offset: 0,
      beta: beta,
      maxPositionEmbeddings: originalMax,
      dtype: hidden.dtype
    )

    for (index, layer) in layers.enumerated() {
      let mask = layer.useSliding ? slidingMask : fullMask
      hidden = layer(hidden, attentionMask: mask, attnScale: attnScale)
      evaluationPolicy.evalIfNeeded(hidden)
      let hiddenIndex = index + 1
      if wanted.contains(hiddenIndex) {
        results[hiddenIndex] = hidden
      }
    }

    let finalIndex = configuration.hiddenLayers
    if wanted.contains(finalIndex) {
      let normalized = norm(hidden)
      evaluationPolicy.evalIfNeeded(normalized)
      results[finalIndex] = normalized
    }

    return results
  }

  func tokenEmbeddings(inputIds: MLXArray) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    return embedTokens(tokenIds)
  }

  func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    var hidden = embedTokens(tokenIds)
    let fullMask = Flux2Mistral3AttentionMask.make(hiddenStates: hidden, attentionMask: attentionMask, slidingWindow: nil)
    let slidingMask = Flux2Mistral3AttentionMask.make(
      hiddenStates: hidden,
      attentionMask: attentionMask,
      slidingWindow: configuration.slidingWindow
    )

    let beta = configuration.ropeParameters?["llama_4_scaling_beta"]?.asFloat()
    let originalMax = configuration.ropeParameters?["original_max_position_embeddings"]?.asInt()
    let attnScale = llama4AttentionScale(
      sequenceLength: hidden.dim(1),
      offset: 0,
      beta: beta,
      maxPositionEmbeddings: originalMax,
      dtype: hidden.dtype
    )

    for layer in layers {
      let mask = layer.useSliding ? slidingMask : fullMask
      hidden = layer(hidden, attentionMask: mask, attnScale: attnScale)
    }
    let normalized = norm(hidden)
    return normalized
  }

  func forwardWithCache(
    inputIds: MLXArray,
    inputsEmbeds: MLXArray?,
    caches: [Flux2KVCache],
    evaluationPolicy: Flux2EvaluationPolicy
  ) -> MLXArray {
    precondition(caches.count == layers.count, "caches must match hidden layer count.")

    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    let offset = caches.first?.offset ?? 0

    var hidden: MLXArray
    if let inputsEmbeds {
      hidden = inputsEmbeds
    } else {
      hidden = embedTokens(tokenIds)
    }
    evaluationPolicy.evalIfNeeded(hidden)

    let beta = configuration.ropeParameters?["llama_4_scaling_beta"]?.asFloat()
    let originalMax = configuration.ropeParameters?["original_max_position_embeddings"]?.asInt()
    let attnScale = llama4AttentionScale(
      sequenceLength: hidden.dim(1),
      offset: offset,
      beta: beta,
      maxPositionEmbeddings: originalMax,
      dtype: hidden.dtype
    )

    let mask: MLXFast.ScaledDotProductAttentionMaskMode = hidden.dim(1) <= 1 ? .none : .causal

    for (index, layer) in layers.enumerated() {
      hidden = layer(hidden, cache: caches[index], attentionMask: mask, attnScale: attnScale)
      evaluationPolicy.evalIfNeeded(hidden)
    }

    let normalized = norm(hidden)
    evaluationPolicy.evalIfNeeded(normalized)
    return normalized
  }
}

private final class Flux2Mistral3TransformerBlock: Module {
  let useSliding: Bool

  @ModuleInfo(key: "self_attn") private var attention: Flux2Mistral3Attention
  private let mlp: Flux2Mistral3MLP

  @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

  init(_ configuration: Flux2Mistral3Configuration, useSliding: Bool) {
    self.useSliding = useSliding
    _attention.wrappedValue = Flux2Mistral3Attention(configuration)
    mlp = Flux2Mistral3MLP(dimensions: configuration.hiddenSize, hiddenDimensions: configuration.intermediateSize)
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
    attnScale: MLXArray
  ) -> MLXArray {
    callAsFunction(x, cache: nil, attentionMask: attentionMask, attnScale: attnScale)
  }

  func callAsFunction(
    _ x: MLXArray,
    cache: Flux2KVCache?,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
    attnScale: MLXArray
  ) -> MLXArray {
    let attn = attention(inputLayerNorm(x), cache: cache, attentionMask: attentionMask, attnScale: attnScale)
    let residual = x + attn
    let mlpOut = mlp(postAttentionLayerNorm(residual))
    return residual + mlpOut
  }
}

private final class Flux2Mistral3Attention: Module {
  private let configuration: Flux2Mistral3Configuration
  private let scale: Float

  @ModuleInfo(key: "q_proj") private var qProj: Linear
  @ModuleInfo(key: "k_proj") private var kProj: Linear
  @ModuleInfo(key: "v_proj") private var vProj: Linear
  @ModuleInfo(key: "o_proj") private var oProj: Linear

  private let rope: RoPE

  init(_ configuration: Flux2Mistral3Configuration) {
    self.configuration = configuration

    let headDim = configuration.resolvedHeadDimensions
    scale = pow(Float(headDim), -0.5)

    _qProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _kProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _vProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _oProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)

    let ropeType = configuration.ropeParameters?["type"]?.asString() ?? configuration.ropeParameters?["rope_type"]?.asString()
    let ropeScale: Float
    if ropeType == "linear",
       let factor = configuration.ropeParameters?["factor"]?.asFloat() {
      ropeScale = 1 / factor
    } else {
      ropeScale = 1.0
    }

    rope = RoPE(dimensions: headDim, traditional: false, base: configuration.ropeTheta, scale: ropeScale)
  }

  func callAsFunction(
    _ x: MLXArray,
    cache: Flux2KVCache?,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
    attnScale: MLXArray
  ) -> MLXArray {
    let batch = x.dim(0)
    let seqLen = x.dim(1)

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    queries = queries.reshaped(batch, seqLen, configuration.attentionHeads, -1).transposed(0, 2, 1, 3)
    keys = keys.reshaped(batch, seqLen, configuration.kvHeads, -1).transposed(0, 2, 1, 3)
    values = values.reshaped(batch, seqLen, configuration.kvHeads, -1).transposed(0, 2, 1, 3)

    let ropeOffset = cache?.offset ?? 0
    queries = rope(queries, offset: ropeOffset)
    keys = rope(keys, offset: ropeOffset)

    let scaleTyped = attnScale.asType(queries.dtype)
    queries = queries * scaleTyped

    let cachedKeys: MLXArray
    let cachedValues: MLXArray
    if let cache {
      (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    } else {
      cachedKeys = keys
      cachedValues = values
    }

    var output = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: cachedKeys,
      values: cachedValues,
      scale: scale,
      mask: attentionMask
    )

    output = output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
    return oProj(output)
  }

  func callAsFunction(
    _ x: MLXArray,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
    attnScale: MLXArray
  ) -> MLXArray {
    callAsFunction(x, cache: nil, attentionMask: attentionMask, attnScale: attnScale)
  }
}

private final class Flux2Mistral3MLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") private var gate: Linear
  @ModuleInfo(key: "down_proj") private var down: Linear
  @ModuleInfo(key: "up_proj") private var up: Linear

  init(dimensions: Int, hiddenDimensions: Int) {
    _gate.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _down.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _up.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    down(silu(gate(x)) * up(x))
  }
}
