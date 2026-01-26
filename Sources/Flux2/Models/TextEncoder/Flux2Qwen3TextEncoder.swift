import Foundation
import MLX
import MLXNN

public enum Flux2Qwen3TextEncoderError: Error {
  case configNotFound(URL)
  case invalidInputShape
  case missingHiddenState(Int)
}

public final class Flux2Qwen3TextEncoder: Module {
  public let configuration: Flux2Qwen3Configuration

  @ModuleInfo(key: "model") private var model: Flux2Qwen3Model
  @ModuleInfo(key: "lm_head") private var lmHead: Linear?

  public init(configuration: Flux2Qwen3Configuration) {
    self.configuration = configuration
    _model.wrappedValue = Flux2Qwen3Model(configuration)

    if !configuration.tieWordEmbeddings {
      _lmHead.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    }

    super.init()
  }

  public static func load(from snapshot: URL, dtype: DType = .bfloat16) throws -> Flux2Qwen3TextEncoder {
    let configURL = snapshot
      .appendingPathComponent("text_encoder")
      .appendingPathComponent("config.json")

    guard FileManager.default.fileExists(atPath: configURL.path) else {
      throw Flux2Qwen3TextEncoderError.configNotFound(configURL)
    }

    let configData = try Data(contentsOf: configURL)
    let configuration = try JSONDecoder().decode(Flux2Qwen3Configuration.self, from: configData)
    let encoder = Flux2Qwen3TextEncoder(configuration: configuration)

    let loader = Flux2WeightsLoader(snapshot: snapshot)
    let weights = try loader.load(component: .textEncoder, dtype: dtype)
    if let manifest = try Flux2Quantizer.loadManifest(from: snapshot) {
      Flux2Quantizer.applyQuantization(to: encoder, manifest: manifest, weights: weights)
    }
    try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: .none)

    return encoder
  }

  public func promptEmbeds(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    hiddenStateLayers: [Int] = [9, 18, 27],
    evaluationPolicy: Flux2EvaluationPolicy = .deferred
  ) throws -> MLXArray {
    guard inputIds.ndim == 2, attentionMask.ndim == 2 else {
      throw Flux2Qwen3TextEncoderError.invalidInputShape
    }
    guard inputIds.shape == attentionMask.shape else {
      throw Flux2Qwen3TextEncoderError.invalidInputShape
    }

    let hiddenStates = model.hiddenStates(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputLayerIndices: hiddenStateLayers,
      evaluationPolicy: evaluationPolicy
    )

    var selected: [MLXArray] = []
    selected.reserveCapacity(hiddenStateLayers.count)
    for index in hiddenStateLayers {
      guard let state = hiddenStates[index] else {
        throw Flux2Qwen3TextEncoderError.missingHiddenState(index)
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

private enum Flux2Qwen3AttentionMask {
  static func make(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?
  ) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let seqLen = hiddenStates.dim(1)

    guard let attentionMask else {
      return seqLen == 1 ? .none : .causal
    }

    precondition(attentionMask.ndim == 2, "attentionMask must be [batch, seq_len].")

    let mask = attentionMask.asType(hiddenStates.dtype)
    let zeros = MLX.zeros(mask.shape, dtype: hiddenStates.dtype)
    let negInf = MLXArray(-Float.infinity).asType(hiddenStates.dtype)
    let keepMask = mask .== MLXArray(1).asType(hiddenStates.dtype)
    var additivePadding = MLX.where(keepMask, zeros, zeros + negInf)
    additivePadding = additivePadding.reshaped(additivePadding.dim(0), 1, 1, seqLen)

    let causalAdditive = Flux2AttentionMaskCache.causalAdditive(seqLen: seqLen, dtype: hiddenStates.dtype)
    return .array(causalAdditive + additivePadding)
  }
}

private final class Flux2Qwen3Model: Module {
  @ModuleInfo(key: "embed_tokens") private var embedTokens: Embedding

  private let layers: [Flux2Qwen3TransformerBlock]
  private let norm: RMSNorm
  private let configuration: Flux2Qwen3Configuration

  init(_ configuration: Flux2Qwen3Configuration) {
    self.configuration = configuration
    _embedTokens.wrappedValue = Flux2ModulePlaceholders.embedding()
    layers = (0..<configuration.hiddenLayers).map { _ in
      Flux2Qwen3TransformerBlock(configuration)
    }
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

    let mask = Flux2Qwen3AttentionMask.make(hiddenStates: hidden, attentionMask: attentionMask)

    for (index, layer) in layers.enumerated() {
      hidden = layer(hidden, attentionMask: mask)
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

  func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }

    var hidden = embedTokens(tokenIds)
    let mask = Flux2Qwen3AttentionMask.make(hiddenStates: hidden, attentionMask: attentionMask)
    for layer in layers {
      hidden = layer(hidden, attentionMask: mask)
    }
    let normalized = norm(hidden)
    return normalized
  }
}

private final class Flux2Qwen3TransformerBlock: Module {
  @ModuleInfo(key: "self_attn") private var attention: Flux2Qwen3Attention
  private let mlp: Flux2Qwen3MLP

  @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

  init(_ configuration: Flux2Qwen3Configuration) {
    _attention.wrappedValue = Flux2Qwen3Attention(configuration)
    mlp = Flux2Qwen3MLP(dimensions: configuration.hiddenSize, hiddenDimensions: configuration.intermediateSize)
    _inputLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  func callAsFunction(
    _ x: MLXArray,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
  ) -> MLXArray {
    let attn = attention(inputLayerNorm(x), attentionMask: attentionMask)
    let residual = x + attn
    let mlpOut = mlp(postAttentionLayerNorm(residual))
    return residual + mlpOut
  }
}

private final class Flux2Qwen3Attention: Module {
  private let configuration: Flux2Qwen3Configuration
  private let scale: Float

  @ModuleInfo(key: "q_proj") private var qProj: Linear
  @ModuleInfo(key: "k_proj") private var kProj: Linear
  @ModuleInfo(key: "v_proj") private var vProj: Linear
  @ModuleInfo(key: "o_proj") private var oProj: Linear

  @ModuleInfo(key: "q_norm") private var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") private var kNorm: RMSNorm

  private let rope: RoPE

  init(_ configuration: Flux2Qwen3Configuration) {
    self.configuration = configuration

    let headDim = configuration.headDim
    scale = pow(Float(headDim), -0.5)

    _qProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _kProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _vProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _oProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)

    _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
    _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)

    let ropeScale: Float
    if let ropeScaling = configuration.ropeScaling,
       ropeScaling["type"]?.asString() == "linear",
       let factor = ropeScaling["factor"]?.asFloat() {
      ropeScale = 1 / factor
    } else {
      ropeScale = 1
    }

    rope = RoPE(dimensions: headDim, traditional: false, base: configuration.ropeTheta, scale: ropeScale)
  }

  func callAsFunction(
    _ x: MLXArray,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
  ) -> MLXArray {
    let batch = x.dim(0)
    let seqLen = x.dim(1)

    var queries = qProj(x)
    var keys = kProj(x)
    var values = vProj(x)

    queries = qNorm(queries.reshaped(batch, seqLen, configuration.attentionHeads, -1))
      .transposed(0, 2, 1, 3)
    keys = kNorm(keys.reshaped(batch, seqLen, configuration.kvHeads, -1))
      .transposed(0, 2, 1, 3)
    values = values.reshaped(batch, seqLen, configuration.kvHeads, -1)
      .transposed(0, 2, 1, 3)

    queries = rope(queries)
    keys = rope(keys)

    var output = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      mask: attentionMask
    )

    output = output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
    return oProj(output)
  }

}

private final class Flux2Qwen3MLP: Module, UnaryLayer {
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
