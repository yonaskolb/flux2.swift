import Foundation
import MLX
import MLXNN

final class Flux2Attention: Module {
  let headDim: Int
  let innerDim: Int
  let queryDim: Int
  let outDim: Int
  let heads: Int
  let scale: Float
  let addedKvProjDim: Int?

  @ModuleInfo(key: "to_q") private var toQ: Linear
  @ModuleInfo(key: "to_k") private var toK: Linear
  @ModuleInfo(key: "to_v") private var toV: Linear

  @ModuleInfo(key: "norm_q") private var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") private var normK: RMSNorm

  @ModuleInfo(key: "to_out") private var toOut: [Linear]

  @ModuleInfo(key: "norm_added_q") private var normAddedQ: RMSNorm?
  @ModuleInfo(key: "norm_added_k") private var normAddedK: RMSNorm?
  @ModuleInfo(key: "add_q_proj") private var addQProj: Linear?
  @ModuleInfo(key: "add_k_proj") private var addKProj: Linear?
  @ModuleInfo(key: "add_v_proj") private var addVProj: Linear?
  @ModuleInfo(key: "to_add_out") private var toAddOut: Linear?

  init(
    queryDim: Int,
    heads: Int = 8,
    dimHead: Int = 64,
    dropout: Float = 0.0,
    bias: Bool = false,
    addedKvProjDim: Int? = nil,
    addedProjBias: Bool = true,
    outBias: Bool = true,
    eps: Float = 1e-5,
    outDim: Int? = nil
  ) {
    _ = dropout
    self.headDim = dimHead
    self.queryDim = queryDim
    self.outDim = outDim ?? queryDim
    self.innerDim = outDim ?? (dimHead * heads)
    self.heads = outDim != nil ? (self.innerDim / dimHead) : heads
    precondition(self.innerDim % dimHead == 0, "innerDim must be divisible by dimHead.")
    self.scale = pow(Float(dimHead), -0.5)
    self.addedKvProjDim = addedKvProjDim

    _toQ.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)
    _toK.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)
    _toV.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)

    _normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
    _normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)

    _toOut.wrappedValue = [Flux2ModulePlaceholders.linear(bias: outBias)]

    if let addedKvProjDim {
      _normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
      _normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
      _addQProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: addedProjBias)
      _addKProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: addedProjBias)
      _addVProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: addedProjBias)
      _toAddOut.wrappedValue = Flux2ModulePlaceholders.linear(bias: outBias)
    }
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray? = nil,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    imageRotaryEmb: Flux2RotaryEmbeddings? = nil
  ) -> (hiddenStates: MLXArray, encoderHiddenStates: MLXArray?) {
    let batch = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)

    var query = toQ(hiddenStates)
    var key = toK(hiddenStates)
    var value = toV(hiddenStates)

    query = normQ(query.reshaped(batch, seqLen, heads, -1))
    key = normK(key.reshaped(batch, seqLen, heads, -1))
    value = value.reshaped(batch, seqLen, heads, -1)

    var encoderLen: Int? = nil
    if let encoderHiddenStates, let addQProj, let addKProj, let addVProj, let normAddedQ, let normAddedK {
      encoderLen = encoderHiddenStates.dim(1)

      var encoderQuery = addQProj(encoderHiddenStates)
      var encoderKey = addKProj(encoderHiddenStates)
      var encoderValue = addVProj(encoderHiddenStates)

      encoderQuery = normAddedQ(encoderQuery.reshaped(batch, encoderLen!, heads, -1))
      encoderKey = normAddedK(encoderKey.reshaped(batch, encoderLen!, heads, -1))
      encoderValue = encoderValue.reshaped(batch, encoderLen!, heads, -1)

      query = MLX.concatenated([encoderQuery, query], axis: 1)
      key = MLX.concatenated([encoderKey, key], axis: 1)
      value = MLX.concatenated([encoderValue, value], axis: 1)
    }

    if let imageRotaryEmb {
      query = flux2ApplyRotaryEmb(query, imageRotaryEmb, sequenceDim: 1)
      key = flux2ApplyRotaryEmb(key, imageRotaryEmb, sequenceDim: 1)
    }

    let totalSeq = query.dim(1)
    let attnQuery = query.transposed(0, 2, 1, 3)
    let attnKey = key.transposed(0, 2, 1, 3)
    let attnValue = value.transposed(0, 2, 1, 3)

    var computed = MLXFast.scaledDotProductAttention(
      queries: attnQuery,
      keys: attnKey,
      values: attnValue,
      scale: scale,
      mask: attentionMask
    )
    computed = computed.transposed(0, 2, 1, 3).reshaped(batch, totalSeq, -1)

    let typed = computed.dtype == query.dtype ? computed : computed.asType(query.dtype)

    if let encoderLen, let toAddOut {
      let splits = split(typed, indices: [encoderLen], axis: 1)
      let encoderOutput = toAddOut(splits[0])
      let hiddenOutput = toOut[0](splits[1])
      return (hiddenStates: hiddenOutput, encoderHiddenStates: encoderOutput)
    }

    let hiddenOutput = toOut[0](typed)
    return (hiddenStates: hiddenOutput, encoderHiddenStates: nil)
  }
}

final class Flux2ParallelSelfAttention: Module {
  let headDim: Int
  let innerDim: Int
  let queryDim: Int
  let outDim: Int
  let heads: Int
  let mlpRatio: Float
  let mlpHiddenDim: Int
  let mlpMultFactor: Int
  let scale: Float

  @ModuleInfo(key: "to_qkv_mlp_proj") private var toQkvMlpProj: Linear
  @ModuleInfo(key: "norm_q") private var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") private var normK: RMSNorm
  @ModuleInfo(key: "to_out") private var toOut: Linear

  private let mlpActFn = Flux2SwiGLU()

  init(
    queryDim: Int,
    heads: Int = 8,
    dimHead: Int = 64,
    dropout: Float = 0.0,
    bias: Bool = false,
    outBias: Bool = true,
    eps: Float = 1e-5,
    outDim: Int? = nil,
    mlpRatio: Float = 4.0,
    mlpMultFactor: Int = 2
  ) {
    _ = dropout
    self.headDim = dimHead
    self.queryDim = queryDim
    self.outDim = outDim ?? queryDim
    self.innerDim = outDim ?? (dimHead * heads)
    self.heads = outDim != nil ? (self.innerDim / dimHead) : heads
    precondition(self.innerDim % dimHead == 0, "innerDim must be divisible by dimHead.")
    self.mlpRatio = mlpRatio
    self.mlpHiddenDim = Int(Float(queryDim) * mlpRatio)
    self.mlpMultFactor = mlpMultFactor
    self.scale = pow(Float(dimHead), -0.5)

    let projOutDim = innerDim * 3 + mlpHiddenDim * mlpMultFactor
    _toQkvMlpProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)

    _normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)
    _normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: eps)

    _toOut.wrappedValue = Flux2ModulePlaceholders.linear(bias: outBias)
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    imageRotaryEmb: Flux2RotaryEmbeddings? = nil
  ) -> MLXArray {
    let batch = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)

    let projected = toQkvMlpProj(hiddenStates)
    let projectedSplits = split(projected, indices: [3 * innerDim], axis: -1)
    let qkv = projectedSplits[0]
    var mlpHiddenStates = projectedSplits[1]

    let qkvParts = split(qkv, parts: 3, axis: -1)
    var query = qkvParts[0]
    var key = qkvParts[1]
    var value = qkvParts[2]

    query = normQ(query.reshaped(batch, seqLen, heads, -1))
    key = normK(key.reshaped(batch, seqLen, heads, -1))
    value = value.reshaped(batch, seqLen, heads, -1)

    if let imageRotaryEmb {
      query = flux2ApplyRotaryEmb(query, imageRotaryEmb, sequenceDim: 1)
      key = flux2ApplyRotaryEmb(key, imageRotaryEmb, sequenceDim: 1)
    }

    let attnQuery = query.transposed(0, 2, 1, 3)
    let attnKey = key.transposed(0, 2, 1, 3)
    let attnValue = value.transposed(0, 2, 1, 3)

    var computed = MLXFast.scaledDotProductAttention(
      queries: attnQuery,
      keys: attnKey,
      values: attnValue,
      scale: scale,
      mask: attentionMask
    )

    computed = computed.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
    let typed = computed.dtype == query.dtype ? computed : computed.asType(query.dtype)

    mlpHiddenStates = mlpActFn(mlpHiddenStates)
    let fused = MLX.concatenated([typed, mlpHiddenStates], axis: -1)
    return toOut(fused)
  }
}
