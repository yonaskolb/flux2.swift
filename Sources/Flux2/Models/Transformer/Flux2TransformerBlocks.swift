import MLX
import MLXNN

private func flux2ApplyModulation(
  _ x: MLXArray,
  shift: MLXArray,
  scale: MLXArray
) -> MLXArray {
  let one = MLXArray(1.0).asType(scale.dtype)
  return (one + scale) * x + shift
}

private func flux2ClipIfFloat16(_ x: MLXArray) -> MLXArray {
  guard x.dtype == .float16 else {
    return x
  }
  let bound = MLXArray(Float(65504.0))
  let clipped = clip(x.asType(.float32), min: -bound, max: bound)
  return clipped.asType(.float16)
}

final class Flux2SingleTransformerBlock: Module {
  private let norm: Flux2LayerNorm
  @ModuleInfo(key: "attn") private var attn: Flux2ParallelSelfAttention

  init(
    dim: Int,
    numAttentionHeads: Int,
    attentionHeadDim: Int,
    mlpRatio: Float = 3.0,
    eps: Float = 1e-6,
    bias: Bool = false
  ) {
    norm = Flux2LayerNorm(eps: eps)
    _attn.wrappedValue = Flux2ParallelSelfAttention(
      queryDim: dim,
      heads: numAttentionHeads,
      dimHead: attentionHeadDim,
      bias: bias,
      outBias: bias,
      eps: eps,
      outDim: dim,
      mlpRatio: mlpRatio,
      mlpMultFactor: 2
    )
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray? = nil,
    tembModParams: Flux2ModulationParams,
    imageRotaryEmb: Flux2RotaryEmbeddings? = nil,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    splitHiddenStates: Bool = false,
    textSeqLen: Int? = nil
  ) -> (encoderHiddenStates: MLXArray?, hiddenStates: MLXArray) {
    var combined = hiddenStates
    var splitLen = textSeqLen

    if let encoderHiddenStates {
      splitLen = encoderHiddenStates.dim(1)
      combined = MLX.concatenated([encoderHiddenStates, hiddenStates], axis: 1)
    }

    let (shift, scale, gate) = tembModParams
    var normed = norm(combined)
    normed = flux2ApplyModulation(normed, shift: shift, scale: scale)

    let attnOutput = attn(normed, attentionMask: attentionMask, imageRotaryEmb: imageRotaryEmb)
    var output = combined + gate * attnOutput
    output = flux2ClipIfFloat16(output)

    if splitHiddenStates, let splitLen {
      let splits = split(output, indices: [splitLen], axis: 1)
      return (encoderHiddenStates: splits[0], hiddenStates: splits[1])
    }

    return (encoderHiddenStates: nil, hiddenStates: output)
  }
}

final class Flux2TransformerBlock: Module {
  private let norm1: Flux2LayerNorm
  private let norm1Context: Flux2LayerNorm
  @ModuleInfo(key: "attn") private var attn: Flux2Attention

  private let norm2: Flux2LayerNorm
  @ModuleInfo(key: "ff") private var ff: Flux2FeedForward

  private let norm2Context: Flux2LayerNorm
  @ModuleInfo(key: "ff_context") private var ffContext: Flux2FeedForward

  init(
    dim: Int,
    numAttentionHeads: Int,
    attentionHeadDim: Int,
    mlpRatio: Float = 3.0,
    eps: Float = 1e-6,
    bias: Bool = false
  ) {
    norm1 = Flux2LayerNorm(eps: eps)
    norm1Context = Flux2LayerNorm(eps: eps)
    _attn.wrappedValue = Flux2Attention(
      queryDim: dim,
      heads: numAttentionHeads,
      dimHead: attentionHeadDim,
      bias: bias,
      addedKvProjDim: dim,
      addedProjBias: bias,
      outBias: bias,
      eps: eps,
      outDim: dim
    )

    norm2 = Flux2LayerNorm(eps: eps)
    _ff.wrappedValue = Flux2FeedForward(dim: dim, dimOut: dim, mult: mlpRatio, bias: bias)

    norm2Context = Flux2LayerNorm(eps: eps)
    _ffContext.wrappedValue = Flux2FeedForward(dim: dim, dimOut: dim, mult: mlpRatio, bias: bias)
  }

  func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    tembModParamsImg: [Flux2ModulationParams],
    tembModParamsTxt: [Flux2ModulationParams],
    imageRotaryEmb: Flux2RotaryEmbeddings? = nil,
    attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
  ) -> (encoderHiddenStates: MLXArray, hiddenStates: MLXArray) {
    precondition(tembModParamsImg.count >= 2, "Expected at least 2 modulation parameter sets.")
    precondition(tembModParamsTxt.count >= 2, "Expected at least 2 modulation parameter sets.")

    let (shiftMsa, scaleMsa, gateMsa) = tembModParamsImg[0]
    let (shiftMlp, scaleMlp, gateMlp) = tembModParamsImg[1]
    let (shiftCtxMsa, scaleCtxMsa, gateCtxMsa) = tembModParamsTxt[0]
    let (shiftCtxMlp, scaleCtxMlp, gateCtxMlp) = tembModParamsTxt[1]

    var normHiddenStates = norm1(hiddenStates)
    normHiddenStates = flux2ApplyModulation(normHiddenStates, shift: shiftMsa, scale: scaleMsa)

    var normEncoderHiddenStates = norm1Context(encoderHiddenStates)
    normEncoderHiddenStates = flux2ApplyModulation(
      normEncoderHiddenStates,
      shift: shiftCtxMsa,
      scale: scaleCtxMsa
    )

    let attentionOutputs = attn(
      normHiddenStates,
      encoderHiddenStates: normEncoderHiddenStates,
      attentionMask: attentionMask,
      imageRotaryEmb: imageRotaryEmb
    )

    guard let contextAttention = attentionOutputs.encoderHiddenStates else {
      preconditionFailure("Expected encoder attention output when encoderHiddenStates is provided.")
    }

    var updatedHiddenStates = hiddenStates + gateMsa * attentionOutputs.hiddenStates
    var normHiddenMlp = norm2(updatedHiddenStates)
    normHiddenMlp = flux2ApplyModulation(normHiddenMlp, shift: shiftMlp, scale: scaleMlp)
    let ffOutput = ff(normHiddenMlp)
    updatedHiddenStates = updatedHiddenStates + gateMlp * ffOutput

    var updatedEncoderHiddenStates = encoderHiddenStates + gateCtxMsa * contextAttention
    var normEncoderMlp = norm2Context(updatedEncoderHiddenStates)
    normEncoderMlp = flux2ApplyModulation(normEncoderMlp, shift: shiftCtxMlp, scale: scaleCtxMlp)
    let contextFfOutput = ffContext(normEncoderMlp)
    updatedEncoderHiddenStates = updatedEncoderHiddenStates + gateCtxMlp * contextFfOutput
    updatedEncoderHiddenStates = flux2ClipIfFloat16(updatedEncoderHiddenStates)

    return (encoderHiddenStates: updatedEncoderHiddenStates, hiddenStates: updatedHiddenStates)
  }
}
