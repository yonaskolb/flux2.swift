import Foundation
import MLX
import MLXFast
import MLXNN

final class Flux2PixtralVisionTower: Module {
  let configuration: Flux2PixtralVisionConfiguration
  private let rotary: Flux2PixtralRotaryEmbedding

  @ModuleInfo(key: "patch_conv") private var patchConv: Conv2d
  @ModuleInfo(key: "ln_pre") private var lnPre: RMSNorm
  @ModuleInfo(key: "transformer") private var transformer: Flux2PixtralTransformer

  init(configuration: Flux2PixtralVisionConfiguration) {
    self.configuration = configuration
    rotary = Flux2PixtralRotaryEmbedding(configuration: configuration)

    _patchConv.wrappedValue = Conv2d(
      inputChannels: configuration.numChannels,
      outputChannels: configuration.hiddenSize,
      kernelSize: IntOrPair(configuration.patchSize),
      stride: IntOrPair(configuration.patchSize),
      padding: 0,
      bias: false
    )
    _lnPre.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
    _transformer.wrappedValue = Flux2PixtralTransformer(configuration: configuration, rotary: rotary)

    super.init()
  }

  func selectedHiddenStates(
    pixelValues: MLXArray,
    imageSizes: [(height: Int, width: Int)],
    featureLayers: [Int]
  ) -> MLXArray {
    precondition(pixelValues.ndim == 4, "pixelValues must be [batch, channels, height, width].")
    precondition(pixelValues.dim(0) == imageSizes.count, "imageSizes must match pixelValues batch.")

    let totalStates = configuration.hiddenLayers + 1
    let resolvedLayers = featureLayers.map { layer -> Int in
      layer >= 0 ? layer : totalStates + layer
    }

    let wanted = Set(resolvedLayers)

    var perImageSelected: [MLXArray] = []
    perImageSelected.reserveCapacity(imageSizes.count)

    for (index, size) in imageSizes.enumerated() {
      let height = max(size.height, 1)
      let width = max(size.width, 1)

      var image = pixelValues[index, 0..., 0..<height, 0..<width]
      if image.dtype != patchConv.weight.dtype {
        image = image.asType(patchConv.weight.dtype)
      }

      var nhwc = image.transposed(1, 2, 0)
      nhwc = nhwc.reshaped(1, height, width, configuration.numChannels)

      var patchEmbeds = patchConv(nhwc)
      let patchHeight = patchEmbeds.dim(1)
      let patchWidth = patchEmbeds.dim(2)
      patchEmbeds = patchEmbeds.reshaped(1, patchHeight * patchWidth, configuration.hiddenSize)

      var hidden = lnPre(patchEmbeds)

      let positionIds = makePositionIds(height: patchHeight, width: patchWidth)
      let (cos, sin) = rotary(positionIds: positionIds, dtype: hidden.dtype)

      var captured: [Int: MLXArray] = [:]
      if wanted.contains(0) {
        captured[0] = hidden
      }

      for (layerIndex, layer) in transformer.layers.enumerated() {
        hidden = layer(hidden, cos: cos, sin: sin)
        let stateIndex = layerIndex + 1
        if wanted.contains(stateIndex) {
          captured[stateIndex] = hidden
        }
      }

      let selectedStates: [MLXArray] = resolvedLayers.compactMap { captured[$0] }
      precondition(
        selectedStates.count == resolvedLayers.count,
        "Requested feature layers are missing: \(resolvedLayers)"
      )

      let selected = selectedStates.count == 1
        ? selectedStates[0]
        : MLX.concatenated(selectedStates, axis: -1)
      perImageSelected.append(selected)
    }

    if perImageSelected.count == 1 {
      return perImageSelected[0]
    }
    return MLX.concatenated(perImageSelected, axis: 1)
  }

  private func makePositionIds(height: Int, width: Int) -> MLXArray {
    let maxWidth = max(configuration.imageSize / max(configuration.patchSize, 1), 1)
    var ids = [Int32]()
    ids.reserveCapacity(max(height * width, 1))
    for h in 0..<max(height, 1) {
      for w in 0..<max(width, 1) {
        ids.append(Int32(h * maxWidth + w))
      }
    }
    return MLXArray(ids, [ids.count]).asType(.int32)
  }
}

final class Flux2PixtralRotaryEmbedding: @unchecked Sendable {
  private let invFreq: MLXArray

  init(configuration: Flux2PixtralVisionConfiguration) {
    let headDim = max(configuration.resolvedHeadDimensions, 1)
    let base = max(configuration.ropeTheta, 1)
    let maxPatchesPerSide = max(configuration.imageSize / max(configuration.patchSize, 1), 1)

    let halfDim = headDim / 2
    var freqs = [Float32]()
    freqs.reserveCapacity(max(halfDim, 1))
    for i in stride(from: 0, to: headDim, by: 2) {
      let exponent = Float32(i) / Float32(headDim)
      freqs.append(1.0 / pow(Float32(base), exponent))
    }

    let hFreqs = stride(from: 0, to: freqs.count, by: 2).map { freqs[$0] }
    let wFreqs = stride(from: 1, to: freqs.count, by: 2).map { freqs[$0] }

    let hDim = hFreqs.count
    let wDim = wFreqs.count

    let positions = maxPatchesPerSide * maxPatchesPerSide
    let fullHalfDim = hDim + wDim
    var table = [Float32](repeating: 0, count: positions * fullHalfDim)

    for h in 0..<maxPatchesPerSide {
      for w in 0..<maxPatchesPerSide {
        let row = h * maxPatchesPerSide + w
        let baseIndex = row * fullHalfDim
        for j in 0..<hDim {
          table[baseIndex + j] = Float32(h) * hFreqs[j]
        }
        for j in 0..<wDim {
          table[baseIndex + hDim + j] = Float32(w) * wFreqs[j]
        }
      }
    }

    var duplicated = [Float32](repeating: 0, count: positions * fullHalfDim * 2)
    duplicated.withUnsafeMutableBufferPointer { dst in
      table.withUnsafeBufferPointer { src in
        let rowSize = fullHalfDim
        for row in 0..<positions {
          let srcStart = row * rowSize
          let dstStart = row * rowSize * 2
          dst.baseAddress!.advanced(by: dstStart).assign(from: src.baseAddress!.advanced(by: srcStart), count: rowSize)
          dst.baseAddress!.advanced(by: dstStart + rowSize).assign(from: src.baseAddress!.advanced(by: srcStart), count: rowSize)
        }
      }
    }

    let data = duplicated.withUnsafeBufferPointer { Data(buffer: $0) }
    invFreq = MLXArray(data, [positions, fullHalfDim * 2], dtype: .float32)
  }

  func callAsFunction(positionIds: MLXArray, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
    let ids = positionIds.asType(.int32)
    let freqs = invFreq[ids]
    var cos = MLX.cos(freqs)
    var sin = MLX.sin(freqs)
    if cos.dtype != dtype {
      cos = cos.asType(dtype)
    }
    if sin.dtype != dtype {
      sin = sin.asType(dtype)
    }
    return (cos: cos, sin: sin)
  }
}

final class Flux2PixtralTransformer: Module {
  let layers: [Flux2PixtralAttentionLayer]

  init(configuration: Flux2PixtralVisionConfiguration, rotary: Flux2PixtralRotaryEmbedding) {
    layers = (0..<configuration.hiddenLayers).map { _ in
      Flux2PixtralAttentionLayer(configuration: configuration, rotary: rotary)
    }
  }
}

final class Flux2PixtralAttentionLayer: Module {
  @ModuleInfo(key: "attention_norm") private var attentionNorm: RMSNorm
  @ModuleInfo(key: "attention") private var attention: Flux2PixtralAttention
  @ModuleInfo(key: "ffn_norm") private var ffnNorm: RMSNorm
  @ModuleInfo(key: "feed_forward") private var feedForward: Flux2PixtralMLP

  init(configuration: Flux2PixtralVisionConfiguration, rotary: Flux2PixtralRotaryEmbedding) {
    _attentionNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
    _attention.wrappedValue = Flux2PixtralAttention(configuration: configuration, rotary: rotary)
    _ffnNorm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: 1e-5)
    _feedForward.wrappedValue = Flux2PixtralMLP(configuration: configuration)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let residual = x
    let attn = attention(attentionNorm(x), cos: cos, sin: sin)
    let hidden = residual + attn
    return hidden + feedForward(ffnNorm(hidden))
  }
}

final class Flux2PixtralAttention: Module {
  private let configuration: Flux2PixtralVisionConfiguration
  private let scale: Float
  private let rotary: Flux2PixtralRotaryEmbedding

  @ModuleInfo(key: "q_proj") private var qProj: Linear
  @ModuleInfo(key: "k_proj") private var kProj: Linear
  @ModuleInfo(key: "v_proj") private var vProj: Linear
  @ModuleInfo(key: "o_proj") private var oProj: Linear

  init(configuration: Flux2PixtralVisionConfiguration, rotary: Flux2PixtralRotaryEmbedding) {
    self.configuration = configuration
    self.rotary = rotary

    let headDim = max(configuration.resolvedHeadDimensions, 1)
    scale = pow(Float(headDim), -0.5)

    _qProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _kProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _vProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _oProj.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let batch = x.dim(0)
    let seqLen = x.dim(1)
    let heads = max(configuration.attentionHeads, 1)
    let headDim = max(configuration.resolvedHeadDimensions, 1)

    var q = qProj(x)
    var k = kProj(x)
    var v = vProj(x)

    q = q.reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)
    k = k.reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)
    v = v.reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)

    let (qRot, kRot) = applyRotary(q: q, k: k, cos: cos, sin: sin)

    var output = MLXFast.scaledDotProductAttention(
      queries: qRot,
      keys: kRot,
      values: v,
      scale: scale,
      mask: .none
    )

    output = output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, heads * headDim)
    return oProj(output)
  }

  private func applyRotary(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let seqLen = q.dim(2)
    let headDim = q.dim(3)
    let half = headDim / 2

    var cosExpanded = cos
    var sinExpanded = sin
    if cosExpanded.ndim == 2 {
      cosExpanded = cosExpanded.reshaped(1, 1, seqLen, headDim)
      sinExpanded = sinExpanded.reshaped(1, 1, seqLen, headDim)
    }

    func rotateHalf(_ x: MLXArray) -> MLXArray {
      let x1 = x[.ellipsis, 0..<half]
      let x2 = x[.ellipsis, half..<headDim]
      return MLX.concatenated([-x2, x1], axis: -1)
    }

    let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)
    return (qEmbed, kEmbed)
  }
}

final class Flux2PixtralMLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") private var gate: Linear
  @ModuleInfo(key: "up_proj") private var up: Linear
  @ModuleInfo(key: "down_proj") private var down: Linear

  init(configuration: Flux2PixtralVisionConfiguration) {
    _gate.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _up.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    _down.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    down(gelu(gate(x)) * up(x))
  }
}
