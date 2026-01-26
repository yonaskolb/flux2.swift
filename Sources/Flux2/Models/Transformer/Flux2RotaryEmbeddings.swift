import Foundation
import MLX
import MLXNN

typealias Flux2RotaryEmbeddings = (cos: MLXArray, sin: MLXArray)

private func repeatInterleaveLastAxis(_ array: MLXArray) -> MLXArray {
  let stacked = MLX.stacked([array, array], axis: -1)
  var shape = array.shape
  shape[shape.count - 1] *= 2
  return stacked.reshaped(shape)
}

private func flux2Get1DRotaryPosEmbed(
  dim: Int,
  pos: MLXArray,
  theta: Float
) -> (MLXArray, MLXArray) {
  precondition(dim % 2 == 0, "Rotary dimension must be even.")
  precondition(pos.ndim == 1 || pos.ndim == 2, "pos must be a 1D or 2D array.")

  let halfDim = dim / 2
  var exponent = MLXArray(0..<halfDim).asType(.float32)
  exponent = exponent * 2.0
  exponent = exponent / MLXArray(Float(dim))

  let base = MLXArray(theta)
  var freqs = MLX.pow(base, exponent)
  freqs = MLXArray(1.0) / freqs

  let positions = pos.asType(.float32).reshaped(pos.shape + [1])
  let freqsShape = Array(repeating: 1, count: max(positions.ndim - 1, 1)) + [halfDim]
  let angles = positions * freqs.reshaped(freqsShape)

  let cosVals = MLX.cos(angles)
  let sinVals = MLX.sin(angles)

  let cosOut = repeatInterleaveLastAxis(cosVals)
  let sinOut = repeatInterleaveLastAxis(sinVals)

  return (cosOut, sinOut)
}

private func flux2ExpandRotary(_ emb: MLXArray, sequenceDim: Int) -> MLXArray {
  switch sequenceDim {
  case 1:
    if emb.ndim == 2 {
      return emb.reshaped(1, emb.dim(0), 1, emb.dim(1))
    }
    if emb.ndim == 3 {
      return emb.reshaped(emb.dim(0), emb.dim(1), 1, emb.dim(2))
    }
  case 2:
    if emb.ndim == 2 {
      return emb.reshaped(1, 1, emb.dim(0), emb.dim(1))
    }
    if emb.ndim == 3 {
      return emb.reshaped(emb.dim(0), 1, emb.dim(1), emb.dim(2))
    }
  default:
    break
  }
  preconditionFailure("rotary embeddings must be [seq, dim] or [batch, seq, dim] for sequenceDim \(sequenceDim).")
}

func flux2ApplyRotaryEmbReference(
  _ x: MLXArray,
  _ rotary: Flux2RotaryEmbeddings,
  sequenceDim: Int = 1
) -> MLXArray {
  let cos = flux2ExpandRotary(rotary.cos, sequenceDim: sequenceDim)
  let sin = flux2ExpandRotary(rotary.sin, sequenceDim: sequenceDim)

  let shape = x.shape
  let lastDim = shape[shape.count - 1]
  precondition(lastDim % 2 == 0, "Rotary dimension must be even.")

  let newShape = Array(shape.dropLast()) + [lastDim / 2, 2]
  let reshaped = x.reshaped(newShape)
  let xReal = reshaped[0..., 0..., 0..., 0..., 0]
  let xImag = reshaped[0..., 0..., 0..., 0..., 1]
  let rotated = MLX.stacked([-xImag, xReal], axis: -1).reshaped(shape)

  let xFloat = x.asType(.float32)
  let rotatedFloat = rotated.asType(.float32)
  let out = xFloat * cos + rotatedFloat * sin
  return out.asType(x.dtype)
}

func flux2ApplyRotaryEmb(
  _ x: MLXArray,
  _ rotary: Flux2RotaryEmbeddings,
  sequenceDim: Int = 1
) -> MLXArray {
  if sequenceDim == 1,
     let fused = Flux2FusedKernels.applyRotaryEmb(x, cos: rotary.cos, sin: rotary.sin) {
    return fused
  }

  return flux2ApplyRotaryEmbReference(x, rotary, sequenceDim: sequenceDim)
}

final class Flux2PosEmbed: Module {
  let theta: Float
  let axesDims: [Int]

  init(theta: Int, axesDims: [Int]) {
    self.theta = Float(theta)
    self.axesDims = axesDims
  }

  func callAsFunction(_ ids: MLXArray) -> Flux2RotaryEmbeddings {
    precondition(ids.ndim == 3, "ids must be [B, S, num_axes].")
    precondition(ids.dim(2) == axesDims.count, "ids last dimension must equal axes count.")

    let pos = ids.asType(.float32)
    var cosParts: [MLXArray] = []
    var sinParts: [MLXArray] = []
    cosParts.reserveCapacity(axesDims.count)
    sinParts.reserveCapacity(axesDims.count)

    for (axisIndex, dim) in axesDims.enumerated() {
      let axisPos = pos[0..., 0..., axisIndex]
      let (cosVals, sinVals) = flux2Get1DRotaryPosEmbed(dim: dim, pos: axisPos, theta: theta)
      cosParts.append(cosVals)
      sinParts.append(sinVals)
    }

    let cosOut = cosParts.count == 1 ? cosParts[0] : MLX.concatenated(cosParts, axis: -1)
    let sinOut = sinParts.count == 1 ? sinParts[0] : MLX.concatenated(sinParts, axis: -1)
    return (cosOut, sinOut)
  }
}
