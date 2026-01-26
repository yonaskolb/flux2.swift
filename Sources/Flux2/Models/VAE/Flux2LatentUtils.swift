import Foundation
import MLX
import MLXNN

public enum Flux2LatentUtilsError: Error {
  case expectedNCHW(ndim: Int)
  case expectedPackedLatents(ndim: Int)
  case latentsNotDivisibleByPatchSize(height: Int, width: Int, patchHeight: Int, patchWidth: Int)
  case channelsNotDivisibleByPatchArea(channels: Int, patchArea: Int)
  case invalidIdsShape(ndim: Int)
  case idsBatchMismatch(latentsBatch: Int, idsBatch: Int)
  case batchNormMissingRunningStats
}

public enum Flux2LatentUtils {
  public static func patchify(
    _ latents: MLXArray,
    patchSize: (Int, Int)
  ) throws -> MLXArray {
    guard latents.ndim == 4 else {
      throw Flux2LatentUtilsError.expectedNCHW(ndim: latents.ndim)
    }
    let patchH = patchSize.0
    let patchW = patchSize.1
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let height = latents.dim(2)
    let width = latents.dim(3)
    guard height % patchH == 0 && width % patchW == 0 else {
      throw Flux2LatentUtilsError.latentsNotDivisibleByPatchSize(
        height: height,
        width: width,
        patchHeight: patchH,
        patchWidth: patchW
      )
    }

    var hidden = latents.reshaped(
      batch,
      channels,
      height / patchH,
      patchH,
      width / patchW,
      patchW
    )
    hidden = hidden.transposed(0, 1, 3, 5, 2, 4)
    hidden = hidden.reshaped(batch, channels * patchH * patchW, height / patchH, width / patchW)
    return hidden
  }

  public static func unpatchify(
    _ latents: MLXArray,
    patchSize: (Int, Int)
  ) throws -> MLXArray {
    guard latents.ndim == 4 else {
      throw Flux2LatentUtilsError.expectedNCHW(ndim: latents.ndim)
    }
    let patchH = patchSize.0
    let patchW = patchSize.1
    let multiplier = patchH * patchW
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let height = latents.dim(2)
    let width = latents.dim(3)
    guard channels % multiplier == 0 else {
      throw Flux2LatentUtilsError.channelsNotDivisibleByPatchArea(
        channels: channels,
        patchArea: multiplier
      )
    }

    let baseChannels = channels / multiplier
    var hidden = latents.reshaped(batch, baseChannels, patchH, patchW, height, width)
    hidden = hidden.transposed(0, 1, 4, 2, 5, 3)
    hidden = hidden.reshaped(batch, baseChannels, height * patchH, width * patchW)
    return hidden
  }

  public static func packLatents(_ latents: MLXArray) throws -> MLXArray {
    guard latents.ndim == 4 else {
      throw Flux2LatentUtilsError.expectedNCHW(ndim: latents.ndim)
    }

    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let height = latents.dim(2)
    let width = latents.dim(3)

    var packed = latents.reshaped(batch, channels, height * width)
    packed = packed.transposed(0, 2, 1)
    return packed
  }

  public static func unpackLatentsWithIds(
    _ latents: MLXArray,
    ids: MLXArray
  ) throws -> MLXArray {
    guard latents.ndim == 3 else {
      throw Flux2LatentUtilsError.expectedPackedLatents(ndim: latents.ndim)
    }

    var idsLocal = ids
    if idsLocal.ndim == 2 {
      idsLocal = idsLocal.reshaped(1, idsLocal.dim(0), idsLocal.dim(1))
    }
    guard idsLocal.ndim == 3 else {
      throw Flux2LatentUtilsError.invalidIdsShape(ndim: idsLocal.ndim)
    }
    guard idsLocal.dim(0) == latents.dim(0) else {
      throw Flux2LatentUtilsError.idsBatchMismatch(
        latentsBatch: latents.dim(0),
        idsBatch: idsLocal.dim(0)
      )
    }

    let batch = latents.dim(0)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(batch)

    for index in 0..<batch {
      let data = latents[index]
      let pos = idsLocal[index]
      let hIds = pos[0..., 1].asType(.int32)
      let wIds = pos[0..., 2].asType(.int32)

      let height = Int(hIds.max().item(Int32.self)) + 1
      let width = Int(wIds.max().item(Int32.self)) + 1
      let channels = data.dim(1)

      let widthArray = MLXArray(Int32(width))
      let flatIds = hIds * widthArray + wIds

      var out = MLX.zeros([height * width, channels], dtype: data.dtype)
      out[flatIds, .ellipsis] = data

      let reshaped = out.reshaped(height, width, channels).transposed(2, 0, 1)
      outputs.append(reshaped)
    }

    return MLX.stacked(outputs, axis: 0)
  }

  public static func normalizePatchifiedLatents(
    _ latents: MLXArray,
    batchNorm: BatchNorm
  ) throws -> MLXArray {
    let (mean, std) = try runningStats(batchNorm, dtype: latents.dtype)
    return (latents - mean) / std
  }

  public static func denormalizePatchifiedLatents(
    _ latents: MLXArray,
    batchNorm: BatchNorm
  ) throws -> MLXArray {
    let (mean, std) = try runningStats(batchNorm, dtype: latents.dtype)
    return latents * std + mean
  }

  public static func patchifyAndNormalize(
    _ latents: MLXArray,
    vae: Flux2AutoencoderKL
  ) throws -> MLXArray {
    let patchified = try patchify(latents, patchSize: vae.configuration.patchSize)
    return try normalizePatchifiedLatents(patchified, batchNorm: vae.bn)
  }

  public static func denormalizeAndUnpatchify(
    _ latents: MLXArray,
    vae: Flux2AutoencoderKL
  ) throws -> MLXArray {
    let denorm = try denormalizePatchifiedLatents(latents, batchNorm: vae.bn)
    return try unpatchify(denorm, patchSize: vae.configuration.patchSize)
  }

  private static func runningStats(
    _ batchNorm: BatchNorm,
    dtype: DType
  ) throws -> (mean: MLXArray, std: MLXArray) {
    let params = Dictionary(uniqueKeysWithValues: batchNorm.parameters().flattened())
    guard let runningMean = params["running_mean"], let runningVar = params["running_var"] else {
      throw Flux2LatentUtilsError.batchNormMissingRunningStats
    }

    let mean = runningMean.asType(dtype).reshaped(1, -1, 1, 1)
    let variance = runningVar.asType(dtype).reshaped(1, -1, 1, 1)
    let eps = MLXArray(Float(batchNorm.eps)).asType(dtype)
    let std = sqrt(variance + eps)
    return (mean, std)
  }
}
