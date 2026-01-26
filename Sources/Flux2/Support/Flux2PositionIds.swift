import Foundation
import MLX

public enum Flux2PositionIdsError: Error {
  case invalidPromptEmbedsShape([Int])
  case invalidTCoordShape([Int])
  case invalidTCoordBatch(expected: Int, actual: Int)
  case invalidLatentsShape([Int])
  case emptyImageLatents
  case invalidImageLatentsShape(index: Int, shape: [Int])
  case invalidImageLatentsBatch(index: Int, batch: Int)
}

public enum Flux2PositionIds {
  public static func prepareTextIds(
    _ promptEmbeds: MLXArray,
    tCoord: MLXArray? = nil
  ) throws -> MLXArray {
    guard promptEmbeds.ndim == 3 else {
      throw Flux2PositionIdsError.invalidPromptEmbedsShape(promptEmbeds.shape)
    }

    let batch = promptEmbeds.dim(0)
    let seqLen = promptEmbeds.dim(1)
    let l = MLXArray(0..<seqLen).asType(.int32)
    let h = MLXArray(0..<1).asType(.int32)
    let w = MLXArray(0..<1).asType(.int32)

    if let tCoord {
      guard tCoord.ndim == 1 || tCoord.ndim == 2 else {
        throw Flux2PositionIdsError.invalidTCoordShape(tCoord.shape)
      }
      guard tCoord.dim(0) == batch else {
        throw Flux2PositionIdsError.invalidTCoordBatch(expected: batch, actual: tCoord.dim(0))
      }
      var outputs: [MLXArray] = []
      outputs.reserveCapacity(batch)
      for index in 0..<batch {
        var t = tCoord[index].reshaped(-1).asType(.int32)
        if t.size == 0 {
          t = MLXArray(0..<1).asType(.int32)
        }
        let coords = cartesianProd([t, h, w, l])
        outputs.append(coords)
      }
      return MLX.stacked(outputs, axis: 0)
    }

    let t = MLXArray(0..<1).asType(.int32)
    let coords = cartesianProd([t, h, w, l])
    return expandToBatch(coords, batch: batch)
  }

  public static func prepareLatentIds(
    _ latents: MLXArray
  ) throws -> MLXArray {
    guard latents.ndim == 4 else {
      throw Flux2PositionIdsError.invalidLatentsShape(latents.shape)
    }

    let batch = latents.dim(0)
    let height = latents.dim(2)
    let width = latents.dim(3)

    let t = MLXArray(0..<1).asType(.int32)
    let h = MLXArray(0..<height).asType(.int32)
    let w = MLXArray(0..<width).asType(.int32)
    let l = MLXArray(0..<1).asType(.int32)

    let coords = cartesianProd([t, h, w, l])
    return expandToBatch(coords, batch: batch)
  }

  public static func prepareImageIds(
    _ imageLatents: [MLXArray],
    scale: Int = 10
  ) throws -> MLXArray {
    guard !imageLatents.isEmpty else {
      throw Flux2PositionIdsError.emptyImageLatents
    }

    var idsList: [MLXArray] = []
    idsList.reserveCapacity(imageLatents.count)

    for (index, latents) in imageLatents.enumerated() {
      guard latents.ndim == 4 else {
        throw Flux2PositionIdsError.invalidImageLatentsShape(index: index, shape: latents.shape)
      }
      let batch = latents.dim(0)
      guard batch == 1 else {
        throw Flux2PositionIdsError.invalidImageLatentsBatch(index: index, batch: batch)
      }

      let height = latents.dim(2)
      let width = latents.dim(3)
      let t = MLXArray([Int32(scale + scale * index)])
      let h = MLXArray(0..<height).asType(.int32)
      let w = MLXArray(0..<width).asType(.int32)
      let l = MLXArray(0..<1).asType(.int32)

      let coords = cartesianProd([t, h, w, l])
      idsList.append(coords)
    }

    let concatenated = idsList.count == 1 ? idsList[0] : MLX.concatenated(idsList, axis: 0)
    return concatenated.reshaped(1, concatenated.dim(0), concatenated.dim(1))
  }

  private static func cartesianProd(_ arrays: [MLXArray]) -> MLXArray {
    precondition(!arrays.isEmpty, "cartesianProd requires at least one array.")
    for array in arrays {
      precondition(array.ndim == 1, "cartesianProd expects 1D arrays.")
    }

    let grids = meshGrid(arrays, indexing: .ij)
    let stacked = MLX.stacked(grids, axis: -1)
    return stacked.reshaped(-1, arrays.count)
  }

  private static func expandToBatch(_ coords: MLXArray, batch: Int) -> MLXArray {
    let expanded = coords.reshaped(1, coords.dim(0), coords.dim(1))
    if batch == 1 { return expanded }
    return MLX.broadcast(expanded, to: [batch, coords.dim(0), coords.dim(1)])
  }
}
