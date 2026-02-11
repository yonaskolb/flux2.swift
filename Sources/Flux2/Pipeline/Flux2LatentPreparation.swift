import Foundation
import MLX

public struct Flux2PreparedLatents {
  public let latents: MLXArray
  public let ids: MLXArray
  public let height: Int
  public let width: Int
}

public struct Flux2PreparedImageLatents {
  public let latents: MLXArray
  public let ids: MLXArray
}

public enum Flux2LatentPreparationError: Error, LocalizedError {
  case failedToCreateLatents
  case emptyImages
  case invalidImageShape(index: Int, shape: [Int])

  public var errorDescription: String? {
    switch self {
    case .failedToCreateLatents:
      return "Failed to create latent tensor."
    case .emptyImages:
      return "Image list must not be empty."
    case .invalidImageShape(let index, let shape):
      return "Image at index \(index) has invalid shape \(shape); expected 4 dimensions."
    }
  }
}

public enum Flux2LatentPreparation {
  public static func vaeScaleFactor(for vae: Flux2AutoencoderKL) -> Int {
    let blocks = max(vae.configuration.blockOutChannels.count, 1)
    return 1 << (blocks - 1)
  }

  public static func adjustedLatentSize(
    height: Int,
    width: Int,
    vaeScaleFactor: Int,
    patchSize: (Int, Int)
  ) -> (height: Int, width: Int) {
    let divisorH = max(vaeScaleFactor * patchSize.0, 1)
    let divisorW = max(vaeScaleFactor * patchSize.1, 1)
    let adjustedHeight = patchSize.0 * (height / divisorH)
    let adjustedWidth = patchSize.1 * (width / divisorW)
    return (adjustedHeight, adjustedWidth)
  }

  public static func prepareLatents(
    batchSize: Int,
    numLatentChannels: Int,
    height: Int,
    width: Int,
    vae: Flux2AutoencoderKL,
    dtype: DType,
    latents: MLXArray? = nil
  ) throws -> Flux2PreparedLatents {
    let patchSize = vae.configuration.patchSize
    let patchArea = vae.configuration.patchSizeArea
    let scaleFactor = vaeScaleFactor(for: vae)
    let adjusted = adjustedLatentSize(
      height: height,
      width: width,
      vaeScaleFactor: scaleFactor,
      patchSize: patchSize
    )

    var latentInput = latents
    if latentInput == nil {
      let latentHeight = adjusted.height / patchSize.0
      let latentWidth = adjusted.width / patchSize.1
      let shape = [
        batchSize,
        numLatentChannels * patchArea,
        latentHeight,
        latentWidth
      ]
      latentInput = MLXRandom.normal(shape, dtype: dtype)
    }

    guard var latents = latentInput else {
      throw Flux2LatentPreparationError.failedToCreateLatents
    }
    if latents.dtype != dtype {
      latents = latents.asType(dtype)
    }

    let latentIds = try Flux2PositionIds.prepareLatentIds(latents)
    let packed = try Flux2LatentUtils.packLatents(latents)

    return Flux2PreparedLatents(
      latents: packed,
      ids: latentIds,
      height: adjusted.height,
      width: adjusted.width
    )
  }

  public static func prepareImageLatents(
    images: [MLXArray],
    batchSize: Int,
    vae: Flux2AutoencoderKL,
    dtype: DType,
    imageIdScale: Int = 10
  ) throws -> Flux2PreparedImageLatents {
    guard !images.isEmpty else {
      throw Flux2LatentPreparationError.emptyImages
    }

    var imageLatents: [MLXArray] = []
    imageLatents.reserveCapacity(images.count)

    var packedLatents: [MLXArray] = []
    packedLatents.reserveCapacity(images.count)

    for (index, image) in images.enumerated() {
      guard image.ndim == 4 else {
        throw Flux2LatentPreparationError.invalidImageShape(index: index, shape: image.shape)
      }
      var input = image
      if input.dtype != dtype {
        input = input.asType(dtype)
      }

      let latentDist = vae.encode(input)
      let latents = try Flux2LatentUtils.patchifyAndNormalize(latentDist.mode(), vae: vae)
      imageLatents.append(latents)

      let packed = try Flux2LatentUtils.packLatents(latents)[0]
      packedLatents.append(packed)
    }

    let ids = try Flux2PositionIds.prepareImageIds(imageLatents, scale: imageIdScale)
    let concatenated = packedLatents.count == 1
      ? packedLatents[0]
      : MLX.concatenated(packedLatents, axis: 0)

    var packed = concatenated.expandedDimensions(axis: 0)
    var idsBatched = ids

    if batchSize > 1 {
      packed = MLX.broadcast(packed, to: [batchSize, packed.dim(1), packed.dim(2)])
      idsBatched = MLX.broadcast(ids, to: [batchSize, ids.dim(1), ids.dim(2)])
    }

    return Flux2PreparedImageLatents(latents: packed, ids: idsBatched)
  }
}
