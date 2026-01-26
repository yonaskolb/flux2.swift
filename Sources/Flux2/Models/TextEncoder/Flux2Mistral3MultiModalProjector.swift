import Foundation
import MLX
import MLXNN

final class Flux2Mistral3PatchMerger: Module {
  private let spatialMergeSize: Int
  private let patchSize: Int

  @ModuleInfo(key: "merging_layer") private var mergingLayer: Linear

  init(hiddenSize: Int, patchSize: Int, spatialMergeSize: Int) {
    self.patchSize = patchSize
    self.spatialMergeSize = spatialMergeSize
    _mergingLayer.wrappedValue = Flux2ModulePlaceholders.linear(bias: false)
    super.init()
  }

  func callAsFunction(_ imageFeatures: MLXArray, imageSizes: [(height: Int, width: Int)]) -> MLXArray {
    precondition(imageFeatures.ndim == 2, "imageFeatures must be [tokens, dim].")
    let d = imageFeatures.dim(1)

    var slices: [MLXArray] = []
    slices.reserveCapacity(imageSizes.count)

    var offset = 0
    for size in imageSizes {
      let h = max(size.height / max(patchSize, 1), 1)
      let w = max(size.width / max(patchSize, 1), 1)
      let tokens = h * w

      let end = offset + tokens
      let imageTokens = imageFeatures[offset..<end, 0...]
      offset = end

      let merge = max(spatialMergeSize, 1)
      let hBlocks = max(h / merge, 1)
      let wBlocks = max(w / merge, 1)

      var grid = imageTokens.reshaped(h, w, d)
      grid = grid.reshaped(hBlocks, merge, wBlocks, merge, d)
      grid = grid.transposed(0, 2, 4, 1, 3)
      let flattened = grid.reshaped(hBlocks * wBlocks, d * merge * merge)
      slices.append(flattened)
    }

    let concatenated = slices.count == 1 ? slices[0] : MLX.concatenated(slices, axis: 0)
    return mergingLayer(concatenated)
  }
}

final class Flux2Mistral3MultiModalProjector: Module {
  private let spatialMergeSize: Int
  private let patchSize: Int
  private let hiddenAct: String

  @ModuleInfo(key: "norm") private var norm: RMSNorm
  @ModuleInfo(key: "patch_merger") private var patchMerger: Flux2Mistral3PatchMerger
  @ModuleInfo(key: "linear_1") private var linear1: Linear
  @ModuleInfo(key: "linear_2") private var linear2: Linear

  init(vlmConfiguration: Flux2Mistral3VLMConfiguration) {
    spatialMergeSize = vlmConfiguration.spatialMergeSize
    patchSize = vlmConfiguration.visionConfig.patchSize
    hiddenAct = vlmConfiguration.projectorHiddenAct

    _norm.wrappedValue = RMSNorm(dimensions: vlmConfiguration.visionConfig.hiddenSize, eps: vlmConfiguration.textConfig.rmsNormEps)
    _patchMerger.wrappedValue = Flux2Mistral3PatchMerger(
      hiddenSize: vlmConfiguration.visionConfig.hiddenSize,
      patchSize: patchSize,
      spatialMergeSize: spatialMergeSize
    )
    _linear1.wrappedValue = Flux2ModulePlaceholders.linear(bias: vlmConfiguration.multimodalProjectorBias)
    _linear2.wrappedValue = Flux2ModulePlaceholders.linear(bias: vlmConfiguration.multimodalProjectorBias)

    super.init()
  }

  func callAsFunction(_ imageFeatures: MLXArray, imageSizes: [(height: Int, width: Int)]) -> MLXArray {
    var features = norm(imageFeatures)
    features = patchMerger(features, imageSizes: imageSizes)
    features = linear1(features)
    features = activation(features)
    features = linear2(features)
    return features
  }

  private func activation(_ x: MLXArray) -> MLXArray {
    switch hiddenAct.lowercased() {
    case "gelu":
      return gelu(x)
    default:
      return gelu(x)
    }
  }
}

