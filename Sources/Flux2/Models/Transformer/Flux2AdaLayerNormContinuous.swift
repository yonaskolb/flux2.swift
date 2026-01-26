import MLX
import MLXNN

final class Flux2AdaLayerNormContinuous: Module {
  @ModuleInfo(key: "linear") private var linear: Linear
  private let norm: Flux2LayerNorm
  private let elementwiseAffine: Bool

  init(
    embeddingDim: Int,
    conditioningEmbeddingDim: Int,
    elementwiseAffine: Bool = false,
    eps: Float = 1e-5,
    bias: Bool = false
  ) {
    self.elementwiseAffine = elementwiseAffine
    precondition(!elementwiseAffine, "Flux2 uses elementwise_affine=false.")
    _linear.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)
    norm = Flux2LayerNorm(eps: eps)
  }

  func callAsFunction(_ x: MLXArray, conditioningEmbedding: MLXArray) -> MLXArray {
    var emb = silu(conditioningEmbedding).asType(x.dtype)
    emb = linear(emb)

    let parts = split(emb, parts: 2, axis: 1)
    var scale = parts[0].expandedDimensions(axis: 1)
    let shift = parts[1].expandedDimensions(axis: 1)
    let one = MLXArray(1.0).asType(scale.dtype)
    scale = one + scale

    let normed = norm(x)
    return normed * scale + shift
  }
}
