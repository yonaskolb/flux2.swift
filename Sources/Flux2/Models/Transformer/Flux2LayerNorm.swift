import MLX
import MLXNN

final class Flux2LayerNorm: Module, UnaryLayer {
  private let eps: Float

  init(eps: Float = 1e-5) {
    self.eps = eps
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    MLXFast.layerNorm(x, eps: eps)
  }
}
