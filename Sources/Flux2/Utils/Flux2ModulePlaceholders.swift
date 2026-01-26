import MLX
import MLXNN

enum Flux2ModulePlaceholders {
  private static let scalarDType: DType = .float32

  static func linear(bias: Bool = false) -> Linear {
    let weight = MLX.zeros([1, 1], dtype: scalarDType)
    let biasArray = bias ? MLX.zeros([1], dtype: scalarDType) : nil
    return Linear(weight: weight, bias: biasArray)
  }

  static func embedding() -> Embedding {
    Embedding(weight: MLX.zeros([1, 1], dtype: scalarDType))
  }
}

