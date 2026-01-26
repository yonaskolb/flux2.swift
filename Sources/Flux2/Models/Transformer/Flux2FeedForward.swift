import MLX
import MLXNN

final class Flux2SwiGLU: Module, UnaryLayer {
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let (x1, x2) = x.split(axis: -1)
    return silu(x1) * x2
  }
}

final class Flux2FeedForward: Module {
  @ModuleInfo(key: "linear_in") private var linearIn: Linear
  @ModuleInfo(key: "linear_out") private var linearOut: Linear

  private let activation = Flux2SwiGLU()

  init(
    dim: Int,
    dimOut: Int? = nil,
    mult: Float = 3.0,
    innerDim: Int? = nil,
    bias: Bool = false
  ) {
    let resolvedInner = innerDim ?? Int(Float(dim) * mult)
    let resolvedOut = dimOut ?? dim
    _linearIn.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)
    _linearOut.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let projected = linearIn(x)
    let activated = activation(projected)
    return linearOut(activated)
  }
}
