import MLX
import MLXNN

typealias Flux2ModulationParams = (shift: MLXArray, scale: MLXArray, gate: MLXArray)

final class Flux2Modulation: Module {
  let modParamSets: Int
  @ModuleInfo(key: "linear") private var linear: Linear

  init(dim: Int, modParamSets: Int = 2, bias: Bool = false) {
    self.modParamSets = modParamSets
    _linear.wrappedValue = Flux2ModulePlaceholders.linear(bias: bias)
  }

  func callAsFunction(_ temb: MLXArray) -> [Flux2ModulationParams] {
    var mod = silu(temb)
    mod = linear(mod)

    if mod.ndim == 2 {
      mod = mod.expandedDimensions(axis: 1)
    }

    let chunks = mod.split(parts: 3 * modParamSets, axis: -1)
    var params: [Flux2ModulationParams] = []
    params.reserveCapacity(modParamSets)

    for index in 0..<modParamSets {
      let base = index * 3
      params.append((chunks[base], chunks[base + 1], chunks[base + 2]))
    }

    return params
  }
}
