import MLX

public enum Flux2EvaluationPolicy: Sendable {
  case aggressive
  case deferred
}

extension Flux2EvaluationPolicy {
  func evalIfNeeded(_ array: MLXArray) {
    if self == .aggressive {
      MLX.eval(array)
    }
  }
}

