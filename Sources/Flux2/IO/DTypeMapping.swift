import MLX

public extension DType {
  static func fromCLI(_ value: String) -> DType? {
    switch value.lowercased() {
    case "float16", "f16", "fp16":
      return .float16
    case "float32", "f32", "fp32":
      return .float32
    case "bfloat16", "bf16":
      return .bfloat16
    default:
      return nil
    }
  }

  var cliName: String {
    switch self {
    case .float16:
      return "float16"
    case .float32:
      return "float32"
    case .bfloat16:
      return "bfloat16"
    case .float64:
      return "float64"
    case .int64:
      return "int64"
    case .int32:
      return "int32"
    case .int16:
      return "int16"
    case .int8:
      return "int8"
    case .uint64:
      return "uint64"
    case .uint32:
      return "uint32"
    case .uint16:
      return "uint16"
    case .uint8:
      return "uint8"
    case .bool:
      return "bool"
    default:
      return "unknown"
    }
  }
}
