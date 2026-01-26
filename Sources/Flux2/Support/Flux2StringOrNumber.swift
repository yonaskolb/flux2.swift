import Foundation

public enum Flux2StringOrNumber: Codable, Equatable, Sendable {
  case string(String)
  case int(Int)
  case float(Float)
  case ints([Int])
  case floats([Float])
  case bool(Bool)

  public init(from decoder: Decoder) throws {
    let values = try decoder.singleValueContainer()

    if let v = try? values.decode(Int.self) {
      self = .int(v)
    } else if let v = try? values.decode(Float.self) {
      self = .float(v)
    } else if let v = try? values.decode([Int].self) {
      self = .ints(v)
    } else if let v = try? values.decode([Float].self) {
      self = .floats(v)
    } else if let v = try? values.decode(Bool.self) {
      self = .bool(v)
    } else {
      let v = try values.decode(String.self)
      self = .string(v)
    }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let v):
      try container.encode(v)
    case .int(let v):
      try container.encode(v)
    case .float(let v):
      try container.encode(v)
    case .ints(let v):
      try container.encode(v)
    case .floats(let v):
      try container.encode(v)
    case .bool(let v):
      try container.encode(v)
    }
  }

  public func asFloat() -> Float? {
    switch self {
    case .string:
      return nil
    case .int(let v):
      return Float(v)
    case .float(let v):
      return v
    case .ints(let array):
      return array.count == 1 ? Float(array[0]) : nil
    case .floats(let array):
      return array.count == 1 ? array[0] : nil
    case .bool(let v):
      return v ? 1.0 : 0.0
    }
  }

  public func asFloats() -> [Float]? {
    switch self {
    case .string:
      return nil
    case .int(let v):
      return [Float(v)]
    case .float(let v):
      return [v]
    case .ints(let array):
      return array.map { Float($0) }
    case .floats(let array):
      return array
    case .bool(let v):
      return [v ? 1.0 : 0.0]
    }
  }

  public func asInt() -> Int? {
    switch self {
    case .string:
      return nil
    case .int(let v):
      return v
    case .float(let v):
      return Int(v)
    case .ints(let array):
      return array.count == 1 ? array[0] : nil
    case .floats(let array):
      return array.count == 1 ? Int(array[0]) : nil
    case .bool(let v):
      return v ? 1 : 0
    }
  }

  public func asString() -> String? {
    switch self {
    case .string(let v):
      return v
    case .int:
      return nil
    case .float:
      return nil
    case .ints:
      return nil
    case .floats:
      return nil
    case .bool:
      return nil
    }
  }
}
