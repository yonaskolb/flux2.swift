import Foundation
import MLX

public enum Flux2WeightComponent: String, CaseIterable, Sendable {
  case transformer
  case textEncoder = "text_encoder"
  case vae
}

public enum Flux2WeightsLoaderError: Error {
  case componentDirectoryMissing(Flux2WeightComponent, URL)
  case noSafetensorsFound(Flux2WeightComponent, URL)
}

public struct Flux2WeightsLoader: Sendable {
  public let snapshot: URL

  public init(snapshot: URL) {
    self.snapshot = snapshot
  }

  public func listSafetensors(component: Flux2WeightComponent) throws -> [URL] {
    let componentDir = snapshot.appendingPathComponent(component.rawValue)
    guard FileManager.default.fileExists(atPath: componentDir.path) else {
      throw Flux2WeightsLoaderError.componentDirectoryMissing(component, componentDir)
    }

    let contents = try FileManager.default.contentsOfDirectory(at: componentDir, includingPropertiesForKeys: nil)
    let files = contents.filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !files.isEmpty else {
      throw Flux2WeightsLoaderError.noSafetensorsFound(component, componentDir)
    }
    return files
  }

  public func load(component: Flux2WeightComponent, dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    try load(component: component, dtype: dtype, filter: nil)
  }

  public func load(
    component: Flux2WeightComponent,
    dtype: DType? = .bfloat16,
    filter: ((String) -> Bool)?
  ) throws -> [String: MLXArray] {
    let files = try listSafetensors(component: component)
    var tensors: [String: MLXArray] = [:]
    for url in files {
      let fileTensors = try loadArrays(url: url, stream: .cpu)
      let tensorNames = Set(fileTensors.keys)
      for (name, value) in fileTensors {
        if let filter, !filter(name) {
          continue
        }
        var tensor = value
        if let dtype,
           tensor.dtype != dtype,
           shouldCastLoadedTensor(name: name, tensorDType: tensor.dtype, targetDType: dtype, availableNames: tensorNames) {
          tensor = tensor.asType(dtype, stream: .cpu)
        }
        tensors[name] = tensor
      }
    }
    return tensors
  }

  private func shouldCastLoadedTensor(
    name: String,
    tensorDType: DType,
    targetDType: DType,
    availableNames: Set<String>
  ) -> Bool {
    guard isFloatingDType(tensorDType) else { return false }
    guard targetDType != tensorDType else { return false }

    guard Flux2Quantizer.hasQuantization(at: snapshot) else {
      return true
    }

    if name.hasSuffix(".scales") || name.hasSuffix(".biases") {
      return false
    }
    if name.hasSuffix(".weight") {
      let base = String(name.dropLast(".weight".count))
      if availableNames.contains("\(base).scales") {
        return false
      }
    }
    return true
  }

  private func isFloatingDType(_ dtype: DType) -> Bool {
    switch dtype {
    case .float16, .float32, .float64, .bfloat16:
      return true
    default:
      return false
    }
  }

  public func loadAll(dtype: DType? = .bfloat16) throws -> [String: MLXArray] {
    var tensors: [String: MLXArray] = [:]
    for component in Flux2WeightComponent.allCases {
      let componentTensors = try load(component: component, dtype: dtype)
      for (key, value) in componentTensors {
        tensors["\(component.rawValue).\(key)"] = value
      }
    }
    return tensors
  }
}
