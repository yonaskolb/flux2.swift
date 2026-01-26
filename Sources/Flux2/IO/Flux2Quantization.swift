import Foundation
import MLX
import MLXNN

public enum Flux2QuantizationMode: String, Codable, Sendable {
  case affine
  case mxfp4

  public var mlxMode: QuantizationMode {
    switch self {
    case .affine:
      return .affine
    case .mxfp4:
      return .mxfp4
    }
  }
}

public struct Flux2QuantizationSpec: Codable, Sendable {
  public var groupSize: Int
  public var bits: Int
  public var mode: Flux2QuantizationMode

  public init(groupSize: Int = 64, bits: Int = 8, mode: Flux2QuantizationMode = .affine) {
    self.groupSize = groupSize
    self.bits = bits
    self.mode = mode
  }

  enum CodingKeys: String, CodingKey {
    case groupSize = "group_size"
    case bits
    case mode
  }
}

public struct Flux2QuantizationManifest: Codable, Sendable {
  public var modelId: String?
  public var revision: String?
  public var groupSize: Int
  public var bits: Int
  public var mode: String
  public var layers: [QuantizedLayerInfo]

  public struct QuantizedLayerInfo: Codable, Sendable {
    public var name: String
    public var shape: [Int]
    public var inDim: Int
    public var outDim: Int
    public var file: String

    enum CodingKeys: String, CodingKey {
      case name
      case shape
      case inDim = "in_dim"
      case outDim = "out_dim"
      case file
    }
  }

  enum CodingKeys: String, CodingKey {
    case modelId = "model_id"
    case revision
    case groupSize = "group_size"
    case bits
    case mode
    case layers
  }

  public static func load(from url: URL) throws -> Flux2QuantizationManifest {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(Flux2QuantizationManifest.self, from: data)
  }
}

public enum Flux2QuantizationError: Error, LocalizedError {
  case alreadyQuantized(URL)
  case invalidGroupSize(Int)
  case invalidBits(Int)
  case outputDirectoryExists(URL)
  case outputDirectoryCreationFailed(URL)
  case sourceSnapshotMissing(URL)
  case missingComponentDirectory(Flux2WeightComponent, URL)
  case noSafetensorsFound(Flux2WeightComponent, URL)
  case quantizationFailed(String)

  public var errorDescription: String? {
    switch self {
    case .alreadyQuantized(let url):
      return "Snapshot already appears to be quantized (found quantization.json): \(url.path)"
    case .invalidGroupSize(let size):
      return "Invalid group size: \(size). Supported sizes: 32, 64, 128"
    case .invalidBits(let bits):
      return "Invalid bits: \(bits). Supported values: 4, 8"
    case .outputDirectoryExists(let url):
      return "Output directory already exists: \(url.path)"
    case .outputDirectoryCreationFailed(let url):
      return "Failed to create output directory: \(url.path)"
    case .sourceSnapshotMissing(let url):
      return "Snapshot directory not found: \(url.path)"
    case .missingComponentDirectory(let component, let url):
      return "Missing component directory '\(component.rawValue)': \(url.path)"
    case .noSafetensorsFound(let component, let url):
      return "No safetensors found for component '\(component.rawValue)' in \(url.path)"
    case .quantizationFailed(let reason):
      return "Quantization failed: \(reason)"
    }
  }
}

public struct Flux2Quantizer {
  public static let supportedGroupSizes: Set<Int> = [32, 64, 128]
  public static let supportedBits: Set<Int> = [4, 8]

  public static func hasQuantization(at directory: URL) -> Bool {
    FileManager.default.fileExists(atPath: directory.appendingPathComponent("quantization.json").path)
  }

  public static func loadManifest(from directory: URL) throws -> Flux2QuantizationManifest? {
    let manifestURL = directory.appendingPathComponent("quantization.json")
    guard FileManager.default.fileExists(atPath: manifestURL.path) else {
      return nil
    }
    return try Flux2QuantizationManifest.load(from: manifestURL)
  }

  public static func applyQuantization(
    to model: Module,
    manifest: Flux2QuantizationManifest,
    weights: [String: MLXArray]
  ) {
    let groupSize = manifest.groupSize
    let bits = manifest.bits
    let mode: QuantizationMode = manifest.mode == "mxfp4" ? .mxfp4 : .affine

    let replacements = model
      .leafModules()
      .flattened()
      .compactMap { (path, module) -> (String, Module)? in
        guard let linear = module as? Linear else { return nil }
        guard !(linear is QuantizedLinear) else { return nil }
        guard let scales = weights["\(path).scales"] else { return nil }
        guard let weight = weights["\(path).weight"] else { return nil }

        let biases = weights["\(path).biases"]
        let quantized = QuantizedLinear(
          weight: weight,
          bias: linear.bias,
          scales: scales,
          biases: biases,
          groupSize: groupSize,
          bits: bits,
          mode: mode
        )
        quantized.freeze()
        return (path, quantized)
      }

    guard !replacements.isEmpty else { return }
    model.update(modules: ModuleChildren.unflattened(replacements))
  }

  public static func quantizeAndSave(
    from sourceSnapshot: URL,
    to outputSnapshot: URL,
    spec: Flux2QuantizationSpec,
    modelId: String? = nil,
    revision: String? = nil,
    overwrite: Bool = false,
    verbose: Bool = false
  ) throws {
    guard supportedGroupSizes.contains(spec.groupSize) else {
      throw Flux2QuantizationError.invalidGroupSize(spec.groupSize)
    }
    guard supportedBits.contains(spec.bits) else {
      throw Flux2QuantizationError.invalidBits(spec.bits)
    }

    let fm = FileManager.default
    let resolvedSource = sourceSnapshot.resolvingSymlinksInPath()
    var isDir: ObjCBool = false
    guard fm.fileExists(atPath: resolvedSource.path, isDirectory: &isDir), isDir.boolValue else {
      throw Flux2QuantizationError.sourceSnapshotMissing(resolvedSource)
    }

    if hasQuantization(at: resolvedSource) {
      throw Flux2QuantizationError.alreadyQuantized(resolvedSource)
    }

    if fm.fileExists(atPath: outputSnapshot.path) {
      if overwrite {
        try fm.removeItem(at: outputSnapshot)
      } else {
        throw Flux2QuantizationError.outputDirectoryExists(outputSnapshot)
      }
    }

    do {
      try fm.createDirectory(at: outputSnapshot, withIntermediateDirectories: true)
    } catch {
      throw Flux2QuantizationError.outputDirectoryCreationFailed(outputSnapshot)
    }

    // Root files
    try copyFileIfExists(
      from: resolvedSource.appendingPathComponent("model_index.json"),
      to: outputSnapshot.appendingPathComponent("model_index.json")
    )

    // Root directories needed for inference
    try copyDirectoryIfExists(
      from: resolvedSource.appendingPathComponent("scheduler"),
      to: outputSnapshot.appendingPathComponent("scheduler"),
      overwrite: overwrite
    )
    try copyDirectoryIfExists(
      from: resolvedSource.appendingPathComponent("tokenizer"),
      to: outputSnapshot.appendingPathComponent("tokenizer"),
      overwrite: overwrite
    )
    try copyDirectoryIfExists(
      from: resolvedSource.appendingPathComponent("vae"),
      to: outputSnapshot.appendingPathComponent("vae"),
      overwrite: overwrite
    )

    var quantizedLayers: [Flux2QuantizationManifest.QuantizedLayerInfo] = []

    // Quantize transformer + text_encoder weights in-place into a new snapshot.
    try quantizeComponent(
      component: .transformer,
      sourceSnapshot: resolvedSource,
      outputSnapshot: outputSnapshot,
      spec: spec,
      manifestLayers: &quantizedLayers,
      verbose: verbose
    )
    try quantizeComponent(
      component: .textEncoder,
      sourceSnapshot: resolvedSource,
      outputSnapshot: outputSnapshot,
      spec: spec,
      manifestLayers: &quantizedLayers,
      verbose: verbose
    )

    let manifest = Flux2QuantizationManifest(
      modelId: modelId,
      revision: revision,
      groupSize: spec.groupSize,
      bits: spec.bits,
      mode: spec.mode.rawValue,
      layers: quantizedLayers
    )

    let manifestURL = outputSnapshot.appendingPathComponent("quantization.json")
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let manifestData = try encoder.encode(manifest)
    try manifestData.write(to: manifestURL)
  }

  private static func quantizeComponent(
    component: Flux2WeightComponent,
    sourceSnapshot: URL,
    outputSnapshot: URL,
    spec: Flux2QuantizationSpec,
    manifestLayers: inout [Flux2QuantizationManifest.QuantizedLayerInfo],
    verbose: Bool
  ) throws {
    let fm = FileManager.default
    let sourceDir = sourceSnapshot.appendingPathComponent(component.rawValue)
    let outputDir = outputSnapshot.appendingPathComponent(component.rawValue)

    guard fm.fileExists(atPath: sourceDir.path) else {
      throw Flux2QuantizationError.missingComponentDirectory(component, sourceDir)
    }

    try fm.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let contents = try fm.contentsOfDirectory(at: sourceDir, includingPropertiesForKeys: nil)
    let safetensorsFiles = contents
      .filter { $0.pathExtension == "safetensors" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !safetensorsFiles.isEmpty else {
      throw Flux2QuantizationError.noSafetensorsFound(component, sourceDir)
    }

    // Copy non-weight files (configs, tokenizer json, etc.). Avoid copying shard index JSON
    // because it won't reflect the added *.scales/*.biases tensors.
    for item in contents where item.pathExtension != "safetensors" {
      if item.lastPathComponent.lowercased().hasSuffix(".safetensors.index.json") {
        continue
      }
      let dest = outputDir.appendingPathComponent(item.lastPathComponent)
      try copyItem(at: item, to: dest, overwrite: true)
    }

    for input in safetensorsFiles {
      let output = outputDir.appendingPathComponent(input.lastPathComponent)
      try quantizeSafetensorsFile(
        inputURL: input,
        outputURL: output,
        component: component,
        spec: spec,
        manifestLayers: &manifestLayers,
        verbose: verbose
      )
    }
  }

  private static func quantizeSafetensorsFile(
    inputURL: URL,
    outputURL: URL,
    component: Flux2WeightComponent,
    spec: Flux2QuantizationSpec,
    manifestLayers: inout [Flux2QuantizationManifest.QuantizedLayerInfo],
    verbose: Bool
  ) throws {
    let reader = try SafeTensorsReader(fileURL: inputURL)
    let metadata = reader.allMetadata().sorted { $0.name < $1.name }

    var output: [String: MLXArray] = [:]
    output.reserveCapacity(metadata.count + (metadata.count / 4))

    var quantizedCount = 0
    for meta in metadata {
      let key = meta.name
      let tensor = try reader.tensor(named: key)

      guard shouldQuantizeTensor(named: key, meta: meta) else {
        output[key] = tensor
        continue
      }

      let outDim = tensor.dim(0)
      let inDim = tensor.dim(1)
      guard inDim % spec.groupSize == 0 else {
        output[key] = tensor
        continue
      }

      let base = String(key.dropLast(".weight".count))
      var floatTensor = tensor
      if floatTensor.dtype != .float32 {
        floatTensor = floatTensor.asType(.float32)
      }

      let (wq, scales, biases) = MLX.quantized(
        floatTensor,
        groupSize: spec.groupSize,
        bits: spec.bits,
        mode: spec.mode.mlxMode
      )

      output[key] = wq
      output["\(base).scales"] = scales
      if let biases {
        output["\(base).biases"] = biases
      }

      manifestLayers.append(.init(
        name: "\(component.rawValue).\(base)",
        shape: [outDim, inDim],
        inDim: inDim,
        outDim: outDim,
        file: "\(component.rawValue)/\(outputURL.lastPathComponent)"
      ))
      quantizedCount += 1
    }

    if verbose {
      print("[quantize] \(component.rawValue)/\(inputURL.lastPathComponent): quantized \(quantizedCount) tensors")
    }

    try MLX.save(arrays: output, metadata: [:], url: outputURL)
  }

  private static func shouldQuantizeTensor(named name: String, meta: SafeTensorMetadata) -> Bool {
    guard name.hasSuffix(".weight") else { return false }
    guard meta.shape.count == 2 else { return false }

    let lower = name.lowercased()
    if lower.hasSuffix("embed_tokens.weight") || lower.contains(".embed_tokens.") {
      return false
    }
    if lower.hasSuffix("token_embedding.weight") || lower.contains(".token_embedding.") {
      return false
    }
    if lower.contains("embedding") {
      return false
    }
    if lower.contains("norm") || lower.contains("layernorm") || lower.contains("rmsnorm") {
      return false
    }
    return true
  }

  private static func copyFileIfExists(from src: URL, to dst: URL) throws {
    let fm = FileManager.default
    guard fm.fileExists(atPath: src.path) else { return }
    try copyItem(at: src, to: dst, overwrite: true)
  }

  private static func copyDirectoryIfExists(from src: URL, to dst: URL, overwrite: Bool) throws {
    let fm = FileManager.default
    var isDir: ObjCBool = false
    guard fm.fileExists(atPath: src.path, isDirectory: &isDir), isDir.boolValue else { return }
    try copyDirectoryResolvingSymlinks(from: src, to: dst, overwrite: overwrite)
  }

  private static func copyItem(at src: URL, to dst: URL, overwrite: Bool) throws {
    let fm = FileManager.default
    if fm.fileExists(atPath: dst.path) {
      if overwrite {
        try fm.removeItem(at: dst)
      } else {
        return
      }
    }

    let resolved = src.resolvingSymlinksInPath()
    var isDir: ObjCBool = false
    if fm.fileExists(atPath: resolved.path, isDirectory: &isDir), isDir.boolValue {
      try copyDirectoryResolvingSymlinks(from: resolved, to: dst, overwrite: overwrite)
      return
    }

    try fm.copyItem(at: resolved, to: dst)
  }

  private static func copyDirectoryResolvingSymlinks(from src: URL, to dst: URL, overwrite: Bool) throws {
    let fm = FileManager.default
    if fm.fileExists(atPath: dst.path) {
      if overwrite {
        try fm.removeItem(at: dst)
      } else {
        return
      }
    }

    try fm.createDirectory(at: dst, withIntermediateDirectories: true)

    guard let enumerator = fm.enumerator(at: src, includingPropertiesForKeys: [.isDirectoryKey], options: []) else {
      return
    }

    for case let item as URL in enumerator {
      let relative = item.path.replacingOccurrences(of: src.path + "/", with: "")
      let dest = dst.appendingPathComponent(relative)

      let resolvedItem = item.resolvingSymlinksInPath()
      var isDir: ObjCBool = false
      _ = fm.fileExists(atPath: resolvedItem.path, isDirectory: &isDir)
      if isDir.boolValue {
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)
        continue
      }

      try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
      if fm.fileExists(atPath: dest.path) {
        try fm.removeItem(at: dest)
      }
      try fm.copyItem(at: resolvedItem, to: dest)
    }
  }
}
