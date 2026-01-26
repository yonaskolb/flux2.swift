import Flux2
import MLX
import MLXNN
import XCTest

final class Flux2QuantizationTests: XCTestCase {
  func testQuantizeSnapshotWritesManifestAndLoadsQuantizedWeights() throws {
    let fm = FileManager.default
    let root = fm.temporaryDirectory.appendingPathComponent("flux2-quant-\(UUID().uuidString)")
    let source = root.appendingPathComponent("source")
    let output = root.appendingPathComponent("output")

    try fm.createDirectory(at: source, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }

    try Data("{\"_class_name\":\"Flux2Pipeline\"}".utf8).write(to: source.appendingPathComponent("model_index.json"))

    try createMinimalDirectory(
      at: source.appendingPathComponent("scheduler"),
      files: ["scheduler_config.json": "{}"]
    )
    try createMinimalDirectory(
      at: source.appendingPathComponent("tokenizer"),
      files: ["tokenizer_config.json": "{}"]
    )
    try createMinimalDirectory(
      at: source.appendingPathComponent("vae"),
      files: ["config.json": "{}"]
    )

    let transformerDir = source.appendingPathComponent("transformer")
    try createMinimalDirectory(at: transformerDir, files: ["config.json": "{}"])
    try writeWeights(
      url: transformerDir.appendingPathComponent("model.safetensors"),
      arrays: [
        // Quantized (in_dim = 64, group_size = 64)
        "context_embedder.weight": MLXRandom.normal([8, 64], dtype: .float32),
        // Not quantized (in_dim not divisible by 64) - should be cast to bf16 when loading.
        "small.weight": MLXRandom.normal([8, 63], dtype: .float32),
      ]
    )

    let textEncoderDir = source.appendingPathComponent("text_encoder")
    try createMinimalDirectory(at: textEncoderDir, files: ["config.json": "{}"])
    try writeWeights(
      url: textEncoderDir.appendingPathComponent("model.safetensors"),
      arrays: [
        "linear.weight": MLXRandom.normal([8, 64], dtype: .float32),
      ]
    )

    let spec = Flux2QuantizationSpec(groupSize: 64, bits: 8, mode: .affine)
    try Flux2Quantizer.quantizeAndSave(
      from: source,
      to: output,
      spec: spec,
      modelId: "test",
      revision: "main"
    )

    XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("quantization.json").path))

    let quantizedTransformer = output.appendingPathComponent("transformer").appendingPathComponent("model.safetensors")
    let reader = try SafeTensorsReader(fileURL: quantizedTransformer)
    XCTAssertTrue(reader.contains("context_embedder.weight"))
    XCTAssertTrue(reader.contains("context_embedder.scales"))
    XCTAssertTrue(reader.contains("small.weight"))

    let loader = Flux2WeightsLoader(snapshot: output)
    let loaded = try loader.load(component: .transformer, dtype: .bfloat16)
    XCTAssertNotNil(loaded["context_embedder.weight"])
    XCTAssertNotNil(loaded["context_embedder.scales"])
    XCTAssertNotNil(loaded["small.weight"])

    XCTAssertNotEqual(loaded["context_embedder.weight"]?.dtype, .bfloat16, "Quantized weights must not be dtype-cast.")
    XCTAssertEqual(loaded["small.weight"]?.dtype, .bfloat16, "Unquantized float weights should be cast to the target dtype.")

    final class TinyTransformer: Module {
      @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
      @ModuleInfo(key: "small") var small: Linear

      override init() {
        _contextEmbedder.wrappedValue = Linear(64, 8, bias: false)
        _small.wrappedValue = Linear(63, 8, bias: false)
        super.init()
      }
    }

    let model = TinyTransformer()
    let manifest = try XCTUnwrap(Flux2Quantizer.loadManifest(from: output))
    Flux2Quantizer.applyQuantization(to: model, manifest: manifest, weights: loaded)
    XCTAssertTrue(model.contextEmbedder is QuantizedLinear)
    XCTAssertFalse(model.small is QuantizedLinear)

    try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .none)
  }

  private func createMinimalDirectory(at url: URL, files: [String: String]) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    for (name, contents) in files {
      try Data(contents.utf8).write(to: url.appendingPathComponent(name))
    }
  }

  private func writeWeights(url: URL, arrays: [String: MLXArray]) throws {
    try MLX.save(arrays: arrays, metadata: [:], url: url)
  }
}
