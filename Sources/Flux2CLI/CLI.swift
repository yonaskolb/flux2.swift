import ArgumentParser
import Foundation
import Flux2

enum CLIError: Error, LocalizedError {
  case missingArgument(String)
  case invalidOption(String)

  var errorDescription: String? {
    switch self {
    case .missingArgument(let argument):
      return "Missing argument: \(argument)"
    case .invalidOption(let message):
      return message
    }
  }
}

@main
struct CLI: AsyncParsableCommand {
  static let defaultSnapshotGlobs = [
    "model_index.json",
    "quantization.json",
    "scheduler/*.json",
    "tokenizer/*",
    "transformer/*.json",
    "transformer/*.safetensors",
    "vae/*.json",
    "vae/*.safetensors",
    "text_encoder/*.json",
    "text_encoder/*.safetensors"
  ]

  static let configuration = CommandConfiguration(
    commandName: "flux2-cli",
    abstract: "Run FLUX.2 diffusion models with mlx-swift.",
    subcommands: [DownloadSnapshot.self, Quantize.self, Generate.self],
    defaultSubcommand: Generate.self
  )

  enum UpsamplePrompt: String, ExpressibleByArgument {
    case none
    case local
  }

  enum QuantizationModeArgument: String, ExpressibleByArgument {
    case affine
    case mxfp4
  }

  struct DownloadSnapshot: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "download-snapshot",
      abstract: "Download a Hugging Face model snapshot into the local cache."
    )

    @Option(name: .customLong("repo"), help: "Hugging Face model id (org/repo).")
    var repoId: String

    @Option(name: .customLong("revision"), help: "Git revision / branch / tag on the hub.")
    var revision: String = "main"

    @Option(
      name: .customLong("download-base"),
      help: "Override the hub cache directory (default: ~/.cache/huggingface/hub)."
    )
    var downloadBasePath: String?

    @Option(
      name: .customLong("hf-token"),
      help: "Optional Hugging Face token; otherwise uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN or ~/.cache/huggingface/token."
    )
    var hfToken: String?

    @Option(
      name: .customLong("include"),
      help: "Comma-separated list of globs to download. If omitted, downloads Flux2-required files only."
    )
    var include: String?

    @Flag(name: .customLong("download-all"), help: "Download the full repo snapshot (no include filter).")
    var downloadAll: Bool = false

    mutating func run() async throws {
      let includes = CLI.parseCommaSeparatedList(include)
      try await CLI.runDownloadSnapshot(
        repoId: repoId,
        revision: revision,
        downloadBasePath: downloadBasePath,
        hfToken: hfToken,
        globs: downloadAll ? [] : (includes.isEmpty ? CLI.defaultSnapshotGlobs : includes)
      )
    }
  }

  struct Quantize: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "quantize",
      abstract: "Quantize a model snapshot (transformer + text_encoder) and write a new snapshot directory."
    )

    @Option(
      name: [.customLong("model"), .customLong("snapshot")],
      help: "Model path or Hugging Face model id. (--snapshot is a compatible alias.)"
    )
    var modelSpec: String?

    @Option(name: .customLong("repo"), help: "Hugging Face model id (org/repo). Overrides --model/--snapshot if set.")
    var repoId: String?

    @Option(name: .customLong("revision"), help: "Git revision / branch / tag on the hub.")
    var revision: String = "main"

    @Option(
      name: .customLong("download-base"),
      help: "Override the hub cache directory (default: ~/.cache/huggingface/hub)."
    )
    var downloadBasePath: String?

    @Option(
      name: .customLong("hf-token"),
      help: "Optional Hugging Face token; otherwise uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN or ~/.cache/huggingface/token."
    )
    var hfToken: String?

    @Option(
      name: .customLong("include"),
      help: "Comma-separated list of globs to download. If omitted, downloads Flux2-required files only."
    )
    var include: String?

    @Flag(name: .customLong("download-all"), help: "Download the full repo snapshot (no include filter).")
    var downloadAll: Bool = false

    @Option(name: .customLong("bits"), help: "Quantization bit width (default: 8).")
    var bits: Int = 8

    @Option(name: .customLong("group-size"), help: "Quantization group size (default: 64).")
    var groupSize: Int = 64

    @Option(name: .customLong("mode"), help: "Quantization mode: affine (default) or mxfp4.")
    var mode: QuantizationModeArgument = .affine

    @Option(
      name: [.short, .customLong("output")],
      help: "Output snapshot directory (will be created)."
    )
    var outputPath: String

    @Flag(name: .customLong("overwrite"), help: "Overwrite the output directory if it already exists.")
    var overwrite: Bool = false

    @Flag(name: .customLong("verbose"), help: "Print per-shard quantization statistics.")
    var verbose: Bool = false

    mutating func validate() throws {
      if (repoId ?? modelSpec)?.isEmpty ?? true {
        throw ValidationError("Missing model spec. Provide --model (path or repo id) or --repo.")
      }
    }

    mutating func run() async throws {
      let includes = CLI.parseCommaSeparatedList(include)

      let modelSpec = repoId ?? modelSpec!
      let snapshotURL = try await CLI.resolveSnapshotURL(
        modelSpec: modelSpec,
        defaultRevision: revision,
        downloadBasePath: downloadBasePath,
        hfToken: hfToken,
        globs: downloadAll ? [] : (includes.isEmpty ? CLI.defaultSnapshotGlobs : includes)
      ) { progress in
        let percent = Int(progress.fractionCompleted * 100.0)
        if percent % 5 == 0 {
          print("[download] \(percent)%")
        }
      }

      let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL

      let quantMode: Flux2QuantizationMode = mode == .mxfp4 ? .mxfp4 : .affine
      let spec = Flux2QuantizationSpec(groupSize: groupSize, bits: bits, mode: quantMode)
      try Flux2Quantizer.quantizeAndSave(
        from: snapshotURL,
        to: outputURL,
        spec: spec,
        modelId: repoId,
        revision: revision,
        overwrite: overwrite,
        verbose: verbose
      )

      print(outputURL.path)
    }
  }

  struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "generate",
      abstract: "Generate images with a FLUX.2 model snapshot or Hugging Face model id."
    )

    @Option(
      name: [.customLong("model"), .customLong("snapshot")],
      help: "Model path or Hugging Face model id. (--snapshot is a compatible alias.)"
    )
    var modelSpec: String?

    @Option(name: .customLong("repo"), help: "Hugging Face model id (org/repo). Overrides --model/--snapshot if set.")
    var repoId: String?

    @Option(name: .customLong("revision"), help: "Git revision / branch / tag on the hub.")
    var revision: String = "main"

    @Option(
      name: .customLong("download-base"),
      help: "Override the hub cache directory (default: ~/.cache/huggingface/hub)."
    )
    var downloadBasePath: String?

    @Option(
      name: .customLong("hf-token"),
      help: "Optional Hugging Face token; otherwise uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN or ~/.cache/huggingface/token."
    )
    var hfToken: String?

    @Option(
      name: .customLong("include"),
      help: "Comma-separated list of globs to download. If omitted, downloads Flux2-required files only."
    )
    var include: String?

    @Flag(name: .customLong("download-all"), help: "Download the full repo snapshot (no include filter).")
    var downloadAll: Bool = false

    @Option(name: .customLong("prompt"), help: "Text prompt for generation.")
    var prompt: String?

    @Option(name: .customLong("upsample-prompt"), help: "Prompt upsampling (FLUX.2-dev only).")
    var upsamplePrompt: UpsamplePrompt = .none

    @Flag(name: .customLong("print-upsampled-prompt"), help: "Print the upsampled prompt when using --upsample-prompt local.")
    var printUpsampledPrompt: Bool = false

    @Option(name: .customLong("image"), parsing: .unconditionalSingleValue, help: "Conditioning image path or URL. Repeatable.")
    var images: [String] = []

    @Option(name: .customLong("images"), help: "Comma-separated list of conditioning images (path or URL).")
    var imagesCSV: String?

    @Option(name: .customLong("image-id-scale"), help: "Image id scale.")
    var imageIdScale: Int = 10

    @Option(name: .customLong("dtype"), help: "Weight dtype: float16, float32, bfloat16.")
    var dtypeName: String = "bfloat16"

    @Option(name: .customLong("max-length"), help: "Max token length for the main text encoder.")
    var maxLength: Int = 512

    @Option(name: .customLong("seed"), help: "Random seed.")
    var seed: Int?

    @Option(name: .customLong("guidance-scale"), help: "Classifier-free guidance scale.")
    var guidanceScale: Float = 4.0

    @Option(name: .customLong("height"), help: "Output height (defaults to conditioning image height or 512).")
    var height: Int?

    @Option(name: .customLong("width"), help: "Output width (defaults to conditioning image width or 512).")
    var width: Int?

    @Option(name: .customLong("steps"), help: "Number of inference steps.")
    var steps: Int?

    @Option(
      name: [.short, .customLong("output")],
      help: "Output image path. Supported: png, jpg, jpeg. If no extension is provided, .png is appended."
    )
    var outputPath: String = "flux2.png"

    @Option(name: .customLong("metrics-json"), help: "Optional path to write JSON timing metrics for this run.")
    var metricsJSONPath: String?

    mutating func validate() throws {
      if prompt == nil {
        throw ValidationError("Missing required option: --prompt")
      }
      if (repoId ?? modelSpec)?.isEmpty ?? true {
        throw ValidationError("Missing model spec. Provide --model (path or repo id) or --repo.")
      }
    }

    mutating func run() async throws {
      let includes = CLI.parseCommaSeparatedList(include)
      let imageSpecs = images + CLI.parseCommaSeparatedList(imagesCSV)
      try await CLI.runGenerate(
        options: GenerateOptions(
          modelSpec: modelSpec,
          repoId: repoId,
          revision: revision,
          downloadBasePath: downloadBasePath,
          hfToken: hfToken,
          globs: downloadAll ? [] : (includes.isEmpty ? CLI.defaultSnapshotGlobs : includes),
          prompt: prompt,
          upsamplePrompt: upsamplePrompt,
          printUpsampledPrompt: printUpsampledPrompt,
          imageSpecs: imageSpecs,
          imageIdScale: imageIdScale,
          dtypeName: dtypeName,
          maxLength: maxLength,
          seed: seed,
          guidanceScale: guidanceScale,
          height: height,
          width: width,
          steps: steps,
          outputPath: outputPath,
          metricsJSONPath: metricsJSONPath
        )
      )
    }
  }
}
