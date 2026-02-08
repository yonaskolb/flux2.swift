import Foundation
import Flux2
import Hub
import MLX

extension CLI {
  struct GenerateOptions {
    let modelSpec: String?
    let repoId: String?
    let revision: String
    let downloadBasePath: String?
    let hfToken: String?
    let globs: [String]

    let prompt: String?
    let upsamplePrompt: UpsamplePrompt
    let printUpsampledPrompt: Bool

    let imageSpecs: [String]
    let imageIdScale: Int

    let dtypeName: String
    let maxLength: Int
    let seed: Int?
    let guidanceScale: Float
    let height: Int?
    let width: Int?
    let steps: Int?

    let outputPath: String
    let metricsJSONPath: String?
  }

  private struct GenerateRunMetrics: Codable {
    let modelSpec: String
    let snapshotPath: String
    let modelKind: String
    let mode: String
    let steps: Int?
    let dtype: String
    let outputPath: String
    let stages: [String: Double]
  }

  private static func seconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds) + (Double(components.attoseconds) / 1e18)
  }

  private static func writeMetrics(_ metrics: GenerateRunMetrics, to path: String) {
    let url = URL(fileURLWithPath: path).standardizedFileURL
    do {
      try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true,
        attributes: nil
      )
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let data = try encoder.encode(metrics)
      try data.write(to: url, options: [.atomic])
    } catch {
      print("[warn] Failed to write metrics JSON to \(url.path): \(error)")
    }
  }

  private struct ModelIndex: Decodable {
    let className: String

    private enum CodingKeys: String, CodingKey {
      case className = "_class_name"
    }
  }

  private enum ModelKind {
    case klein
    case dev
  }

  static func runDownloadSnapshot(
    repoId: String,
    revision: String,
    downloadBasePath: String?,
    hfToken: String?,
    globs: [String]
  ) async throws {
    var lastPercentage = -1
    let snapshotURL = try await resolveSnapshotURL(
      modelSpec: repoId,
      defaultRevision: revision,
      downloadBasePath: downloadBasePath,
      hfToken: hfToken,
      globs: globs
    ) { progress in
      let percent = Int(progress.fractionCompleted * 100.0)
      if percent != lastPercentage && percent % 5 == 0 {
        print("[download] \(percent)%")
        lastPercentage = percent
      }
    }

    print(snapshotURL.path)
  }

  static func runGenerate(options: GenerateOptions) async throws {
    let clock = ContinuousClock()
    let overallStart = clock.now

    var lastPercentage = -1
    guard let modelSpec = options.repoId ?? options.modelSpec, !modelSpec.isEmpty else {
      throw CLIError.missingArgument("--model/--repo")
    }
    let resolveStart = clock.now
    let snapshotURL = try await resolveSnapshotURL(
      modelSpec: modelSpec,
      defaultRevision: options.revision,
      downloadBasePath: options.downloadBasePath,
      hfToken: options.hfToken,
      globs: options.globs
    ) { progress in
      let percent = Int(progress.fractionCompleted * 100.0)
      if percent != lastPercentage && percent % 5 == 0 {
        print("[download] \(percent)%")
        lastPercentage = percent
      }
    }
    let resolveSeconds = seconds(clock.now - resolveStart)

    let modelKind = try detectModelKind(snapshotURL: snapshotURL)
    var stageTimes: [String: Double] = ["resolve_snapshot_s": resolveSeconds]
    var mode = "t2i"
    let outputURL = resolveOutputURL(options.outputPath)
    if !options.imageSpecs.isEmpty {
      mode = options.upsamplePrompt == .local ? "i2i_upsample" : "i2i"
    } else if options.upsamplePrompt == .local {
      mode = "t2i_upsample"
    }

    switch modelKind {
    case .klein:
      try await generateKleinModel(options: options, snapshotURL: snapshotURL, stageTimes: &stageTimes)
    case .dev:
      try await generateDevModel(options: options, snapshotURL: snapshotURL, stageTimes: &stageTimes)
    }

    stageTimes["total_s"] = seconds(clock.now - overallStart)

    if let metricsPath = options.metricsJSONPath {
      let metrics = GenerateRunMetrics(
        modelSpec: modelSpec,
        snapshotPath: snapshotURL.path,
        modelKind: modelKind == .dev ? "dev" : "klein",
        mode: mode,
        steps: options.steps,
        dtype: options.dtypeName,
        outputPath: outputURL.path,
        stages: stageTimes
      )
      writeMetrics(metrics, to: metricsPath)
    }
  }

  private static func detectModelKind(snapshotURL: URL) throws -> ModelKind {
    let modelIndexURL = snapshotURL.appendingPathComponent("model_index.json")
    guard FileManager.default.fileExists(atPath: modelIndexURL.path) else {
      throw CLIError.invalidOption("model_index.json not found in snapshot: \(snapshotURL.path)")
    }

    let data = try Data(contentsOf: modelIndexURL)
    let index = try JSONDecoder().decode(ModelIndex.self, from: data)
    switch index.className {
    case "Flux2KleinPipeline":
      return .klein
    case "Flux2Pipeline":
      return .dev
    default:
      throw CLIError.invalidOption("Unsupported _class_name '\(index.className)' in \(modelIndexURL.path)")
    }
  }

  private static func generateKleinModel(
    options: GenerateOptions,
    snapshotURL: URL,
    stageTimes: inout [String: Double]
  ) async throws {
    if options.upsamplePrompt != .none {
      throw CLIError.invalidOption("--upsample-prompt is only supported for FLUX.2-dev.")
    }

    guard let resolvedDType = DType.fromCLI(options.dtypeName) else {
      throw CLIError.invalidOption("Unsupported dtype: \(options.dtypeName)")
    }

    if let seed = options.seed {
      MLXRandom.seed(UInt64(seed))
    }

    let dtype = resolvedDType
    let clock = ContinuousClock()

    guard let prompt = options.prompt else {
      throw CLIError.missingArgument("--prompt")
    }

    var height = options.height
    var width = options.width
    let imageSpecs = options.imageSpecs
    let conditioning: [ConditioningImage]
    if imageSpecs.isEmpty {
      conditioning = []
    } else {
      let loadStart = clock.now
      var loaded: [ConditioningImage] = []
      for spec in imageSpecs {
        try await loaded.append(loadConditioningImage(spec: spec))
      }
      conditioning = loaded
      stageTimes["conditioning_load_s"] = seconds(clock.now - loadStart)
      height = height ?? conditioning.first?.height
      width = width ?? conditioning.first?.width
    }
    let conditioningImages = conditioning.isEmpty ? nil : conditioning.map(\.array)

    let steps = options.steps ?? 50
    let guidanceScale = options.guidanceScale
    let imageIdScale = options.imageIdScale

    try Device.withDefaultDevice(.gpu) {
      let initStart = clock.now
      let pipeline = try Flux2KleinPipeline(
        snapshot: snapshotURL,
        dtype: dtype,
        maxLengthOverride: options.maxLength,
        loadTokenizer: true
      )
      stageTimes["pipeline_init_s"] = seconds(clock.now - initStart)

      if pipeline.isDistilled && guidanceScale > 1.0 {
        print("[warn] guidance scale \(guidanceScale) is ignored for step-wise distilled models.")
      }

      let resolvedHeight = height ?? 512
      let resolvedWidth = width ?? 512

      let genStart = clock.now
      let output = try pipeline.generate(
        prompts: [prompt],
        height: resolvedHeight,
        width: resolvedWidth,
        numInferenceSteps: steps,
        guidanceScale: guidanceScale,
        images: conditioningImages,
        imageIdScale: imageIdScale
      )
      stageTimes["pipeline_generate_s"] = seconds(clock.now - genStart)

      let writeStart = clock.now
      let outputURL = resolveOutputURL(options.outputPath)
      try writeImage(image: output.decoded, url: outputURL)
      stageTimes["write_image_s"] = seconds(clock.now - writeStart)
    }
  }

  private static func generateDevModel(
    options: GenerateOptions,
    snapshotURL: URL,
    stageTimes: inout [String: Double]
  ) async throws {
    guard let resolvedDType = DType.fromCLI(options.dtypeName) else {
      throw CLIError.invalidOption("Unsupported dtype: \(options.dtypeName)")
    }

    if let seed = options.seed {
      MLXRandom.seed(UInt64(seed))
    }

    let dtype = resolvedDType
    let clock = ContinuousClock()

    guard let prompt = options.prompt else {
      throw CLIError.missingArgument("--prompt")
    }

    var height = options.height
    var width = options.width
    let imageSpecs = options.imageSpecs
    let conditioning: [ConditioningImage]
    if imageSpecs.isEmpty {
      conditioning = []
    } else {
      let loadStart = clock.now
      var loaded: [ConditioningImage] = []
      for spec in imageSpecs {
        try await loaded.append(loadConditioningImage(spec: spec))
      }
      conditioning = loaded
      stageTimes["conditioning_load_s"] = seconds(clock.now - loadStart)
      height = height ?? conditioning.first?.height
      width = width ?? conditioning.first?.width
    }
    let conditioningImages = conditioning.isEmpty ? nil : conditioning.map(\.array)
    let upsampleImages = conditioning.isEmpty ? nil : conditioning.map(\.original)

    let steps = options.steps ?? 50
    let guidanceScale = options.guidanceScale
    let imageIdScale = options.imageIdScale
    let maxLength = options.maxLength

    try Device.withDefaultDevice(.gpu) {
      let initStart = clock.now
      let pipeline = try Flux2DevPipeline(
        snapshot: snapshotURL,
        dtype: dtype,
        maxLengthOverride: maxLength,
        loadProcessor: true
      )
      stageTimes["pipeline_init_s"] = seconds(clock.now - initStart)

      let resolvedHeight = height ?? 512
      let resolvedWidth = width ?? 512

      let resolvedPrompt: String
      if options.upsamplePrompt == .local {
        var upsampleProcessor = try Flux2PixtralProcessor.load(from: snapshotURL, maxLengthOverride: 2048)
        if let upsampleImages, !upsampleImages.isEmpty, let multimodal = upsampleProcessor.multimodal {
          let baseLongestEdge = multimodal.imageProcessor.configuration.longestEdge
          let cappedLongestEdge = min(baseLongestEdge, 768)
          if cappedLongestEdge < baseLongestEdge {
            upsampleProcessor = try upsampleProcessor.withImageLongestEdge(cappedLongestEdge)
          }
        }
        guard let promptEncoder = pipeline.promptEncoder else {
          throw CLIError.invalidOption("Prompt encoder unavailable (pipeline already used).")
        }
        let upsampler = Flux2DevPromptUpsampler(
          textEncoder: promptEncoder.textEncoder,
          processor: upsampleProcessor
        )

        let upsampleStart = clock.now
        let upsampled = try upsampler.upsample(
          prompts: [prompt],
          images: upsampleImages,
          temperature: 0.15,
          maxNewTokens: 512,
          seed: options.seed.map(UInt64.init),
          prefillChunkSize: 256,
          evaluationPolicy: .deferred,
          onError: { _, error in
            print("[warn] prompt upsampling failed: \(error)")
          }
        )
        stageTimes["prompt_upsample_s"] = seconds(clock.now - upsampleStart)
        resolvedPrompt = upsampled.first ?? prompt
        if options.printUpsampledPrompt {
          print("[upsample] \(resolvedPrompt)")
        }
      } else {
        resolvedPrompt = prompt
      }

      let genStart = clock.now
      let output = try pipeline.generate(
        prompts: [resolvedPrompt],
        height: resolvedHeight,
        width: resolvedWidth,
        numInferenceSteps: steps,
        guidanceScale: guidanceScale,
        images: conditioningImages,
        imageIdScale: imageIdScale,
        maxLength: maxLength
      )
      stageTimes["pipeline_generate_s"] = seconds(clock.now - genStart)

      let writeStart = clock.now
      let outputURL = resolveOutputURL(options.outputPath)
      try writeImage(image: output.decoded, url: outputURL)
      stageTimes["write_image_s"] = seconds(clock.now - writeStart)
    }
  }
}
