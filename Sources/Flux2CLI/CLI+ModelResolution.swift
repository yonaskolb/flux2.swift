import Foundation
import Hub

extension CLI {
  static func isHuggingFaceModelId(_ modelSpec: String) -> Bool {
    if modelSpec.hasPrefix("/") || modelSpec.hasPrefix("./") || modelSpec.hasPrefix("../") {
      return false
    }

    let localURL = URL(fileURLWithPath: modelSpec).standardizedFileURL
    if FileManager.default.fileExists(atPath: localURL.path) {
      return false
    }

    let baseSpec = String(modelSpec.split(separator: ":", maxSplits: 1)[0])
    let parts = baseSpec.split(separator: "/")
    guard parts.count == 2 else {
      return false
    }

    let org = String(parts[0])
    let repo = String(parts[1])
    guard !org.isEmpty && !repo.isEmpty else {
      return false
    }

    let pathIndicators = ["models", "model", "weights", "data", "datasets", "checkpoints", "output", "tmp", "temp", "cache"]
    if pathIndicators.contains(org.lowercased()) {
      return false
    }

    if org.filter({ $0 == "." }).count > 1 {
      return false
    }

    return true
  }

  static func resolveSnapshotURL(
    modelSpec: String,
    defaultRevision: String,
    downloadBasePath: String?,
    hfToken: String?,
    globs: [String],
    progressHandler: ((Progress) -> Void)? = nil
  ) async throws -> URL {
    let localURL = URL(fileURLWithPath: modelSpec).standardizedFileURL
    if FileManager.default.fileExists(atPath: localURL.path) {
      return localURL
    }

    guard isHuggingFaceModelId(modelSpec) else {
      throw CLIError.invalidOption("Model not found: \(modelSpec)")
    }

    let (repoId, revisionFromSpec) = parseModelSpec(modelSpec, defaultRevision: defaultRevision)
    let revision = revisionFromSpec ?? defaultRevision
    let resolvedGlobs = globsIncludingQuantization(globs)

    let hubCacheDir = resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBasePath)
    if let cached = findCachedSnapshot(repoId: repoId, revision: revision, hubCacheDir: hubCacheDir) {
      try await ensureQuantizationManifestIfNeeded(
        snapshotURL: cached,
        repoId: repoId,
        revision: revision,
        downloadBasePath: hubCacheDir.path,
        hfToken: hfToken
      )
      return cached
    }

    let hubApi = makeHubApi(downloadBasePath: hubCacheDir.path, hfToken: hfToken)
    return try await snapshotWithRetry(
      hubApi: hubApi,
      repoId: repoId,
      revision: revision,
      globs: resolvedGlobs,
      progressHandler: progressHandler ?? { _ in }
    )
  }

  private static func parseModelSpec(_ modelSpec: String, defaultRevision: String) -> (repoId: String, revision: String?) {
    let parts = modelSpec.split(separator: ":", maxSplits: 1)
    let repoId = String(parts[0])
    if parts.count == 2 {
      let revision = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
      return (repoId, revision.isEmpty ? defaultRevision : revision)
    }
    return (repoId, nil)
  }

  private static func findCachedSnapshot(
    repoId: String,
    revision: String,
    hubCacheDir: URL
  ) -> URL? {
    let fm = FileManager.default

    let swiftPath = hubCacheDir.appendingPathComponent("models").appendingPathComponent(repoId)
    let modelIdKey = repoId.replacingOccurrences(of: "/", with: "--")
    let pythonBase = hubCacheDir.appendingPathComponent("models--\(modelIdKey)")
    let pythonSnapshot = resolvePythonSnapshot(base: pythonBase, revision: revision)
    let swiftIsValid = fm.fileExists(atPath: swiftPath.path) && isValidFlux2SnapshotDirectory(swiftPath)
    let pythonIsValid = pythonSnapshot.map(isValidFlux2SnapshotDirectory) ?? false

    if swiftIsValid { return swiftPath }
    if pythonIsValid, let pythonSnapshot { return pythonSnapshot }
    return nil
  }

  private static func globsIncludingQuantization(_ globs: [String]) -> [String] {
    guard !globs.isEmpty else { return globs }
    guard !globs.contains("quantization.json") else { return globs }
    return globs + ["quantization.json"]
  }

  private static func ensureQuantizationManifestIfNeeded(
    snapshotURL: URL,
    repoId: String,
    revision: String,
    downloadBasePath: String,
    hfToken: String?
  ) async throws {
    let fm = FileManager.default
    let manifestURL = snapshotURL.appendingPathComponent("quantization.json")
    guard !fm.fileExists(atPath: manifestURL.path) else { return }
    guard repoIdLooksQuantized(repoId) else { return }

    let hubApi = makeHubApi(downloadBasePath: downloadBasePath, hfToken: hfToken)
    let endpoint = resolveHuggingFaceEndpoint()
    guard let base = URL(string: endpoint) else {
      throw CLIError.invalidOption("Invalid HF_ENDPOINT: \(endpoint)")
    }

    var url = base
    url = url.appending(path: repoId)
    url = url.appending(path: "resolve")
    url = url.appending(component: revision)
    url = url.appending(path: "quantization.json")

    do {
      let (data, _) = try await hubApi.httpGet(for: url)
      try data.write(to: manifestURL, options: [.atomic])
    } catch let error as Hub.HubClientError {
      if case .fileNotFound(_) = error {
        return
      }
      throw error
    }
  }

  private static func resolveHuggingFaceEndpoint() -> String {
    let raw = ProcessInfo.processInfo.environment["HF_ENDPOINT"] ?? "https://huggingface.co"
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? "https://huggingface.co" : trimmed
  }

  private static func repoIdLooksQuantized(_ repoId: String) -> Bool {
    let lower = repoId.lowercased()
    if lower.contains("quant") { return true }
    if lower.contains("int4") || lower.contains("int8") { return true }
    if lower.contains("q4") || lower.contains("q8") { return true }
    return false
  }

  private static func resolvePythonSnapshot(base: URL, revision: String) -> URL? {
    let fm = FileManager.default

    let snapshotsDir = base.appendingPathComponent("snapshots")
    guard fm.fileExists(atPath: snapshotsDir.path) else {
      return nil
    }

    let refsPath = base.appendingPathComponent("refs").appendingPathComponent(revision)
    if let data = try? Data(contentsOf: refsPath),
       let raw = String(data: data, encoding: .utf8)?
       .trimmingCharacters(in: .whitespacesAndNewlines),
       !raw.isEmpty {
      let candidate = snapshotsDir.appendingPathComponent(raw)
      if isValidFlux2SnapshotDirectory(candidate) {
        return candidate
      }
    }

    guard let snapshots = try? fm.contentsOfDirectory(
      at: snapshotsDir,
      includingPropertiesForKeys: [.contentModificationDateKey],
      options: [.skipsHiddenFiles]
    ) else {
      return nil
    }

    let sorted = snapshots.sorted { lhs, rhs in
      let lhsDate = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
      let rhsDate = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
      return lhsDate > rhsDate
    }

    for snapshot in sorted {
      var isDir: ObjCBool = false
      guard fm.fileExists(atPath: snapshot.path, isDirectory: &isDir), isDir.boolValue else { continue }
      if isValidFlux2SnapshotDirectory(snapshot) {
        return snapshot
      }
    }

    return nil
  }

  private static func isValidFlux2SnapshotDirectory(_ dir: URL) -> Bool {
    let fm = FileManager.default

    let modelIndexURL = dir.appendingPathComponent("model_index.json")
    guard fm.fileExists(atPath: modelIndexURL.path) else {
      return false
    }

    let tokenizerConfig = dir
      .appending(path: "tokenizer", directoryHint: .isDirectory)
      .appendingPathComponent("tokenizer_config.json")
    guard fm.fileExists(atPath: tokenizerConfig.path) else {
      return false
    }

    let schedulerConfig = dir
      .appending(path: "scheduler", directoryHint: .isDirectory)
      .appendingPathComponent("scheduler_config.json")
    guard fm.fileExists(atPath: schedulerConfig.path) else {
      return false
    }

    for component in ["transformer", "text_encoder", "vae"] {
      let config = dir
        .appending(path: component, directoryHint: .isDirectory)
        .appendingPathComponent("config.json")
      guard fm.fileExists(atPath: config.path) else {
        return false
      }

      let weightsDir = dir.appending(path: component, directoryHint: .isDirectory)
      guard directoryContainsSafetensors(weightsDir) else {
        return false
      }
    }

    return true
  }

  private static func directoryContainsSafetensors(_ dir: URL) -> Bool {
    let fm = FileManager.default
    guard let contents = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
      return false
    }
    return contents.contains { $0.pathExtension == "safetensors" }
  }
}
