import Darwin
import Foundation
import Hub

extension CLI {
  static func disableSwiftTransformersNetworkMonitorIfNeeded() {
    if ProcessInfo.processInfo.environment["CI_DISABLE_NETWORK_MONITOR"] == nil {
      _ = setenv("CI_DISABLE_NETWORK_MONITOR", "1", 0)
    }
  }

  static func resolveHuggingFaceHubCacheDirectory(downloadBasePath: String?) -> URL {
    func normalize(_ value: String?) -> String? {
      let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
      guard let trimmed, !trimmed.isEmpty else { return nil }
      return trimmed
    }

    if let downloadBasePath = normalize(downloadBasePath) {
      return URL(fileURLWithPath: downloadBasePath).standardizedFileURL
    }

    let env = ProcessInfo.processInfo.environment
    if let hubCache = normalize(env["HF_HUB_CACHE"]) {
      return URL(fileURLWithPath: hubCache).standardizedFileURL
    }
    if let hfHome = normalize(env["HF_HOME"]) {
      return URL(fileURLWithPath: hfHome).appendingPathComponent("hub").standardizedFileURL
    }

    let home = FileManager.default.homeDirectoryForCurrentUser
    return home.appendingPathComponent(".cache/huggingface/hub").standardizedFileURL
  }

  static func snapshotWithRetry(
    hubApi: HubApi,
    repoId: String,
    revision: String,
    globs: [String],
    progressHandler: @escaping (Progress) -> Void
  ) async throws -> URL {
    var attempt = 0
    while true {
      do {
        return try await hubApi.snapshot(
          from: repoId,
          revision: revision,
          matching: globs,
          progressHandler: progressHandler
        )
      } catch let error as HubApi.EnvironmentError {
        if case .offlineModeError = error, attempt < 5 {
          attempt += 1
          try await Task.sleep(nanoseconds: 250_000_000)
          continue
        }
        throw error
      } catch let error as Hub.HubClientError {
        switch error {
        case .authorizationRequired, .httpStatusCode(401):
          throw CLIError.invalidOption(
            "Hugging Face authentication required for \(repoId). Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN), run `hf auth login`, or pass --hf-token."
          )
        default:
          throw error
        }
      }
    }
  }

  static func makeHubApi(downloadBasePath: String?, hfToken: String?) -> HubApi {
    // swift-transformers uses a network monitor to decide "offline mode" which can
    // be overly conservative for short-lived CLI processes (initial `isConnected`
    // defaults to false, and "expensive"/"constrained" networks are treated as offline).
    disableSwiftTransformersNetworkMonitorIfNeeded()

    let token = resolveHuggingFaceToken(cliToken: hfToken)
    let base = resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBasePath)
    try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    return HubApi(downloadBase: base, hfToken: token)
  }

  static func resolveHuggingFaceToken(cliToken: String?) -> String? {
    func normalize(_ token: String?) -> String? {
      let trimmed = token?.trimmingCharacters(in: .whitespacesAndNewlines)
      guard let trimmed, !trimmed.isEmpty else { return nil }
      return trimmed
    }

    if let token = normalize(cliToken) {
      return token
    }

    let env = ProcessInfo.processInfo.environment
    if let token = normalize(env["HF_TOKEN"])
      ?? normalize(env["HUGGINGFACE_HUB_TOKEN"])
      ?? normalize(env["HUGGING_FACE_HUB_TOKEN"])
    {
      return token
    }

    var candidates: [URL] = []
    if let hfHome = normalize(env["HF_HOME"]) {
      candidates.append(URL(fileURLWithPath: hfHome).appendingPathComponent("token"))
    }

    let home = FileManager.default.homeDirectoryForCurrentUser
    candidates.append(home.appendingPathComponent(".cache/huggingface/token"))
    candidates.append(home.appendingPathComponent(".huggingface/token"))

    for url in candidates {
      guard let data = try? Data(contentsOf: url),
            let raw = String(data: data, encoding: .utf8),
            let token = normalize(raw)
      else { continue }
      return token
    }

    return nil
  }
}
