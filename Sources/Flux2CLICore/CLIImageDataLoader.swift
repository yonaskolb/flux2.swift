import Foundation

public enum CLIImageDataLoader {
  public static func loadDataFromSpec(
    _ spec: String,
    remoteTimeout: TimeInterval = 60,
    remoteMaximumBytes: Int = 50 * 1024 * 1024,
    remoteProtocolClasses: [AnyClass]? = nil
  ) async throws -> Data {
    let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      throw CLIError.invalidOption("Empty image path")
    }

    let lower = trimmed.lowercased()
    if lower.hasPrefix("http://") || lower.hasPrefix("https://") {
      guard let url = URL(string: trimmed) else {
        throw CLIError.invalidOption("Invalid image URL: \(trimmed)")
      }
      return try await fetchRemoteData(
        url: url,
        timeout: remoteTimeout,
        maxBytes: remoteMaximumBytes,
        protocolClasses: remoteProtocolClasses
      )
    }
    if lower.hasPrefix("file://") {
      guard let url = URL(string: trimmed) else {
        throw CLIError.invalidOption("Invalid file URL: \(trimmed)")
      }
      return try Data(contentsOf: url)
    }

    let url = URL(fileURLWithPath: trimmed).standardizedFileURL
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw CLIError.invalidOption("Image not found: \(url.path)")
    }
    return try Data(contentsOf: url)
  }

  private static func fetchRemoteData(
    url: URL,
    timeout: TimeInterval,
    maxBytes: Int,
    protocolClasses: [AnyClass]? = nil
  ) async throws -> Data {
    let config = URLSessionConfiguration.ephemeral
    config.timeoutIntervalForRequest = timeout
    config.timeoutIntervalForResource = timeout
    config.waitsForConnectivity = false
    if let protocolClasses {
      config.protocolClasses = protocolClasses
    }

    let session = URLSession(configuration: config)
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout

    let (data, response): (Data, URLResponse)
    do {
      (data, response) = try await session.data(for: request)
    } catch {
      throw CLIError.invalidOption(
        "Failed to download image: \(url.absoluteString) (\(error.localizedDescription))"
      )
    }

    guard let http = response as? HTTPURLResponse else {
      throw CLIError.invalidOption("Invalid response while downloading image: \(url.absoluteString)")
    }

    guard (200..<300).contains(http.statusCode) else {
      throw CLIError.invalidOption("Failed to download image: \(url.absoluteString) (HTTP \(http.statusCode))")
    }

    if http.expectedContentLength > 0, http.expectedContentLength > Int64(maxBytes) {
      throw CLIError.invalidOption(
        "Remote image too large: \(url.absoluteString) (expected \(http.expectedContentLength) bytes > max \(maxBytes))"
      )
    }

    if data.count > maxBytes {
      throw CLIError.invalidOption(
        "Remote image too large: \(url.absoluteString) (downloaded \(data.count) bytes > max \(maxBytes))"
      )
    }

    return data
  }
}
