import Foundation
import Flux2CLICore
import XCTest

private final class FailingURLProtocol: URLProtocol {
  override class func canInit(with request: URLRequest) -> Bool {
    true
  }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest {
    request
  }

  override func startLoading() {
    client?.urlProtocol(self, didFailWithError: URLError(.timedOut))
  }

  override func stopLoading() {}
}

final class CLIImageTests: XCTestCase {
  func testRemoteTransportErrorPreservesCLIContext() async throws {
    let imageURL = "https://example.com/conditioning.png"

    do {
      _ = try await CLIImageDataLoader.loadDataFromSpec(
        imageURL,
        remoteTimeout: 1,
        remoteMaximumBytes: 1024,
        remoteProtocolClasses: [FailingURLProtocol.self]
      )
      XCTFail("Expected transport failure")
    } catch let error as CLIError {
      guard case .invalidOption(let message) = error else {
        XCTFail("Expected CLIError.invalidOption, got \(error)")
        return
      }
      XCTAssertTrue(message.contains("Failed to download image: \(imageURL)"))
      XCTAssertTrue(message.contains(URLError(.timedOut).localizedDescription))
    } catch {
      XCTFail("Expected CLIError.invalidOption, got \(error)")
    }
  }
}
