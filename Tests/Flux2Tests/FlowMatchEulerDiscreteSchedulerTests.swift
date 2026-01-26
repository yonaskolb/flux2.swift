import Foundation
import XCTest

@testable import Flux2

final class FlowMatchEulerDiscreteSchedulerTests: XCTestCase {
  func testLoadSupportsSchedulerConfigFilename() throws {
    let tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("flux2-scheduler-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let schedulerDir = tempDir.appendingPathComponent("scheduler", isDirectory: true)
    try FileManager.default.createDirectory(at: schedulerDir, withIntermediateDirectories: true)

    let configURL = schedulerDir.appendingPathComponent("scheduler_config.json")
    try "{}".write(to: configURL, atomically: true, encoding: .utf8)

    _ = try FlowMatchEulerDiscreteScheduler.load(from: tempDir)
  }
}

