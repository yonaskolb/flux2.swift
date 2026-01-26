import Foundation

extension CLI {
  static func parseCommaSeparatedList(_ raw: String?) -> [String] {
    guard let raw else { return [] }
    return raw
      .split(separator: ",")
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
  }
}
