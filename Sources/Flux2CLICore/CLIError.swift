import Foundation

public enum CLIError: Error, LocalizedError {
  case missingArgument(String)
  case invalidOption(String)

  public var errorDescription: String? {
    switch self {
    case .missingArgument(let argument):
      return "Missing argument: \(argument)"
    case .invalidOption(let message):
      return message
    }
  }
}
