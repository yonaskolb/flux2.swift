import Foundation

public struct FlowMatchEulerDiscreteSchedulerConfiguration: Codable {
  public let numTrainTimesteps: Int
  public let shift: Float
  public let useDynamicShifting: Bool
  public let baseShift: Float
  public let maxShift: Float
  public let baseImageSeqLen: Int
  public let maxImageSeqLen: Int
  public let invertSigmas: Bool
  public let shiftTerminal: Float?
  public let useKarrasSigmas: Bool
  public let useExponentialSigmas: Bool
  public let useBetaSigmas: Bool
  public let timeShiftType: String
  public let stochasticSampling: Bool
  public let useFlowSigmas: Bool?

  enum CodingKeys: String, CodingKey {
    case numTrainTimesteps = "num_train_timesteps"
    case shift
    case useDynamicShifting = "use_dynamic_shifting"
    case baseShift = "base_shift"
    case maxShift = "max_shift"
    case baseImageSeqLen = "base_image_seq_len"
    case maxImageSeqLen = "max_image_seq_len"
    case invertSigmas = "invert_sigmas"
    case shiftTerminal = "shift_terminal"
    case useKarrasSigmas = "use_karras_sigmas"
    case useExponentialSigmas = "use_exponential_sigmas"
    case useBetaSigmas = "use_beta_sigmas"
    case timeShiftType = "time_shift_type"
    case stochasticSampling = "stochastic_sampling"
    case useFlowSigmas = "use_flow_sigmas"
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    numTrainTimesteps = try container.decodeIfPresent(Int.self, forKey: .numTrainTimesteps) ?? 1000
    shift = try container.decodeIfPresent(Float.self, forKey: .shift) ?? 1.0
    useDynamicShifting = try container.decodeIfPresent(Bool.self, forKey: .useDynamicShifting) ?? false
    baseShift = try container.decodeIfPresent(Float.self, forKey: .baseShift) ?? 0.5
    maxShift = try container.decodeIfPresent(Float.self, forKey: .maxShift) ?? 1.15
    baseImageSeqLen = try container.decodeIfPresent(Int.self, forKey: .baseImageSeqLen) ?? 256
    maxImageSeqLen = try container.decodeIfPresent(Int.self, forKey: .maxImageSeqLen) ?? 4096
    invertSigmas = try container.decodeIfPresent(Bool.self, forKey: .invertSigmas) ?? false
    shiftTerminal = try container.decodeIfPresent(Float.self, forKey: .shiftTerminal)
    useKarrasSigmas = try container.decodeIfPresent(Bool.self, forKey: .useKarrasSigmas) ?? false
    useExponentialSigmas = try container.decodeIfPresent(Bool.self, forKey: .useExponentialSigmas) ?? false
    useBetaSigmas = try container.decodeIfPresent(Bool.self, forKey: .useBetaSigmas) ?? false
    timeShiftType = try container.decodeIfPresent(String.self, forKey: .timeShiftType) ?? "exponential"
    stochasticSampling = try container.decodeIfPresent(Bool.self, forKey: .stochasticSampling) ?? false
    useFlowSigmas = try container.decodeIfPresent(Bool.self, forKey: .useFlowSigmas)
  }
}
