import Foundation
import MLX
import MLXRandom

public struct FlowMatchEulerDiscreteSchedulerOutput {
  public let prevSample: MLXArray
}

public enum FlowMatchEulerDiscreteSchedulerError: Error {
  case configNotFound(URL)
  case invalidTimeShiftType(String)
  case betaSigmasUnsupported
  case missingDynamicShiftMu
  case invalidSchedule(String)
}

public final class FlowMatchEulerDiscreteScheduler {
  public let config: FlowMatchEulerDiscreteSchedulerConfiguration
  public let order = 1

  public private(set) var timesteps: MLXArray
  public private(set) var sigmas: MLXArray
  public private(set) var numInferenceSteps: Int

  private var timestepsList: [Float]
  private var sigmasList: [Float]
  private var stepIndex: Int?
  private var beginIndex: Int?
  private var shift: Float
  private var sigmaMin: Float
  private var sigmaMax: Float

  public var timestepsValues: [Float] {
    timestepsList
  }

  public var sigmasValues: [Float] {
    sigmasList
  }

  public init(configuration: FlowMatchEulerDiscreteSchedulerConfiguration) {
    self.config = configuration
    self.shift = configuration.shift

    let trainSteps = max(configuration.numTrainTimesteps, 1)
    let descending = Self.linspace(start: 1.0, end: Float(trainSteps), count: trainSteps).reversed()
    var sigmas = descending.map { $0 / Float(trainSteps) }

    if !configuration.useDynamicShifting {
      sigmas = Self.applyShift(sigmas, shift: configuration.shift)
    }

    let timesteps = sigmas.map { $0 * Float(trainSteps) }
    timestepsList = timesteps
    sigmasList = sigmas
    self.timesteps = MLXArray(timesteps)
    self.sigmas = MLXArray(sigmas)
    numInferenceSteps = timesteps.count
    sigmaMin = sigmas.last ?? 0.0
    sigmaMax = sigmas.first ?? 1.0
  }

  public static func load(from snapshot: URL) throws -> FlowMatchEulerDiscreteScheduler {
    let schedulerDir = snapshot.appendingPathComponent("scheduler")
    let configURL = schedulerDir.appendingPathComponent("config.json")
    let schedulerConfigURL = schedulerDir.appendingPathComponent("scheduler_config.json")

    let resolvedURL: URL
    if FileManager.default.fileExists(atPath: configURL.path) {
      resolvedURL = configURL
    } else if FileManager.default.fileExists(atPath: schedulerConfigURL.path) {
      resolvedURL = schedulerConfigURL
    } else {
      throw FlowMatchEulerDiscreteSchedulerError.configNotFound(schedulerConfigURL)
    }

    let data = try Data(contentsOf: resolvedURL)
    let configuration = try JSONDecoder().decode(FlowMatchEulerDiscreteSchedulerConfiguration.self, from: data)
    return FlowMatchEulerDiscreteScheduler(configuration: configuration)
  }

  public var currentStepIndex: Int? {
    stepIndex
  }

  public var currentBeginIndex: Int? {
    beginIndex
  }

  public func setBeginIndex(_ beginIndex: Int = 0) {
    self.beginIndex = beginIndex
  }

  public func setShift(_ shift: Float) {
    self.shift = shift
  }

  public func setTimesteps(
    numInferenceSteps: Int? = nil,
    sigmas: [Float]? = nil,
    mu: Float? = nil,
    timesteps: [Float]? = nil
  ) throws {
    if config.useDynamicShifting && mu == nil {
      throw FlowMatchEulerDiscreteSchedulerError.missingDynamicShiftMu
    }
    if config.useDynamicShifting && config.timeShiftType != "exponential" && config.timeShiftType != "linear" {
      throw FlowMatchEulerDiscreteSchedulerError.invalidTimeShiftType(config.timeShiftType)
    }

    if let sigmas, let timesteps, sigmas.count != timesteps.count {
      throw FlowMatchEulerDiscreteSchedulerError.invalidSchedule("sigmas and timesteps must match length")
    }

    let resolvedSteps: Int
    if let numInferenceSteps {
      if let sigmas, sigmas.count != numInferenceSteps {
        throw FlowMatchEulerDiscreteSchedulerError.invalidSchedule("sigmas count must match numInferenceSteps")
      }
      if let timesteps, timesteps.count != numInferenceSteps {
        throw FlowMatchEulerDiscreteSchedulerError.invalidSchedule("timesteps count must match numInferenceSteps")
      }
      resolvedSteps = numInferenceSteps
    } else if let sigmas {
      resolvedSteps = sigmas.count
    } else if let timesteps {
      resolvedSteps = timesteps.count
    } else {
      throw FlowMatchEulerDiscreteSchedulerError.invalidSchedule("numInferenceSteps, sigmas, or timesteps required")
    }

    let isTimestepsProvided = timesteps != nil
    let trainSteps = Float(config.numTrainTimesteps)

    var resolvedTimesteps = timesteps ?? []
    var resolvedSigmas: [Float]

    if sigmas == nil {
      if timesteps == nil {
        let start = sigmaMax * trainSteps
        let end = sigmaMin * trainSteps
        resolvedTimesteps = Self.linspace(start: start, end: end, count: resolvedSteps)
      }
      resolvedSigmas = resolvedTimesteps.map { $0 / trainSteps }
    } else {
      resolvedSigmas = sigmas ?? []
    }

    if config.useDynamicShifting {
      resolvedSigmas = Self.timeShift(
        sigmas: resolvedSigmas,
        mu: mu ?? 0.0,
        sigma: 1.0,
        type: config.timeShiftType
      )
    } else {
      resolvedSigmas = Self.applyShift(resolvedSigmas, shift: shift)
    }

    if let shiftTerminal = config.shiftTerminal {
      resolvedSigmas = Self.stretchShiftToTerminal(resolvedSigmas, shiftTerminal: shiftTerminal)
    }

    if config.useKarrasSigmas {
      resolvedSigmas = Self.convertToKarras(resolvedSigmas, numInferenceSteps: resolvedSteps)
    } else if config.useExponentialSigmas {
      resolvedSigmas = Self.convertToExponential(resolvedSigmas, numInferenceSteps: resolvedSteps)
    } else if config.useBetaSigmas {
      throw FlowMatchEulerDiscreteSchedulerError.betaSigmasUnsupported
    }

    if !isTimestepsProvided {
      resolvedTimesteps = resolvedSigmas.map { $0 * trainSteps }
    }

    if config.invertSigmas {
      resolvedSigmas = resolvedSigmas.map { 1.0 - $0 }
      resolvedTimesteps = resolvedSigmas.map { $0 * trainSteps }
      resolvedSigmas.append(1.0)
    } else {
      resolvedSigmas.append(0.0)
    }

    updateSchedule(sigmas: resolvedSigmas, timesteps: resolvedTimesteps)
    self.numInferenceSteps = resolvedSteps
    self.stepIndex = nil
    self.beginIndex = nil
  }

  public func scaleNoise(
    sample: MLXArray,
    timestep: MLXArray,
    noise: MLXArray
  ) throws -> MLXArray {
    let timestepValues = timestep.asType(.float32).reshaped(-1)
    let count = timestepValues.size
    let indices: MLXArray

    if beginIndex == nil {
      let schedule = timesteps.asType(.float32)
      let diffs = MLX.abs(schedule.expandedDimensions(axis: 0) - timestepValues.expandedDimensions(axis: 1))
      let argMin = diffs.argMin(axis: 1)

      let selected = schedule[argMin]
      let delta = MLX.abs(selected - timestepValues)
      let tol = MLX.maximum(MLX.abs(timestepValues) * MLXArray(1e-6), MLXArray(1e-5))
      let ok = (delta .<= tol).all().item(Bool.self)
      if !ok {
        throw FlowMatchEulerDiscreteSchedulerError.invalidSchedule("timestep \(timestepValues) not found in schedule")
      }

      indices = argMin
    } else if let stepIndex {
      indices = MLXArray(Array(repeating: Int32(stepIndex), count: count))
    } else if let beginIndex {
      indices = MLXArray(Array(repeating: Int32(beginIndex), count: count))
    } else {
      indices = MLXArray([])
    }

    var sigma = sigmas[indices].asType(sample.dtype)
    while sigma.ndim < sample.ndim {
      sigma = sigma.expandedDimensions(axis: -1)
    }

    let one = MLXArray(1.0).asType(sample.dtype)
    return sigma * noise + (one - sigma) * sample
  }

  public func step(
    modelOutput: MLXArray,
    timestep: MLXArray,
    sample: MLXArray,
    perTokenTimesteps: MLXArray? = nil
  ) throws -> FlowMatchEulerDiscreteSchedulerOutput {
    if stepIndex == nil {
      try initStepIndex(timestep)
    }

    let sampleFloat = sample.asType(.float32)
    let modelFloat = modelOutput.asType(.float32)
    let prevSample: MLXArray

    if let perTokenTimesteps {
      let trainSteps = MLXArray(Float(config.numTrainTimesteps))
      let perTokenSigmas = perTokenTimesteps.asType(.float32) / trainSteps
      let sigmasExpanded = sigmas.reshaped(sigmas.dim(0), 1, 1)
      let threshold = perTokenSigmas.expandedDimensions(axis: 0) - MLXArray(1e-6)
      let lowerMask = sigmasExpanded .< threshold
      var lowerSigmas = lowerMask.asType(sigmasExpanded.dtype) * sigmasExpanded
      lowerSigmas = lowerSigmas.max(axis: 0)

      let currentSigma = perTokenSigmas.expandedDimensions(axis: -1)
      let nextSigma = lowerSigmas.expandedDimensions(axis: -1)
      let dt = currentSigma - nextSigma

      if config.stochasticSampling {
        let x0 = sampleFloat - currentSigma * modelFloat
        let noise = MLXRandom.normal(sample.shape, dtype: sampleFloat.dtype)
        prevSample = (MLXArray(1.0) - nextSigma) * x0 + nextSigma * noise
      } else {
        prevSample = sampleFloat + dt * modelFloat
      }
    } else {
      let sigmaIdx = stepIndex ?? 0
      let currentSigma = sigmasList[sigmaIdx]
      let nextSigma = sigmasList[sigmaIdx + 1]
      let dt = nextSigma - currentSigma

      if config.stochasticSampling {
        let x0 = sampleFloat - MLXArray(currentSigma) * modelFloat
        let noise = MLXRandom.normal(sample.shape, dtype: sampleFloat.dtype)
        prevSample = (MLXArray(1.0 - nextSigma) * x0) + MLXArray(nextSigma) * noise
      } else {
        prevSample = sampleFloat + MLXArray(dt) * modelFloat
      }
    }

    stepIndex = (stepIndex ?? 0) + 1
    let output = perTokenTimesteps == nil ? prevSample.asType(modelOutput.dtype) : prevSample
    return FlowMatchEulerDiscreteSchedulerOutput(prevSample: output)
  }

  private func updateSchedule(sigmas: [Float], timesteps: [Float]) {
    sigmasList = sigmas
    timestepsList = timesteps
    self.sigmas = MLXArray(sigmas)
    self.timesteps = MLXArray(timesteps)
  }

  private func initStepIndex(_ timestep: MLXArray) throws {
    if beginIndex == nil {
      let value = scalarValue(timestep)
      stepIndex = try indexForTimestep(value, scheduleTimesteps: timestepsList)
    } else {
      stepIndex = beginIndex
    }
  }

  private func scalarValue(_ array: MLXArray) -> Float {
    if array.size == 0 {
      return 0.0
    }
    let flattened = array.asType(.float32).reshaped(-1)
    return Float(flattened[0].item(Float32.self))
  }

  private func indexForTimestep(_ timestep: Float, scheduleTimesteps: [Float]) throws -> Int {
    let exactMatches = scheduleTimesteps.enumerated().filter { $0.element == timestep }
    if !exactMatches.isEmpty {
      let pos = exactMatches.count > 1 ? 1 : 0
      return exactMatches[pos].offset
    }

    let tolerance = max(abs(timestep) * 1e-6, 1e-5)
    var bestIndex: Int? = nil
    var bestDelta = Float.greatestFiniteMagnitude
    for (index, value) in scheduleTimesteps.enumerated() {
      let delta = abs(value - timestep)
      if delta <= tolerance && delta < bestDelta {
        bestDelta = delta
        bestIndex = index
      }
    }
    if let bestIndex {
      return bestIndex
    }

    throw FlowMatchEulerDiscreteSchedulerError.invalidSchedule("timestep \(timestep) not found in schedule")
  }

  private static func applyShift(_ sigmas: [Float], shift: Float) -> [Float] {
    guard shift != 1.0 else { return sigmas }
    return sigmas.map { sigma in
      shift * sigma / (1 + (shift - 1) * sigma)
    }
  }

  private static func timeShift(sigmas: [Float], mu: Float, sigma: Float, type: String) -> [Float] {
    return sigmas.map { t in
      let value = max(Double(t), 1e-6)
      let power = pow((1 / value - 1), Double(sigma))
      if type == "linear" {
        return Float(Double(mu) / (Double(mu) + power))
      }
      let numerator = exp(Double(mu))
      return Float(numerator / (numerator + power))
    }
  }

  private static func stretchShiftToTerminal(_ sigmas: [Float], shiftTerminal: Float) -> [Float] {
    guard let last = sigmas.last else { return sigmas }
    let oneMinusZ = sigmas.map { 1 - $0 }
    let scaleFactor = oneMinusZ.last ?? (1 - last)
    let denom = max(1e-6, 1 - shiftTerminal)
    let factor = scaleFactor / denom
    return oneMinusZ.map { 1 - ($0 / factor) }
  }

  private static func convertToKarras(_ sigmas: [Float], numInferenceSteps: Int) -> [Float] {
    guard let sigmaMin = sigmas.last, let sigmaMax = sigmas.first else { return sigmas }
    let rho = 7.0
    let ramp = linspace(start: 0.0, end: 1.0, count: numInferenceSteps)
    let minInvRho = pow(Double(sigmaMin), 1.0 / rho)
    let maxInvRho = pow(Double(sigmaMax), 1.0 / rho)

    return ramp.map { step in
      let value = maxInvRho + Double(step) * (minInvRho - maxInvRho)
      return Float(pow(value, rho))
    }
  }

  private static func convertToExponential(_ sigmas: [Float], numInferenceSteps: Int) -> [Float] {
    guard let sigmaMin = sigmas.last, let sigmaMax = sigmas.first else { return sigmas }
    let start = log(Double(sigmaMax))
    let end = log(Double(sigmaMin))
    let ramp = linspace(start: Float(start), end: Float(end), count: numInferenceSteps)
    return ramp.map { Float(exp(Double($0))) }
  }

  private static func linspace(start: Float, end: Float, count: Int) -> [Float] {
    guard count > 0 else { return [] }
    if count == 1 { return [start] }
    let step = (end - start) / Float(count - 1)
    return (0..<count).map { start + Float($0) * step }
  }
}

extension FlowMatchEulerDiscreteScheduler {
  func flux2DefaultSigmas(numInferenceSteps: Int) -> [Float]? {
    if config.useFlowSigmas == true {
      return nil
    }

    let start: Float = 1.0
    let end: Float = 1.0 / Float(numInferenceSteps)
    if numInferenceSteps <= 1 {
      return [start]
    }
    let step = (end - start) / Float(numInferenceSteps - 1)
    return (0..<numInferenceSteps).map { start + Float($0) * step }
  }

  static func flux2EmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
    let a1: Float = 8.73809524e-05
    let b1: Float = 1.89833333
    let a2: Float = 0.00016927
    let b2: Float = 0.45666666

    if imageSeqLen > 4300 {
      return a2 * Float(imageSeqLen) + b2
    }

    let m200 = a2 * Float(imageSeqLen) + b2
    let m10 = a1 * Float(imageSeqLen) + b1
    let a = (m200 - m10) / 190.0
    let b = m200 - 200.0 * a
    return a * Float(numSteps) + b
  }
}
