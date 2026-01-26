import Foundation
import MLX

enum Flux2AttentionMaskCache {
  private struct Key: Hashable {
    let seqLen: Int
    let dtype: DType
    let deviceType: DeviceType
    let slidingWindow: Int
  }

  private static let lock = NSLock()
  #if swift(>=5.10)
    nonisolated(unsafe) private static var causalMasks: [Key: MLXArray] = [:]
  #else
    private static var causalMasks: [Key: MLXArray] = [:]
  #endif

  static func clear() {
    lock.lock()
    causalMasks.removeAll()
    lock.unlock()
  }

  static func causalAdditive(
    seqLen: Int,
    dtype: DType,
    slidingWindow: Int? = nil
  ) -> MLXArray {
    let normalizedWindow = max(slidingWindow ?? 0, 0)
    let deviceType = Device.defaultDevice().deviceType ?? .gpu
    let key = Key(
      seqLen: seqLen,
      dtype: dtype,
      deviceType: deviceType,
      slidingWindow: normalizedWindow
    )

    lock.lock()
    if let cached = causalMasks[key] {
      lock.unlock()
      return cached
    }
    lock.unlock()

    let built = buildCausalAdditive(seqLen: seqLen, dtype: dtype, slidingWindow: normalizedWindow)

    lock.lock()
    causalMasks[key] = built
    lock.unlock()

    return built
  }

  private static func buildCausalAdditive(
    seqLen: Int,
    dtype: DType,
    slidingWindow: Int
  ) -> MLXArray {
    let negInf = MLXArray(-Float.infinity).asType(dtype)

    let indices = MLXArray(Int32(0)..<Int32(seqLen))
    let row = indices.reshaped(seqLen, 1)
    let col = indices.reshaped(1, seqLen)

    var blocked = col .> row
    if slidingWindow > 0, seqLen > slidingWindow {
      let windowStart = row - MLXArray(Int32(slidingWindow - 1))
      let tooFar = col .< windowStart
      blocked = blocked .|| tooFar
    }

    let zeros = MLX.zeros([seqLen, seqLen], dtype: dtype)
    let additive = MLX.where(blocked, zeros + negInf, zeros)
    return additive.reshaped(1, 1, seqLen, seqLen)
  }
}
