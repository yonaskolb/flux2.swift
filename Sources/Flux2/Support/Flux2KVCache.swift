import MLX

final class Flux2KVCache {
  private var keys: MLXArray?
  private var values: MLXArray?
  private(set) var offset: Int = 0
  private let step: Int

  init(step: Int = 256) {
    self.step = max(step, 1)
  }

  func reset() {
    keys = nil
    values = nil
    offset = 0
  }

  func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
    precondition(newKeys.ndim == 4, "keys must be [B, kvHeads, T, D].")
    precondition(newValues.ndim == 4, "values must be [B, kvHeads, T, D].")
    precondition(
      newKeys.dim(0) == newValues.dim(0) && newKeys.dim(1) == newValues.dim(1) && newKeys.dim(2) == newValues.dim(2),
      "keys and values must match [B, kvHeads, T]."
    )

    let batch = newKeys.dim(0)
    let kvHeads = newKeys.dim(1)
    let newCount = newKeys.dim(2)

    let required = offset + newCount
    let capacity = ((required + step - 1) / step) * step

    if keys == nil || values == nil || required > keys!.dim(2) {
      let keyHeadDim = newKeys.dim(3)
      let valueHeadDim = newValues.dim(3)

      let newKeyStore = MLX.zeros([batch, kvHeads, capacity, keyHeadDim], dtype: newKeys.dtype)
      let newValueStore = MLX.zeros([batch, kvHeads, capacity, valueHeadDim], dtype: newValues.dtype)

      if let existingKeys = keys, let existingValues = values, offset > 0 {
        let copiedKeys = existingKeys[.ellipsis, ..<offset, 0...]
        let copiedValues = existingValues[.ellipsis, ..<offset, 0...]
        newKeyStore[.ellipsis, ..<offset, 0...] = copiedKeys
        newValueStore[.ellipsis, ..<offset, 0...] = copiedValues
      }

      keys = newKeyStore
      values = newValueStore
    }

    let previous = offset
    offset += newCount

    keys![.ellipsis, previous ..< offset, 0...] = newKeys
    values![.ellipsis, previous ..< offset, 0...] = newValues

    let returnedKeys = keys![.ellipsis, ..<offset, 0...]
    let returnedValues = values![.ellipsis, ..<offset, 0...]
    return (returnedKeys, returnedValues)
  }
}

