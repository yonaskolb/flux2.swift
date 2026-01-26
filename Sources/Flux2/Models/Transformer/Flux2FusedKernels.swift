import Foundation
import MLX

enum Flux2FusedKernels {
  private static let ropeHeadsPerThread = 4

  private static let ropeKernel = MLXFast.metalKernel(
    name: "flux2_apply_rope",
    inputNames: ["x", "cos", "sin", "seq", "heads", "head_dim", "batch", "rotary_batch_stride"],
    outputNames: ["out"],
    source: """
      uint dp = thread_position_in_grid.x;
      uint seq_idx = thread_position_in_grid.y;
      uint z = thread_position_in_grid.z;

      uint half_dim = uint(head_dim / 2);
      if (dp >= half_dim || seq_idx >= uint(seq)) {
        return;
      }

      int head_groups = (heads + (N - 1)) / N;
      int batch_idx = int(z) / head_groups;
      int head_group = int(z) - batch_idx * head_groups;
      if (batch_idx >= batch) {
        return;
      }

      int head_base = head_group * N;

      ulong row_base = (ulong(seq_idx) + ulong(batch_idx) * ulong(rotary_batch_stride)) * ulong(head_dim);
      device const float* cos_row = cos + row_base;
      device const float* sin_row = sin + row_base;

      uint d0 = dp * 2;
      float c = cos_row[d0];
      float s = sin_row[d0];

      ulong hidden = ulong(heads) * ulong(head_dim);
      ulong token_base = (ulong(batch_idx) * ulong(seq) + ulong(seq_idx)) * hidden;

      for (int i = 0; i < N; ++i) {
        int head_idx = head_base + i;
        if (head_idx >= heads) {
          break;
        }

        ulong base = token_base + ulong(head_idx) * ulong(head_dim) + ulong(d0);
        float x0 = float(x[base]);
        float x1 = float(x[base + 1]);

        out[base] = T(x0 * c - x1 * s);
        out[base + 1] = T(x1 * c + x0 * s);
      }
      """
  )

  static func applyRotaryEmbBf16(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray
  ) -> MLXArray? {
    guard x.dtype == .bfloat16 else {
      return nil
    }
    return applyRotaryEmb(x, cos: cos, sin: sin)
  }

  static func applyRotaryEmb(
    _ x: MLXArray,
    cos: MLXArray,
    sin: MLXArray
  ) -> MLXArray? {
    guard Device.defaultDevice().deviceType == .gpu else {
      return nil
    }
    guard (x.dtype == .bfloat16 || x.dtype == .float16 || x.dtype == .float32),
          cos.dtype == .float32,
          sin.dtype == .float32
    else {
      return nil
    }
    guard x.ndim == 4 else {
      return nil
    }
    let batch = x.dim(0)
    let seq = x.dim(1)
    let heads = x.dim(2)
    let headDim = x.dim(3)

    guard (cos.ndim == 2 || cos.ndim == 3), cos.ndim == sin.ndim else { return nil }
    let batchStride: Int32
    if cos.ndim == 2 {
      guard cos.dim(0) == seq, sin.dim(0) == seq else { return nil }
      guard cos.dim(1) == headDim, sin.dim(1) == headDim else { return nil }
      batchStride = 0
    } else if cos.ndim == 3 {
      guard cos.dim(0) == batch, sin.dim(0) == batch else { return nil }
      guard cos.dim(1) == seq, sin.dim(1) == seq else { return nil }
      guard cos.dim(2) == headDim, sin.dim(2) == headDim else { return nil }
      batchStride = Int32(seq)
    } else {
      return nil
    }
    guard headDim % 2 == 0 else {
      return nil
    }

    let halfDim = headDim / 2
    let headsPerThread = ropeHeadsPerThread
    let headGroups = (heads + headsPerThread - 1) / headsPerThread

    let tgX = max(min(32, halfDim), 1)
    let tgY = max(min(4, seq), 1)

    let out = ropeKernel(
      [
        x,
        cos,
        sin,
        Int32(seq),
        Int32(heads),
        Int32(headDim),
        Int32(batch),
        batchStride,
      ],
      template: [
        ("T", x.dtype),
        ("N", headsPerThread),
      ],
      grid: (halfDim, seq, batch * headGroups),
      threadGroup: (tgX, tgY, 1),
      outputShapes: [x.shape],
      outputDTypes: [x.dtype],
      stream: .gpu
    )
    return out[0]
  }
}
