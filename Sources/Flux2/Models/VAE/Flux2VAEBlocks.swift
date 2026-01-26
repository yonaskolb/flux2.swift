import Foundation
import MLX
import MLXFast
import MLXNN

final class Flux2VAESelfAttention: Module {
  @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: [Linear]

  init(channels: Int, normGroups: Int, eps: Float) {
    _groupNorm.wrappedValue = GroupNorm(
      groupCount: normGroups,
      dimensions: channels,
      eps: eps,
      affine: true,
      pytorchCompatible: true
    )
    _toQ.wrappedValue = Linear(channels, channels)
    _toK.wrappedValue = Linear(channels, channels)
    _toV.wrappedValue = Linear(channels, channels)
    _toOut.wrappedValue = [Linear(channels, channels)]
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let b = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)

    var hidden = groupNorm(x)
    let queries = toQ(hidden).reshaped(b, h * w, c).expandedDimensions(axis: 1)
    let keys = toK(hidden).reshaped(b, h * w, c).expandedDimensions(axis: 1)
    let values = toV(hidden).reshaped(b, h * w, c).expandedDimensions(axis: 1)

    let scale = 1 / sqrt(Float(c))
    let attn = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: keys,
      values: values,
      scale: scale,
      mask: nil
    )

    hidden = attn.squeezed(axis: 1).reshaped(b, h, w, c)
    hidden = toOut[0](hidden)
    return x + hidden
  }
}

final class Flux2ResnetBlock2D: Module {
  @ModuleInfo(key: "norm1") var norm1: GroupNorm
  @ModuleInfo(key: "norm2") var norm2: GroupNorm
  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

  private let usesConvShortcut: Bool

  init(inChannels: Int, outChannels: Int, normGroups: Int, eps: Float) {
    _norm1.wrappedValue = GroupNorm(
      groupCount: normGroups,
      dimensions: inChannels,
      eps: eps,
      affine: true,
      pytorchCompatible: true
    )
    _norm2.wrappedValue = GroupNorm(
      groupCount: normGroups,
      dimensions: outChannels,
      eps: eps,
      affine: true,
      pytorchCompatible: true
    )
    _conv1.wrappedValue = Conv2d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    _conv2.wrappedValue = Conv2d(
      inputChannels: outChannels,
      outputChannels: outChannels,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )

    usesConvShortcut = inChannels != outChannels
    if usesConvShortcut {
      _convShortcut.wrappedValue = Conv2d(
        inputChannels: inChannels,
        outputChannels: outChannels,
        kernelSize: 1,
        stride: 1,
        padding: 0
      )
    }

    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = silu(norm1(x))
    hidden = conv1(hidden)
    hidden = silu(norm2(hidden))
    hidden = conv2(hidden)

    let residual = usesConvShortcut ? convShortcut!(x) : x
    return residual + hidden
  }
}

final class Flux2VAEUpSampler: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d

  init(channels: Int) {
    _conv.wrappedValue = Conv2d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 3,
      stride: 1,
      padding: 1
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let upscaled = Flux2VAEUpSampler.upSampleNearest(x)
    return conv(upscaled)
  }

  static func upSampleNearest(_ x: MLXArray, scale: Int = 2) -> MLXArray {
    precondition(x.ndim == 4)
    let b = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)
    var expanded = broadcast(
      x[0..., 0..., .newAxis, 0..., .newAxis, 0...],
      to: [b, h, scale, w, scale, c]
    )
    expanded = expanded.reshaped(b, h * scale, w * scale, c)
    return expanded
  }
}

final class Flux2VAEDownSampler: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d

  init(channels: Int) {
    _conv.wrappedValue = Conv2d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 3,
      stride: 2,
      padding: 0
    )
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = padded(x, widths: [[0, 0], [0, 1], [0, 1], [0, 0]])
    hidden = conv(hidden)
    return hidden
  }
}

final class Flux2VAEUpBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
  @ModuleInfo(key: "upsamplers") var upsamplers: [Flux2VAEUpSampler]

  init(
    inChannels: Int,
    outChannels: Int,
    blockCount: Int,
    hasUpsampler: Bool,
    normGroups: Int,
    eps: Float
  ) {
    _resnets.wrappedValue = (0..<blockCount).map { index in
      let isFirst = index == 0
      let resnetIn = isFirst ? inChannels : outChannels
      return Flux2ResnetBlock2D(
        inChannels: resnetIn,
        outChannels: outChannels,
        normGroups: normGroups,
        eps: eps
      )
    }

    if hasUpsampler {
      _upsamplers.wrappedValue = [Flux2VAEUpSampler(channels: outChannels)]
    } else {
      _upsamplers.wrappedValue = []
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    if !upsamplers.isEmpty {
      hidden = upsamplers[0](hidden)
    }
    return hidden
  }
}

final class Flux2VAEDownBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
  @ModuleInfo(key: "downsamplers") var downsamplers: [Flux2VAEDownSampler]

  init(
    inChannels: Int,
    outChannels: Int,
    blockCount: Int,
    hasDownsampler: Bool,
    normGroups: Int,
    eps: Float
  ) {
    _resnets.wrappedValue = (0..<blockCount).map { index in
      let isFirst = index == 0
      let resnetIn = isFirst ? inChannels : outChannels
      return Flux2ResnetBlock2D(
        inChannels: resnetIn,
        outChannels: outChannels,
        normGroups: normGroups,
        eps: eps
      )
    }

    if hasDownsampler {
      _downsamplers.wrappedValue = [Flux2VAEDownSampler(channels: outChannels)]
    } else {
      _downsamplers.wrappedValue = []
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    if !downsamplers.isEmpty {
      hidden = downsamplers[0](hidden)
    }
    return hidden
  }
}

final class Flux2VAEMidBlock: Module {
  @ModuleInfo(key: "attentions") var attentions: [Flux2VAESelfAttention]
  @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]

  init(channels: Int, normGroups: Int, addAttention: Bool, eps: Float) {
    _attentions.wrappedValue = addAttention
      ? [Flux2VAESelfAttention(channels: channels, normGroups: normGroups, eps: eps)]
      : []
    _resnets.wrappedValue = [
      Flux2ResnetBlock2D(inChannels: channels, outChannels: channels, normGroups: normGroups, eps: eps),
      Flux2ResnetBlock2D(inChannels: channels, outChannels: channels, normGroups: normGroups, eps: eps)
    ]
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = resnets[0](x)
    if !attentions.isEmpty {
      hidden = attentions[0](hidden)
    }
    hidden = resnets[1](hidden)
    return hidden
  }
}
