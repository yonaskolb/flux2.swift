import Foundation

public struct Flux2TransformerConfig: Sendable {
  public let hiddenSize: Int
  public let numHeads: Int
  public let headDim: Int
  public let mlpHidden: Int
  public let numDoubleLayers: Int
  public let numSingleLayers: Int
  public let textDim: Int
  public let latentChannels: Int
  public let ropeTheta: Float
  public let axesDimsRope: [Int]
}

public struct Flux2TextEncoderConfig: Sendable {
  public let hiddenSize: Int
  public let numHeads: Int
  public let numKeyValueHeads: Int
  public let headDim: Int
  public let intermediateSize: Int
  public let numLayers: Int
  public let vocabSize: Int
  public let rmsNormEps: Float
  public let ropeTheta: Float
}

public struct Flux2VAEConfig: Sendable {
  public let latentChannels: Int
  public let patchSize: (Int, Int)
}

public struct Flux2ModelConfig: Sendable {
  public let transformer: Flux2TransformerConfig
  public let textEncoder: Flux2TextEncoderConfig
  public let vae: Flux2VAEConfig
}

public enum Flux2ModelVariant: String, CaseIterable, Sendable {
  case klein4B = "flux2-klein-4b"
  case klein9B = "flux2-klein-9b"
  case dev = "flux2-dev"
}

public enum Flux2Configs {
  public static let klein4B = Flux2ModelConfig(
    transformer: .init(
      hiddenSize: 3072,
      numHeads: 24,
      headDim: 128,
      mlpHidden: 9216,
      numDoubleLayers: 5,
      numSingleLayers: 20,
      textDim: 7680,
      latentChannels: 128,
      ropeTheta: 2000.0,
      axesDimsRope: [32, 32, 32, 32]
    ),
    textEncoder: .init(
      hiddenSize: 2560,
      numHeads: 32,
      numKeyValueHeads: 8,
      headDim: 128,
      intermediateSize: 9728,
      numLayers: 36,
      vocabSize: 151_936,
      rmsNormEps: 1e-6,
      ropeTheta: 1_000_000.0
    ),
    vae: .init(
      latentChannels: 32,
      patchSize: (2, 2)
    )
  )
}
