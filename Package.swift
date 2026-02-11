// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "Flux2",
  platforms: [
    .macOS(.v14)
  ],
  products: [
    .library(name: "Flux2", targets: ["Flux2"]),
    .library(name: "Flux2CLICore", targets: ["Flux2CLICore"]),
    .executable(name: "flux2-cli", targets: ["Flux2CLI"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.6")),
    .package(url: "https://github.com/apple/swift-argument-parser.git", .upToNextMinor(from: "1.4.0")),
    .package(
      url: "https://github.com/huggingface/swift-transformers",
      .upToNextMinor(from: "1.1.6")
    )
  ],
  targets: [
    .target(
      name: "Flux2",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers")
      ]
    ),
    .target(
      name: "Flux2CLICore",
      dependencies: []
    ),
    .executableTarget(
      name: "Flux2CLI",
      dependencies: [
        "Flux2",
        "Flux2CLICore",
        .product(name: "ArgumentParser", package: "swift-argument-parser")
      ]
    ),
    .testTarget(
      name: "Flux2Tests",
      dependencies: ["Flux2"]
    ),
    .testTarget(
      name: "Flux2CLITests",
      dependencies: ["Flux2CLICore"]
    )
  ]
)
