// swift-tools-version: 6.3
import PackageDescription

// ENGINE: the E9 (qwen4_exp) serving engine for M3 Ultra. S1 (2026-09-24) removed the closed arena's targets
// (mlxfast-swift, the runtime worker, the trusted harness and their tests).
let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "engine", targets: ["EngineCLI"]),
    ],
    dependencies: [
        // Exact vendored revisions:
        // mlx-swift df1fdc5f7821a1fabe921fdefbc42ac74dcfb6bc
        // mlx-swift-lm bc1c0ee67d15798343be17c9f8f61f7c0d977149
        .package(path: "Vendor/mlx-swift"),
        .package(path: "Vendor/mlx-swift-lm"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
        // P106 B32 (H42): the segment id cache renders via the vendored Jinja API directly.
        // Same pin as the transitive dependency in Package.resolved.
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.3.6"),
    ],
    targets: [
        // ENGINE: Qwen3.8-Flash-Next (qwen4_exp) text model, our own port.
        .target(
            name: "EngineServeSupport",
            dependencies: [   // P095: the block sampler (BlockSampler.swift) is MLX code
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "Qwen4Exp",
            dependencies: [
                .product(name: "MLXVLM", package: "mlx-swift-lm"),   // P037: Qwen3VLVision.VisionModel
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "EngineCLI",
            dependencies: [
                "EngineServeSupport",
                "Qwen4Exp",
                .product(name: "MLXVLM", package: "mlx-swift-lm"),   // P037: image preprocessing
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
            ]
        ),
        // P086: the prefix store's own suite (T0 swift_tests).
        .testTarget(
            name: "PrefixStoreTests",
            dependencies: ["Qwen4Exp", "EngineServeSupport",
                           .product(name: "Tokenizers", package: "swift-transformers")],   // P093: the detokeniser test runs the checkpoint's own tokenizer
            path: "Tests/PrefixStoreTests"
        ),
    ]
)
