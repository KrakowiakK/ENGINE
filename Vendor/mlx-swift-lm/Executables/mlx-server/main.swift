// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLLM
import MLXLMServer

// ENGINE: MLX_SERVER_MTP=1 attaches the Qwen3.5/3.6 MTP head at load (the checkpoint must carry mtp.* tensors);
// GenerationBatch then drafts with it whenever a single sequence is running.
if ProcessInfo.processInfo.environment["MLX_SERVER_MTP"] == "1" {
    _qwen35MTPEnabled = true
    FileHandle.standardError.write(Data("mlx-server: Qwen3.5 MTP head enabled\n".utf8))
}

do {
    switch try MLXServerCLI.parse() {
    case .help:
        print(MLXServerCLI.help)
    case .listRoutes:
        let data = try JSONEncoder.openAIServer.encode(MLXServerRoute.manifest)
        print(String(decoding: data, as: UTF8.self))
    case .run(let configuration):
        try await MLXServer.run(configuration: configuration)
    }
} catch {
    FileHandle.standardError.write(Data("mlx-server: \(error.localizedDescription)\n".utf8))
    Foundation.exit(1)
}
