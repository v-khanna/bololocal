// Bolo/Engine/Chatterbox/WeightLoader.swift
import Foundation
import MLX

/// Downloads and loads Chatterbox-Turbo model.safetensors from HuggingFace.
/// Cache lives at ~/Library/Application Support/Bolo/models/chatterbox-turbo-fp16/model.safetensors
///
/// Pattern mirrors v1's download-to-Application-Support approach:
/// URLSession download with progress callback, move to destination,
/// then MLX.loadArrays to get [String: MLXArray] for injection into T3/S3Gen modules.
enum WeightLoader {

    private static let repoID = "mlx-community/chatterbox-turbo-fp16"
    private static let weightFilename = "model.safetensors"

    /// Where the weight file lives on disk after download.
    static var cachedWeightURL: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return support
            .appendingPathComponent("Bolo", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("chatterbox-turbo-fp16", isDirectory: true)
            .appendingPathComponent(weightFilename)
    }

    /// Returns true if model.safetensors already exists at the cache path.
    static func isAlreadyDownloaded() -> Bool {
        FileManager.default.fileExists(atPath: cachedWeightURL.path)
    }

    /// Download (if needed) and load weights into an MLXArray dictionary.
    /// Reports progress 0.0...1.0 with descriptive labels via progressHandler.
    static func downloadAndLoad(
        progressHandler: @escaping @Sendable (Double, String) -> Void
    ) async throws -> [String: MLXArray] {
        if !isAlreadyDownloaded() {
            try await download(progressHandler: progressHandler)
        }
        progressHandler(1.0, "Loading weights into memory…")
        let arrays = try MLX.loadArrays(url: cachedWeightURL)
        return arrays
    }

    // MARK: - Private

    private static func download(
        progressHandler: @escaping @Sendable (Double, String) -> Void
    ) async throws {
        let url = URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(weightFilename)")!
        let destination = cachedWeightURL

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        progressHandler(0.0, "Downloading Chatterbox-Turbo weights (~2.99 GB)…")

        let delegate = ProgressDelegate(onProgress: { p in
            progressHandler(p, "Downloading… \(Int(p * 100))%")
        })

        let (tempURL, response) = try await URLSession.shared.download(
            for: URLRequest(url: url),
            delegate: delegate
        )

        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw TTSError.synthesisFailed(
                "Chatterbox weight download HTTP error: \((response as? HTTPURLResponse)?.statusCode ?? -1)"
            )
        }

        // Move completed download from temp location to the permanent cache path.
        // Remove any stale partial file first (e.g. from a previous failed download).
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
        let onProgress: @Sendable (Double) -> Void

        init(onProgress: @escaping @Sendable (Double) -> Void) {
            self.onProgress = onProgress
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                        totalBytesExpectedToWrite: Int64) {
            guard totalBytesExpectedToWrite > 0 else { return }
            onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
        }

        func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                        didFinishDownloadingTo location: URL) {
            // The async download(for:delegate:) API delivers the file via its return value,
            // so we don't move it here — the caller handles the move after this delegate
            // method returns.
        }
    }
}
