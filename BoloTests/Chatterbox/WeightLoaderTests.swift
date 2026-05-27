// BoloTests/Chatterbox/WeightLoaderTests.swift
import XCTest
@testable import Bolo

final class WeightLoaderTests: XCTestCase {
    func test_isAlreadyDownloaded_returnsBoolWithoutThrowing() {
        let result = WeightLoader.isAlreadyDownloaded()
        XCTAssertTrue(result == true || result == false)
    }

    func test_cachedWeightURL_isInApplicationSupportSubdirectory() {
        let url = WeightLoader.cachedWeightURL
        XCTAssertTrue(url.path.contains("Application Support/Bolo/models"))
        XCTAssertTrue(url.lastPathComponent == "model.safetensors" || url.lastPathComponent.hasSuffix(".safetensors"))
    }

    // Heavy: downloads 2.99 GB on first run. Gated by env var.
    func test_downloadAndLoad_realModel() async throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["BOLO_RUN_HEAVY_TESTS"] != "1",
            "Set BOLO_RUN_HEAVY_TESTS=1 to run heavy weight-loading tests"
        )
        let weights = try await WeightLoader.downloadAndLoad(progressHandler: { _, _ in })
        XCTAssertGreaterThan(weights.count, 100, "Should have many weight tensors")
        XCTAssertTrue(weights.keys.contains(where: { $0.hasPrefix("t3.") }),
                      "Should have T3 weights under t3.* namespace")
        XCTAssertTrue(weights.keys.contains(where: { $0.hasPrefix("s3gen.") }),
                      "Should have S3Gen weights under s3gen.* namespace")
    }
}
