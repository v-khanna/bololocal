import XCTest
@testable import HearIt

private struct FakeModel: Sendable {}

final class ModelManagerTests: XCTestCase {

    func test_initialState_isUnloaded() async {
        let manager = ModelManager<FakeModel>(idleTimeout: 1.0, loader: { FakeModel() })
        let state = await manager.state
        XCTAssertEqual(state, .unloaded)
    }

    func test_ensureLoaded_transitionsToLoaded() async throws {
        let manager = ModelManager<FakeModel>(idleTimeout: 1.0, loader: { FakeModel() })
        _ = try await manager.ensureLoaded()
        let state = await manager.state
        XCTAssertEqual(state, .loaded)
    }

    func test_idleTimeout_unloadsModel() async throws {
        let manager = ModelManager<FakeModel>(idleTimeout: 0.1, loader: { FakeModel() })
        _ = try await manager.ensureLoaded()
        let stateAfterLoad = await manager.state
        XCTAssertEqual(stateAfterLoad, .loaded)
        try await Task.sleep(nanoseconds: 250_000_000) // 0.25s, well past 0.1s idle
        let stateAfterIdle = await manager.state
        XCTAssertEqual(stateAfterIdle, .unloaded)
    }

    func test_touchResetsIdleTimer() async throws {
        // idleTimeout = 0.5s; touch at ~0.3s resets clock; check at ~0.4s (0.1s since touch,
        // well under 0.5s) — model must still be loaded.
        let manager = ModelManager<FakeModel>(idleTimeout: 0.5, loader: { FakeModel() })
        _ = try await manager.ensureLoaded()
        try await Task.sleep(nanoseconds: 300_000_000) // 0.3s (within initial 0.5s window)
        await manager.touch()                           // reset idle timer to 0.5s from now
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1s after touch (0.4s total)
        let state = await manager.state                 // 0.1s since last touch < 0.5s timeout
        XCTAssertEqual(state, .loaded)
    }
}
