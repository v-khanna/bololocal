import XCTest
@testable import Bolo

@MainActor
final class HotkeyManagerTests: XCTestCase {
    func test_register_storesCallback() {
        let manager = HotkeyManager()
        var fired = false
        manager.register { fired = true }
        manager.fire()
        XCTAssertTrue(fired)
    }

    func test_unregister_clearsCallback() {
        let manager = HotkeyManager()
        var fired = false
        manager.register { fired = true }
        manager.unregister()
        manager.fire()
        XCTAssertFalse(fired)
    }
}
