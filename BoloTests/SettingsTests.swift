import XCTest
@testable import Bolo

@MainActor
final class SettingsTests: XCTestCase {
    override func setUp() async throws {
        try await super.setUp()
        Settings.shared.reset()
    }

    func test_defaultLanguage_isEnglish() {
        XCTAssertEqual(Settings.shared.selectedLanguage, "english")
    }

    func test_defaultSpeed_isOne() {
        XCTAssertEqual(Settings.shared.speed.value, 1.0)
    }

    func test_speedSetterPersists() {
        Settings.shared.speed = Speed(1.5)
        XCTAssertEqual(Settings.shared.speed.value, 1.5)
    }

    func test_hasCompletedOnboarding_defaultsFalse() {
        XCTAssertFalse(Settings.shared.hasCompletedOnboarding)
    }
}
