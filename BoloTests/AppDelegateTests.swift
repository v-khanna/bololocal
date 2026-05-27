import XCTest
import AppKit
@testable import Bolo

@MainActor
final class AppDelegateTests: XCTestCase {
    func test_applicationDidFinishLaunching_createsStatusItem() {
        let delegate = AppDelegate()
        delegate.applicationDidFinishLaunching(Notification(name: .init("test")))
        XCTAssertNotNil(delegate.statusItem)
        XCTAssertEqual(delegate.statusItem?.button?.image?.accessibilityDescription, "Bolo")
    }
}
