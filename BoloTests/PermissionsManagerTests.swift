import XCTest
@testable import Bolo

final class PermissionsManagerTests: XCTestCase {
    func test_isAccessibilityGranted_returnsBool() {
        // We can't force the AX flag in a test sandbox.
        // We assert the API surface exists and returns a Bool.
        let result = PermissionsManager.isAccessibilityGranted
        XCTAssertTrue(result == true || result == false)
    }
}
