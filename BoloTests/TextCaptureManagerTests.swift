import XCTest
@testable import Bolo

@MainActor
final class TextCaptureManagerTests: XCTestCase {
    func test_captureFromClipboard_returnsCurrentClipboardString() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString("hello world", forType: .string)
        let captured = TextCaptureManager.captureFromClipboard()
        XCTAssertEqual(captured, "hello world")
    }

    func test_captureFromClipboard_returnsNilWhenEmpty() {
        NSPasteboard.general.clearContents()
        let captured = TextCaptureManager.captureFromClipboard()
        XCTAssertNil(captured)
    }
}
