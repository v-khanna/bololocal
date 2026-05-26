import XCTest
import AppKit
@testable import HearIt

@MainActor
final class PopoverControllerTests: XCTestCase {
    func test_init_createsPopoverWithCorrectSize() {
        let controller = PopoverController()
        XCTAssertEqual(controller.popover.contentSize, NSSize(width: 320, height: 280))
        XCTAssertEqual(controller.popover.behavior, .transient)
    }

    func test_show_attachesPopoverToView() {
        let controller = PopoverController()
        // NSPopover requires a view with a window; use an offscreen panel.
        let window = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 100, height: 50),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        let view = NSView(frame: NSRect(x: 0, y: 0, width: 100, height: 50))
        window.contentView = view
        window.orderBack(nil)           // give the view a window without making it key
        controller.show(relativeTo: view)
        XCTAssertTrue(controller.popover.isShown)
        controller.hide()
        window.close()
    }
}
