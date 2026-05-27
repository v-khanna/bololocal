import XCTest
import AppKit
@testable import Bolo

@MainActor
final class PopoverControllerTests: XCTestCase {
    /// Minimal helper to construct a PopoverController with dummy dependencies.
    private func makeController() -> PopoverController {
        let settings = Settings(defaults: UserDefaults(suiteName: "PopoverControllerTests-\(UUID())")!)
        let state = CoordinatorState()
        return PopoverController(settings: settings, coordinatorState: state, onOpenSettings: {})
    }

    func test_init_createsPopoverWithCorrectSize() {
        let controller = makeController()
        XCTAssertEqual(controller.popover.contentSize, NSSize(width: 320, height: 320))
        XCTAssertEqual(controller.popover.behavior, .transient)
    }

    func test_show_attachesPopoverToView() {
        let controller = makeController()
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
