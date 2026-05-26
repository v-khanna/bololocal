import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    let popoverController = PopoverController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
        icon?.isTemplate = true
        item.button?.image = icon
        item.button?.action = #selector(togglePopover)
        item.button?.target = self
        self.statusItem = item
    }

    @objc private func togglePopover() {
        guard let button = statusItem?.button else { return }
        if popoverController.popover.isShown {
            popoverController.hide()
        } else {
            popoverController.show(relativeTo: button)
        }
    }
}
