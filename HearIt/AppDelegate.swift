import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
        icon?.isTemplate = true
        item.button?.image = icon
        item.button?.action = #selector(handleStatusItemClick)
        item.button?.target = self
        self.statusItem = item
    }

    @objc private func handleStatusItemClick() {
        // Popover opens here (Task 3 wires this up).
        NSLog("HearIt status item clicked")
    }
}
