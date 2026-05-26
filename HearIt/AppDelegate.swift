import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    let popoverController = PopoverController()
    let hotkeyManager = HotkeyManager()
    var coordinator: Coordinator?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Status item
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
        icon?.isTemplate = true
        item.button?.image = icon
        item.button?.action = #selector(togglePopover)
        item.button?.target = self
        self.statusItem = item

        // Pipeline
        let engine: any TTSEngine = MockTTSEngine()
        let playback = PlaybackController(engine: engine)
        let coordinator = Coordinator(hotkey: hotkeyManager, playback: playback)
        coordinator.start()
        self.coordinator = coordinator
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
