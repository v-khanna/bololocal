import AppKit
@preconcurrency import Qwen3TTS
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    let popoverController = PopoverController()
    let hotkeyManager = HotkeyManager()
    var coordinator: Coordinator?
    var modelManager: ModelManager<Qwen3TTSModel>?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Status item
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
        icon?.isTemplate = true
        item.button?.image = icon
        item.button?.action = #selector(togglePopover)
        item.button?.target = self
        self.statusItem = item

        // Pipeline — model is lazy-loaded on first ⌘⇧R; unloaded after 5 min idle.
        let manager = ModelManager<Qwen3TTSModel>(idleTimeout: 300) {
            try await Qwen3TTSModel.fromPretrained()
        }
        let engine: any TTSEngine = Qwen3TTSEngine(modelProvider: {
            try await manager.ensureLoaded()
        })
        let playback = PlaybackController(engine: engine)
        let coordinator = Coordinator(hotkey: hotkeyManager, playback: playback)
        coordinator.start()
        self.coordinator = coordinator
        self.modelManager = manager
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
