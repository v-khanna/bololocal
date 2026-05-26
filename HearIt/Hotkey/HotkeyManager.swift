import Foundation
import HotKey
import AppKit

@MainActor
final class HotkeyManager {
    private var hotkey: HotKey?
    private var callback: (() -> Void)?

    /// Register the global ⌘⇧R hotkey. Called once at app launch.
    func register(handler: @escaping () -> Void) {
        self.callback = handler
        let hk = HotKey(key: .r, modifiers: [.command, .shift])
        hk.keyDownHandler = { [weak self] in
            self?.callback?()
        }
        self.hotkey = hk
    }

    func unregister() {
        callback = nil
        hotkey = nil
    }

    /// Manual fire helper for unit tests (does not actually press the hotkey).
    func fire() {
        callback?()
    }
}
