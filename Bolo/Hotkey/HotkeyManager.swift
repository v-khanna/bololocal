import Foundation
import KeyboardShortcuts
import AppKit

// Central definition of the shortcut name — referenced by both HotkeyManager and SettingsView.
extension KeyboardShortcuts.Name {
    static let readSelection = Self("readSelection", default: .init(.r, modifiers: [.command, .shift]))
}

@MainActor
final class HotkeyManager {
    private var handler: (() -> Void)?

    /// Register the user's chosen hotkey (defaults to ⌘⇧R).
    /// KeyboardShortcuts handles persistence via UserDefaults automatically.
    func register(handler: @escaping () -> Void) {
        self.handler = handler
        KeyboardShortcuts.onKeyDown(for: .readSelection) { [weak self] in
            self?.handler?()
        }
    }

    func unregister() {
        KeyboardShortcuts.removeAllHandlers()
        handler = nil
    }

    /// Manual fire helper for unit tests (does not actually press the hotkey).
    func fire() {
        handler?()
    }
}
