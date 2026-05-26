@preconcurrency import ApplicationServices
import AppKit

enum PermissionsManager {
    /// Current AX trust state — does NOT prompt.
    static var isAccessibilityGranted: Bool {
        AXIsProcessTrusted()
    }

    /// Show the system AX prompt (one-shot — only shows once per app install).
    static func requestAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: true]
        _ = AXIsProcessTrustedWithOptions(options as CFDictionary)
    }

    /// Deep-link the user to System Settings → Privacy → Accessibility.
    static func openAccessibilitySettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
        NSWorkspace.shared.open(url)
    }
}
