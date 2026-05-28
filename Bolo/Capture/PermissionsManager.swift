@preconcurrency import ApplicationServices
import AppKit
import os.log

private let permLogger = Logger(subsystem: "com.virkhanna.bolo", category: "permissions")

enum PermissionsManager {
    /// Current AX trust state — does NOT prompt.
    /// TEMPORARY DEV OVERRIDE: returns true unconditionally to bypass TCC.
    /// During ad-hoc signed dev rebuilds the cdhash changes every build, so
    /// granting permission in System Settings doesn't stick. With this bypass:
    /// - AX text capture in Coordinator will be attempted; if it fails (no
    ///   permission), TextCaptureManager falls back to clipboard.
    /// - User must ⌘C to copy text first, then press ⌘⇧R.
    /// Remove this override once we have stable signing (Apple ID in Xcode or
    /// a paid Developer ID).
    static var isAccessibilityGranted: Bool {
        let real = AXIsProcessTrusted()
        permLogger.notice("AX check (BYPASSED): real=\(real, privacy: .public)")
        return true
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
