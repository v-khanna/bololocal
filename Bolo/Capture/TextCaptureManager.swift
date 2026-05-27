import ApplicationServices
import AppKit

enum TextCaptureManager {

    /// Best-effort: try AX selected text first, fall back to clipboard.
    /// Returns the captured string or nil.
    static func captureSelectedText() -> String? {
        if let ax = captureFromAccessibility() { return ax }
        return captureFromClipboard()
    }

    /// Read selected text from the frontmost app's focused UI element via AX.
    static func captureFromAccessibility() -> String? {
        guard PermissionsManager.isAccessibilityGranted else { return nil }

        let systemWide = AXUIElementCreateSystemWide()
        var focused: AnyObject?
        let focusErr = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focused
        )
        guard focusErr == .success, let element = focused else { return nil }

        var selected: AnyObject?
        let textErr = AXUIElementCopyAttributeValue(
            element as! AXUIElement,
            kAXSelectedTextAttribute as CFString,
            &selected
        )
        guard textErr == .success, let s = selected as? String, !s.isEmpty else {
            return nil
        }
        return s
    }

    /// Plain clipboard read. Used when AX returns nothing.
    static func captureFromClipboard() -> String? {
        NSPasteboard.general.string(forType: .string)
    }
}
