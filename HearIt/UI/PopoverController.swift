import AppKit
import SwiftUI

@MainActor
final class PopoverController {
    let popover: NSPopover

    init(settings: Settings, coordinatorState: CoordinatorState, onOpenSettings: @escaping () -> Void) {
        let p = NSPopover()
        p.contentSize = NSSize(width: 320, height: 320)
        p.behavior = .transient
        p.animates = true
        let host = NSHostingController(rootView:
            PopoverView(settings: settings, coordinator: coordinatorState, onOpenSettings: onOpenSettings)
        )
        p.contentViewController = host
        self.popover = p
    }

    func show(relativeTo view: NSView) {
        popover.show(relativeTo: view.bounds, of: view, preferredEdge: .minY)
    }

    func hide() { popover.performClose(nil) }
}

/// NSVisualEffectView wrapped for SwiftUI. Native-stealth foundation.
struct VisualEffectBackground: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .popover
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
}
