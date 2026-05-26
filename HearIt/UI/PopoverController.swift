import AppKit
import SwiftUI

final class PopoverController {
    let popover: NSPopover

    init() {
        let p = NSPopover()
        p.contentSize = NSSize(width: 320, height: 280)
        p.behavior = .transient
        p.animates = true
        // Placeholder content; replaced by PopoverView in Task 11.
        let host = NSHostingController(rootView: PopoverPlaceholderView())
        p.contentViewController = host
        self.popover = p
    }

    func show(relativeTo view: NSView) {
        popover.show(relativeTo: view.bounds, of: view, preferredEdge: .minY)
    }

    func hide() {
        popover.performClose(nil)
    }
}

private struct PopoverPlaceholderView: View {
    var body: some View {
        ZStack {
            VisualEffectBackground()
            Text("HearIt")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(width: 320, height: 280)
    }
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
