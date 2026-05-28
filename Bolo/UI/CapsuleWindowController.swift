// Bolo/UI/CapsuleWindowController.swift
// Faithful port of FreeFlow's RecordingOverlayManager (zachlatta/freeflow,
// Sources/RecordingOverlay.swift) — panel config, notch-aware winged frame
// math, and the signature drop-down spring animation. Bolo shows this whenever
// `coordinator.state.isPlaying` is true and hides it when playback ends.

import AppKit
import SwiftUI
import Combine

@MainActor
final class CapsuleWindowController {
    private let coordinatorState: CoordinatorState
    private let onStop: () -> Void

    private var panel: NSPanel?
    private var isPlayingCancellable: AnyCancellable?

    /// Wing width — tight to the waveform / stop button so the panel stays
    /// clear of right-side menu-bar items. (FreeFlow uses 36.)
    private static let wingWidth: CGFloat = 36
    /// Width of the centered pill on non-notched displays.
    private static let pillWidth: CGFloat = 150

    init(coordinatorState: CoordinatorState, onStop: @escaping () -> Void) {
        self.coordinatorState = coordinatorState
        self.onStop = onStop
        isPlayingCancellable = coordinatorState.$isPlaying
            .receive(on: RunLoop.main)
            .sink { [weak self] playing in
                if playing { self?.show() } else { self?.hide() }
            }
    }

    // MARK: - Screen / notch geometry (verbatim from FreeFlow)

    private var targetScreen: NSScreen? {
        NSScreen.main ?? NSScreen.screens.first
    }

    private var screenHasNotch: Bool {
        guard let screen = targetScreen else { return false }
        return screen.safeAreaInsets.top > 0
    }

    private var notchWidth: CGFloat {
        guard let screen = targetScreen, screenHasNotch else { return 0 }
        guard let leftArea = screen.auxiliaryTopLeftArea,
              let rightArea = screen.auxiliaryTopRightArea else { return 0 }
        return screen.frame.width - leftArea.width - rightArea.width
    }

    /// Height of the menu-bar strip (== notch height on notched displays).
    private var notchOverlap: CGFloat {
        guard let screen = targetScreen else { return 0 }
        return screen.frame.maxY - screen.visibleFrame.maxY
    }

    private var useWingedLayout: Bool { screenHasNotch && notchWidth > 0 }

    private var overlayFrame: NSRect {
        guard let screen = targetScreen else { return .zero }
        if useWingedLayout {
            let nWidth = notchWidth
            let nLeftX = screen.auxiliaryTopLeftArea?.maxX ?? (screen.frame.midX - nWidth / 2)
            let panelHeight = notchOverlap
            let panelWidth = Self.wingWidth + nWidth + Self.wingWidth
            let panelX = nLeftX - Self.wingWidth
            let panelY = screen.frame.maxY - panelHeight
            return NSRect(x: panelX, y: panelY, width: panelWidth, height: panelHeight)
        }
        // Flat display: centered pill flush under the menu bar.
        let width = Self.pillWidth
        let height: CGFloat = max(notchOverlap, 28)
        let x = screen.frame.midX - width / 2
        let y = screen.frame.maxY - height
        return NSRect(x: x, y: y, width: width, height: height)
    }

    // MARK: - Show / hide (drop-down spring, verbatim timing from FreeFlow)

    private func show() {
        let frame = overlayFrame
        guard let screen = targetScreen else { return }

        if let panel {
            panel.contentView = makeContent(frame: frame)
            panel.setFrame(frame, display: true)
            panel.alphaValue = 1
            panel.orderFrontRegardless()
            return
        }

        let panel = makePanel(frame: frame)
        panel.contentView = makeContent(frame: frame)

        // Start hidden above the screen, then spring down into place.
        let hiddenFrame = NSRect(x: frame.origin.x, y: screen.frame.maxY,
                                 width: frame.width, height: frame.height)
        panel.setFrame(hiddenFrame, display: true)
        panel.alphaValue = 1
        panel.orderFrontRegardless()

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.18
            context.timingFunction = CAMediaTimingFunction(controlPoints: 0.34, 1.56, 0.64, 1.0)
            panel.animator().setFrame(frame, display: true)
        }
        self.panel = panel
    }

    private func hide() {
        guard let panel else { return }
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.12
            context.timingFunction = CAMediaTimingFunction(name: .easeIn)
            panel.animator().alphaValue = 0
        }, completionHandler: { [weak self] in
            self?.panel?.orderOut(nil)
            self?.panel = nil
        })
    }

    // MARK: - Panel + content

    private func makePanel(frame: NSRect) -> NSPanel {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: frame.width, height: frame.height),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false                 // FreeFlow disables shadow for the winged overlay
        panel.level = .screenSaver               // higher than .statusBar — floats above more UI
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        panel.ignoresMouseEvents = false         // Bolo needs the stop button clickable
        return panel
    }

    /// Builds the SwiftUI content, clipped with rounded BOTTOM corners only so
    /// the overlay tucks flush against the top edge / notch.
    private func makeContent(frame: NSRect) -> NSView {
        let cornerRadius: CGFloat = useWingedLayout ? 14 : (screenHasNotch ? 18 : 12)
        let inner: AnyView
        if useWingedLayout {
            inner = AnyView(
                WingedReadingView(
                    coordinator: coordinatorState,
                    leftWingWidth: Self.wingWidth,
                    notchWidth: notchWidth,
                    rightWingWidth: Self.wingWidth,
                    height: frame.height,
                    onStop: { [weak self] in self?.onStop() }
                )
            )
        } else {
            inner = AnyView(
                PillReadingView(
                    coordinator: coordinatorState,
                    onStop: { [weak self] in self?.onStop() }
                )
            )
        }

        let shaped = inner
            .frame(width: frame.width, height: frame.height)
            .background(Color.black)
            .clipShape(UnevenRoundedRectangle(bottomLeadingRadius: cornerRadius,
                                              bottomTrailingRadius: cornerRadius))

        let hosting = NSHostingView(rootView: shaped)
        hosting.frame = NSRect(x: 0, y: 0, width: frame.width, height: frame.height)
        hosting.autoresizingMask = [.width, .height]
        return hosting
    }
}
