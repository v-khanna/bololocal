import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var popoverController: PopoverController!
    let hotkeyManager = HotkeyManager()
    private var settingsWindow: NSWindow?
    private var onboardingWindow: NSWindow?
    private var capsuleController: CapsuleWindowController?
    var coordinator: Coordinator?
    var modelManager: ModelManager<ChatterboxPipeline>?
    var downloadProgress: ModelDownloadProgress?

    func applicationDidFinishLaunching(_ notification: Notification) {
        Settings.shared.load()

        // Status item
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Bolo")
        icon?.isTemplate = true
        item.button?.image = icon
        item.button?.action = #selector(togglePopover)
        item.button?.target = self
        self.statusItem = item

        // Pipeline — model is lazy-loaded on first ⌘⇧R; unloaded after 5 min idle.
        let progress = ModelDownloadProgress()
        self.downloadProgress = progress

        // Production runs at 4-bit. An isolated benchmark showed 4-bit ~0.91×
        // (10% slower) than fp16 on an EMPTY machine — but that measured the
        // wrong thing. On a real 16 GB Mac under load (browser, calls, other
        // menu-bar AI apps), the ~3 GB fp16 model + MLX inference spikes
        // exhaust memory and trigger a system-wide jetsam cascade (observed
        // 2026-05-27 21:41 — cfprefsd/installd/etc. all jettisoned, machine
        // froze). 4-bit cuts the resident model ~4× (≈750 MB), which keeps the
        // system alive. Memory headroom beats 10% speed every time on 16 GB.
        // (See ChatterboxPipeline.applyQuantization — hybrid two-pass that
        // handles the `[Module]` array wrappers. quality verified: peak 0.24.)
        let quantizeBits: Int? = 4
        let modeLabel = quantizeBits.map { "\($0)-bit" } ?? "fp16"
        // ModelManager is created regardless, but it only downloads/loads the
        // multi-GB model when the LOCAL engine actually calls ensureLoaded().
        // With the cloud default, that never happens unless the user opts in.
        let manager = ModelManager<ChatterboxPipeline>(idleTimeout: 300) {
            try await ChatterboxPipeline.load(
                progressHandler: { p, label in
                    Task { @MainActor in
                        progress.update(progress: p, label: "\(label) [\(modeLabel)]")
                        if p >= 1.0 { progress.complete() }
                    }
                },
                quantizeBits: quantizeBits
            )
        }
        self.modelManager = manager

        // Engine selection. DEFAULT = cloud (Groq); on-device Chatterbox is
        // opt-in via Settings → "Use on-device voice". The local engine lazy-
        // loads its model only when first used, so cloud-default users never
        // trigger the big download.
        let engine: any TTSEngine
        if Settings.shared.useLocalEngine {
            engine = ChatterboxTTSEngine(modelProvider: {
                try await manager.ensureLoaded()
            })
        } else {
            engine = GroqTTSEngine(configProvider: {
                await MainActor.run {
                    (Settings.shared.groqAPIKey, Settings.shared.cloudVoice)
                }
            })
        }
        let playback = PlaybackController(engine: engine)
        let coordinator = Coordinator(hotkey: hotkeyManager, playback: playback)
        coordinator.start()
        self.coordinator = coordinator
        self.modelManager = manager

        // Popover (depends on coordinator.state)
        self.popoverController = PopoverController(
            settings: Settings.shared,
            coordinatorState: coordinator.state,
            onOpenSettings: { [weak self] in self?.openSettings() }
        )

        // Floating reading capsule — Freeflow-style 185×71 pill that shows
        // below the menu bar whenever audio is playing. Owns its own observer
        // on `coordinator.state.isPlaying` and shows/hides automatically.
        self.capsuleController = CapsuleWindowController(
            coordinatorState: coordinator.state,
            onStop: { coordinator.state.stop() }
        )

        // First-run onboarding
        if !Settings.shared.hasCompletedOnboarding {
            showOnboarding()
        }
    }

    @objc private func togglePopover() {
        guard let button = statusItem?.button else { return }
        if popoverController.popover.isShown {
            popoverController.hide()
        } else {
            popoverController.show(relativeTo: button)
        }
    }

    @objc func openSettings() {
        if let win = settingsWindow {
            win.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let view = SettingsView(settings: Settings.shared)
        let host = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: host)
        window.title = "Bolo Settings"
        window.styleMask = [.titled, .closable]
        window.isReleasedWhenClosed = false
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        settingsWindow = window
    }

    private func showOnboarding() {
        guard let downloadProgress = downloadProgress, let modelManager = modelManager else { return }
        let view = OnboardingView(
            downloadProgress: downloadProgress,
            onStartDownload: {
                Task { @MainActor in
                    do {
                        _ = try await modelManager.ensureLoaded()
                        downloadProgress.complete()
                    } catch {
                        downloadProgress.fail(error.localizedDescription)
                    }
                }
            },
            onComplete: { [weak self] in
                Settings.shared.hasCompletedOnboarding = true
                self?.onboardingWindow?.close()
                self?.onboardingWindow = nil
            }
        )
        let host = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: host)
        window.title = "Welcome to Bolo"
        window.styleMask = [.titled, .closable]
        window.isReleasedWhenClosed = false
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        onboardingWindow = window
    }
}
