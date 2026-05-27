import AppKit
@preconcurrency import Qwen3TTS
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var popoverController: PopoverController!
    let hotkeyManager = HotkeyManager()
    private var settingsWindow: NSWindow?
    private var onboardingWindow: NSWindow?
    var coordinator: Coordinator?
    var modelManager: ModelManager<Qwen3TTSModel>?
    var downloadProgress: ModelDownloadProgress?

    func applicationDidFinishLaunching(_ notification: Notification) {
        Settings.shared.load()

        // Status item
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
        icon?.isTemplate = true
        item.button?.image = icon
        item.button?.action = #selector(togglePopover)
        item.button?.target = self
        self.statusItem = item

        // Pipeline — model is lazy-loaded on first ⌘⇧R; unloaded after 5 min idle.
        let progress = ModelDownloadProgress()
        self.downloadProgress = progress

        let manager = ModelManager<Qwen3TTSModel>(idleTimeout: 300) {
            try await Qwen3TTSModel.fromPretrained(progressHandler: { p, label in
                Task { @MainActor in
                    progress.update(progress: p, label: label)
                    if p >= 1.0 { progress.complete() }
                }
            })
        }
        let engine: any TTSEngine = Qwen3TTSEngine(modelProvider: {
            try await manager.ensureLoaded()
        })
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
        window.title = "HearIt Settings"
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
        window.title = "Welcome to HearIt"
        window.styleMask = [.titled, .closable]
        window.isReleasedWhenClosed = false
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        onboardingWindow = window
    }
}
