import AppKit

@MainActor
final class Coordinator {
    let state = CoordinatorState()
    private let hotkey: HotkeyManager
    private let playback: PlaybackController

    init(hotkey: HotkeyManager, playback: PlaybackController) {
        self.hotkey = hotkey
        self.playback = playback
        state.togglePlayPause = { [weak self] in self?.togglePlayPause() }
        state.stop = { [weak self] in self?.stopPlayback() }
    }

    func start() {
        hotkey.register { [weak self] in
            MainActor.assumeIsolated { self?.handleHotkey() }
        }
    }

    private func handleHotkey() {
        guard PermissionsManager.isAccessibilityGranted else {
            PermissionsManager.openAccessibilitySettings()
            return
        }
        guard let text = TextCaptureManager.captureSelectedText(), !text.isEmpty else {
            NSLog("HearIt: nothing selected")
            return
        }
        state.lastCapturedText = text
        startPlayback(text: text)
    }

    private func togglePlayPause() {
        if state.isPlaying {
            stopPlayback()
        } else if !state.lastCapturedText.isEmpty {
            startPlayback(text: state.lastCapturedText)
        }
    }

    private func startPlayback(text: String) {
        let s = Settings.shared
        let voice = VoiceID(rawValue: s.selectedLanguage)
        state.isPlaying = true
        playback.play(text: text, voice: voice, speed: s.speed) { [weak self] in
            Task { @MainActor in self?.state.isPlaying = false }
        }
    }

    private func stopPlayback() {
        playback.stop()
        state.isPlaying = false
    }
}
