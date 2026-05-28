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
        state.speakTest = { [weak self] in self?.speakTest() }
    }

    /// DEV: speak a hardcoded sentence. Bypasses hotkey + AX + clipboard.
    private func speakTest() {
        BoloDebug.log("speakTest invoked")
        let text = "Hello, this is a test of Bolo's audio pipeline. If you hear this, the engine works."
        state.lastCapturedText = text
        startPlayback(text: text)
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
            NSLog("Bolo: nothing selected")
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
        BoloDebug.log("startPlayback: text=\(text.prefix(40))… voice=\(s.selectedLanguage) speed=\(s.speed.value)")
        state.isPlaying = true
        playback.play(text: text, voice: voice, speed: s.speed) { [weak self] in
            BoloDebug.log("playback callback fired (done or error)")
            Task { @MainActor in self?.state.isPlaying = false }
        }
    }

    private func stopPlayback() {
        playback.stop()
        state.isPlaying = false
    }
}
