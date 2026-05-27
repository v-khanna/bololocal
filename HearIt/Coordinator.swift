import AppKit

@MainActor
final class Coordinator {
    private let hotkey: HotkeyManager
    private let playback: PlaybackController

    init(hotkey: HotkeyManager, playback: PlaybackController) {
        self.hotkey = hotkey
        self.playback = playback
    }

    func start() {
        hotkey.register { [weak self] in
            MainActor.assumeIsolated {
                self?.handleHotkey()
            }
        }
    }

    private func handleHotkey() {
        guard PermissionsManager.isAccessibilityGranted else {
            PermissionsManager.openAccessibilitySettings()
            return
        }
        guard let text = TextCaptureManager.captureSelectedText(),
              !text.isEmpty else {
            NSLog("HearIt: nothing selected")
            return
        }
        let s = Settings.shared
        // VoiceID.rawValue carries the language string for Qwen3TTSEngine.
        let voice = VoiceID(rawValue: s.selectedLanguage)
        playback.play(text: text, voice: voice, speed: s.speed)
    }
}
