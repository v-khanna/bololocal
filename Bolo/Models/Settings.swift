import Foundation
import Combine

/// Singleton settings backed by UserDefaults. SwiftUI views observe via @ObservedObject.
/// Reset for tests via `Settings.shared.reset()`.
@MainActor
final class Settings: ObservableObject {
    static let shared = Settings()

    private enum Key {
        static let language = "bolo.language"
        static let speed = "bolo.speed"
        static let launchAtLogin = "bolo.launchAtLogin"
        static let hasCompletedOnboarding = "bolo.hasCompletedOnboarding"
        static let showCapsuleWhileReading = "bolo.showCapsuleWhileReading"
        static let useLocalEngine = "bolo.useLocalEngine"
        static let groqAPIKey = "bolo.groqAPIKey"
        static let cloudVoice = "bolo.cloudVoice"
    }

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// One of the 10 Qwen3 languages — v1 defaults to English; picker in popover (Task 11).
    @Published var selectedLanguage: String = "english" {
        didSet { defaults.set(selectedLanguage, forKey: Key.language) }
    }

    @Published var speed: Speed = Speed(1.0) {
        didSet { defaults.set(speed.value, forKey: Key.speed) }
    }

    @Published var launchAtLogin: Bool = false {
        didSet { defaults.set(launchAtLogin, forKey: Key.launchAtLogin) }
    }

    @Published var hasCompletedOnboarding: Bool = false {
        didSet { defaults.set(hasCompletedOnboarding, forKey: Key.hasCompletedOnboarding) }
    }

    /// Whether the floating playback capsule is shown during reading. Defaults to on.
    @Published var showCapsuleWhileReading: Bool = true {
        didSet { defaults.set(showCapsuleWhileReading, forKey: Key.showCapsuleWhileReading) }
    }

    /// Engine selection. DEFAULT is cloud (Groq) — fast, no model download.
    /// Flip on for the on-device Chatterbox engine (private/offline, but
    /// downloads a multi-GB model on first use). Opt-in only.
    @Published var useLocalEngine: Bool = false {
        didSet { defaults.set(useLocalEngine, forKey: Key.useLocalEngine) }
    }

    /// Groq API key for cloud TTS (also works for FreeFlow-style dictation).
    /// Stored in UserDefaults for now — TODO: move to Keychain before ship.
    @Published var groqAPIKey: String = "" {
        didSet { defaults.set(groqAPIKey, forKey: Key.groqAPIKey) }
    }

    /// Orpheus voice name for cloud TTS (e.g. "troy", "hannah", "austin").
    @Published var cloudVoice: String = "troy" {
        didSet { defaults.set(cloudVoice, forKey: Key.cloudVoice) }
    }

    func load() {
        if let lang = defaults.string(forKey: Key.language) { selectedLanguage = lang }
        let storedSpeed = defaults.object(forKey: Key.speed) as? Double ?? 1.0
        speed = Speed(storedSpeed)
        launchAtLogin = defaults.bool(forKey: Key.launchAtLogin)
        hasCompletedOnboarding = defaults.bool(forKey: Key.hasCompletedOnboarding)
        // Default to true when the key has never been written.
        showCapsuleWhileReading = defaults.object(forKey: Key.showCapsuleWhileReading) as? Bool ?? true
        // Cloud engine is the default; local is opt-in.
        useLocalEngine = defaults.bool(forKey: Key.useLocalEngine)
        groqAPIKey = defaults.string(forKey: Key.groqAPIKey) ?? ""
        cloudVoice = defaults.string(forKey: Key.cloudVoice) ?? "troy"
    }

    /// Test-only: clear all keys and reset to defaults.
    func reset() {
        [Key.language, Key.speed, Key.launchAtLogin, Key.hasCompletedOnboarding,
         Key.showCapsuleWhileReading, Key.useLocalEngine, Key.groqAPIKey, Key.cloudVoice]
            .forEach { defaults.removeObject(forKey: $0) }
        selectedLanguage = "english"
        speed = Speed(1.0)
        launchAtLogin = false
        hasCompletedOnboarding = false
        showCapsuleWhileReading = true
        useLocalEngine = false
        groqAPIKey = ""
        cloudVoice = "troy"
    }
}
