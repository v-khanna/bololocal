import Foundation
import Combine

/// Singleton settings backed by UserDefaults. SwiftUI views observe via @ObservedObject.
/// Reset for tests via `Settings.shared.reset()`.
@MainActor
final class Settings: ObservableObject {
    static let shared = Settings()

    private enum Key {
        static let language = "hearit.language"
        static let speed = "hearit.speed"
        static let launchAtLogin = "hearit.launchAtLogin"
        static let hasCompletedOnboarding = "hearit.hasCompletedOnboarding"
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

    func load() {
        if let lang = defaults.string(forKey: Key.language) { selectedLanguage = lang }
        let storedSpeed = defaults.object(forKey: Key.speed) as? Double ?? 1.0
        speed = Speed(storedSpeed)
        launchAtLogin = defaults.bool(forKey: Key.launchAtLogin)
        hasCompletedOnboarding = defaults.bool(forKey: Key.hasCompletedOnboarding)
    }

    /// Test-only: clear all keys and reset to defaults.
    func reset() {
        [Key.language, Key.speed, Key.launchAtLogin, Key.hasCompletedOnboarding]
            .forEach { defaults.removeObject(forKey: $0) }
        selectedLanguage = "english"
        speed = Speed(1.0)
        launchAtLogin = false
        hasCompletedOnboarding = false
    }
}
