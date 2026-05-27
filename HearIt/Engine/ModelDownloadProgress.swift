import Foundation
import Combine

/// SwiftUI-observable wrapper around speech-swift's progress callback.
/// Pass an instance into ModelManager's loader closure; the closure forwards
/// progress events here, and the OnboardingView observes them.
@MainActor
final class ModelDownloadProgress: ObservableObject {
    @Published var progress: Double = 0
    @Published var label: String = "Preparing…"
    @Published var error: String?
    @Published var isComplete: Bool = false

    func update(progress: Double, label: String) {
        self.progress = progress
        self.label = label
    }

    func fail(_ message: String) {
        self.error = message
    }

    func complete() {
        self.progress = 1.0
        self.isComplete = true
    }

    func reset() {
        progress = 0
        label = "Preparing…"
        error = nil
        isComplete = false
    }
}
