import Foundation

/// Generic actor that owns a model's lifecycle: lazy-load on first use, auto-unload after idle timeout.
///
/// Usage:
/// ```swift
/// let manager = ModelManager<Qwen3TTSModel>(idleTimeout: 300) {
///     try await Qwen3TTSModel.fromPretrained()
/// }
/// let model = try await manager.ensureLoaded()
/// ```
// ModelManager is an actor, which is inherently Sendable. The `Model` generic parameter
// does not need to be `Sendable` because access is always serialized through the actor.
// We mark `@unchecked Sendable` so the type can be passed across isolation domains
// (e.g. stored on @MainActor AppDelegate). The actor guarantees thread safety.
actor ModelManager<Model>: @unchecked Sendable {

    enum State: Equatable {
        case unloaded
        case loading
        case loaded
    }

    private(set) var state: State = .unloaded
    private var model: Model?
    private var idleTask: Task<Void, Never>?

    private let idleTimeout: TimeInterval
    // The loader closure is stored and called only within the actor's isolation domain.
    // We use `nonisolated(unsafe)` to suppress Sendable checking on the closure itself —
    // safety is guaranteed by the actor's serial executor (only one call at a time).
    nonisolated(unsafe) private let loader: () async throws -> Model

    init(
        idleTimeout: TimeInterval = 300,
        loader: sending @escaping () async throws -> Model
    ) {
        self.idleTimeout = idleTimeout
        self.loader = loader
    }

    /// Returns the loaded model, loading it on first call (or after an idle unload).
    func ensureLoaded() async throws -> Model {
        if let m = model {
            resetIdleTimer()
            return m
        }
        state = .loading
        do {
            let m = try await loader()
            self.model = m
            self.state = .loaded
            resetIdleTimer()
            return m
        } catch {
            self.state = .unloaded
            throw error
        }
    }

    /// Mark the model as recently used — resets the idle countdown.
    func touch() {
        resetIdleTimer()
    }

    /// Immediately drop the loaded model (frees memory; next ensureLoaded() re-loads).
    func unload() {
        idleTask?.cancel()
        idleTask = nil
        model = nil
        state = .unloaded
    }

    // MARK: - Private

    private func resetIdleTimer() {
        idleTask?.cancel()
        let timeout = idleTimeout
        idleTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            } catch {
                // Task was cancelled (e.g. by a subsequent touch or explicit unload). Stop here.
                return
            }
            await self?.unload()
        }
    }
}
