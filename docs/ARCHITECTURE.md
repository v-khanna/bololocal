# Bolo — Architecture

This document explains how Bolo works, top to bottom. It covers the high-level
flow when you press ⌘⇧R, then drills into each component, the concurrency
model, the model lifecycle, the audio pipeline, and the design rationale
behind major choices.

If you want practical build/test/release instructions instead, see
[DEVELOPING.md](DEVELOPING.md) and [RELEASE.md](RELEASE.md).

---

## 1. What Bolo is

A macOS menu bar app that reads selected text aloud in a natural AI voice
running entirely on your Mac.

- **Trigger**: ⌘⇧R while text is selected in any app.
- **Engine**: [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
  (MLX 4-bit quantized variant from `aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit`,
  ~500 MB) via the [soniqo/speech-swift](https://github.com/soniqo/speech-swift)
  Swift Package. Runs on the Apple Neural Engine / GPU through MLX.
- **Network**: exactly one HTTP request, ever — the first-run download of the
  voice weights from Hugging Face. After that: zero network calls.
- **Idle cost**: under ~50 MB RAM when not speaking. The model object is
  unloaded after 5 minutes of inactivity.
- **Platform**: Apple Silicon only, macOS 15+.

---

## 2. The hot path — what happens when you press ⌘⇧R

A single user action — ⌘⇧R with text selected in Safari — traverses the entire
stack. Following it end-to-end is the fastest way to understand the system.

### 2.1 Sequence diagram

```
┌────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐
│  User  │    │ HotKey SPM  │    │ HotkeyManager │    │   Coordinator   │    │TextCaptureMgr│    │PlaybackController│    │  Qwen3TTSEngine  │
│        │    │  (Carbon)   │    │  @MainActor   │    │   @MainActor    │    │   (enum)     │    │   @MainActor   │    │     (actor)      │
└───┬────┘    └──────┬──────┘    └───────┬───────┘    └────────┬────────┘    └──────┬───────┘    └───────┬────────┘    └─────────┬────────┘
    │    ⌘⇧R         │                   │                     │                    │                    │                       │
    │───────────────►│                   │                     │                    │                    │                       │
    │                │  keyDownHandler   │                     │                    │                    │                       │
    │                │──────────────────►│                     │                    │                    │                       │
    │                │                   │ callback()          │                    │                    │                       │
    │                │                   │────────────────────►│                    │                    │                       │
    │                │                   │                     │ handleHotkey()     │                    │                       │
    │                │                   │                     │───────────────────►│ captureSelectedText│                       │
    │                │                   │                     │                    │ ─► AX API          │                       │
    │                │                   │                     │◄───────────────────│   ─► clipboard fb  │                       │
    │                │                   │                     │  String?           │                    │                       │
    │                │                   │                     │                                         │                       │
    │                │                   │                     │  state.lastCapturedText = text          │                       │
    │                │                   │                     │  state.isPlaying = true                 │                       │
    │                │                   │                     │  play(text, voice, speed, onComplete)   │                       │
    │                │                   │                     │────────────────────────────────────────►│                       │
    │                │                   │                     │                                         │ Task: synthesize(...) │
    │                │                   │                     │                                         │──────────────────────►│
    │                │                   │                     │                                         │                       │  modelProvider() ──► ModelManager.ensureLoaded()
    │                │                   │                     │                                         │                       │  (lazy-load model on first call, ~500MB DL on first ever)
    │                │                   │                     │                                         │                       │  model.synthesize(text:language:) ──► [Float] @ 24kHz mono
    │                │                   │                     │                                         │                       │  wrap into AVAudioPCMBuffer
    │                │                   │                     │                                         │                       │  AVAudioEngine → AVAudioPlayerNode → AVAudioUnitVarispeed → mainMixer
    │                │                   │                     │                                         │                       │  await scheduleBuffer completion
    │                │                   │                     │                                         │◄──────────────────────│
    │                │                   │                     │                                         │ onComplete() fires    │
    │                │                   │                     │◄────────────────────────────────────────│                       │
    │                │                   │                     │  state.isPlaying = false                │                       │
    │◄──────────────────── audio playback through system output ─────────────────────────────────────────────────────────────────┤
```

### 2.2 Step-by-step narrative

1. **`HotKey` package (`HotKey` SPM dep)** registers a Carbon `EventHotKey`
   for ⌘⇧R at app launch via `HotkeyManager.register(handler:)`. The Carbon
   API is the macOS primitive for global, system-wide keyboard shortcuts that
   work even when our app isn't focused.

2. When you press ⌘⇧R, Carbon dispatches to `HotKey`'s `keyDownHandler`,
   which invokes the callback stored in `HotkeyManager.callback`. The
   callback was set by `Coordinator.start()` to invoke
   `Coordinator.handleHotkey()` on the main actor.

3. **`Coordinator.handleHotkey()`** runs three guards in order:
   - Accessibility granted? If not, opens System Settings and bails.
   - Selected text non-empty? `TextCaptureManager.captureSelectedText()`
     tries `AXUIElementCopyAttributeValue(focused, kAXSelectedTextAttribute)`
     against the frontmost app's focused element. If that returns nothing
     (e.g. Terminal has poor AX support), it falls back to the clipboard's
     current `public.utf8-plain-text`.
   - Both guards pass → store the text in `CoordinatorState.lastCapturedText`
     (which the popover observes) and call `startPlayback(text:)`.

4. **`startPlayback`** reads `Settings.shared.selectedLanguage` and
   `Settings.shared.speed`, wraps the language in a `VoiceID`
   (`VoiceID.rawValue` is the language string for the Qwen3 path), flips
   `state.isPlaying = true` so the popover UI updates, then calls
   `PlaybackController.play(text:, voice:, speed:, onComplete:)` with a
   completion callback that flips `state.isPlaying = false`.

5. **`PlaybackController.play`** cancels any in-flight task, then spawns a
   new `Task` that awaits
   `engine.synthesize(text: voice: speed:)` on the engine (`Qwen3TTSEngine`
   in production). On completion (success or error), it fires the
   `onComplete` callback and clears `currentTask` back on the main actor.

6. **`Qwen3TTSEngine.synthesize`** (an `actor` method) calls
   `modelProvider()` which is `{ try await modelManager.ensureLoaded() }`.

7. **`ModelManager.ensureLoaded()`** — on first ever press, the model isn't
   cached, so it calls the loader closure:

   ```swift
   try await Qwen3TTSModel.fromPretrained(progressHandler: { p, label in
       Task { @MainActor in
           progress.update(progress: p, label: label)
           if p >= 1.0 { progress.complete() }
       }
   })
   ```

   `fromPretrained` checks the local cache at `~/Documents/huggingface/...`,
   downloads ~500 MB of weights if absent, and constructs the
   `Qwen3TTSModel`. On every subsequent call, the model is cached in the
   `ModelManager` actor's `model: Model?` field — `ensureLoaded` returns
   it immediately and resets the 5-minute idle timer.

8. Back in `Qwen3TTSEngine._synthesize`, with the model in hand, we call
   the synchronous `model.synthesize(text:, language:)`. It returns
   `[Float]` — raw 24 kHz mono float samples.

9. **The audio graph** is set up lazily once per actor instance:
   `AVAudioEngine` → `AVAudioPlayerNode` → `AVAudioUnitVarispeed`
   → `mainMixerNode`. The float samples are copied into an
   `AVAudioPCMBuffer` (one channel, float32, 24 kHz). `varispeed.rate` is
   set to `Float(speed.value)` — 0.5x to 2.0x. `playerNode.play()` starts
   playback, then we `await` a `CheckedContinuation` that resumes from
   `scheduleBuffer(completionCallbackType: .dataPlayedBack)`.

10. When playback completes, the continuation resumes,
    `Qwen3TTSEngine.synthesize` returns, the `Task` in `PlaybackController`
    fires `onComplete`, and `Coordinator.state.isPlaying` flips to `false`.
    The popover's play/pause button SwiftUI binding updates immediately.

That's the whole hot path. About 8–10 layers, each tightly focused.

---

## 3. Subsystem map

Eighteen Swift files, 1,306 lines of code, organized by responsibility:

```
Bolo/
├── BoloApp.swift                      App entry — @NSApplicationDelegateAdaptor wiring
├── AppDelegate.swift                  Dependency injection root, lifecycle owner
├── Coordinator.swift                  Glues hotkey → capture → playback
│
├── Hotkey/
│   └── HotkeyManager.swift           ⌘⇧R via the HotKey SPM package (Carbon API)
│
├── Capture/
│   ├── PermissionsManager.swift      AX trust check + System Settings deep-link
│   └── TextCaptureManager.swift      AXUIElement + clipboard fallback
│
├── Engine/
│   ├── TTSEngine.swift               Protocol + VoiceID + Speed + TTSError
│   ├── MockTTSEngine.swift           AVSpeechSynthesizer-backed; left for tests/debug
│   ├── Qwen3TTSEngine.swift          Production engine (Qwen3 via speech-swift)
│   ├── ModelManager.swift            Lazy-load + idle-unload actor over Qwen3TTSModel
│   └── ModelDownloadProgress.swift   @MainActor observable for the progress bar
│
├── Playback/
│   └── PlaybackController.swift      Owns the engine, manages current Task
│
├── Models/
│   └── Settings.swift                Singleton @MainActor ObservableObject + UserDefaults
│
├── LaunchAtLogin.swift                SMAppService.mainApp wrapper
│
├── UI/
│   ├── PopoverController.swift       NSPopover lifecycle + VisualEffectBackground
│   ├── PopoverView.swift             SwiftUI popover content + CoordinatorState
│   ├── SettingsView.swift            Two-tab settings window (General + About)
│   └── OnboardingView.swift          Four-step first-run flow
│
└── Resources/
    └── (Info.plist, entitlements generated by xcodegen — not in repo)
```

### 3.1 Layering

Bolo is roughly a three-layer app:

| Layer        | Responsibility                                              | Files                                                                              |
| ------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Platform** | macOS APIs — AppKit, AX, AVFoundation, ServiceManagement    | AppDelegate, Hotkey/, Capture/, LaunchAtLogin                                      |
| **Engine**   | TTS synthesis + model lifecycle + audio playback            | Engine/, Playback/                                                                 |
| **UI/State** | SwiftUI views + user-facing state                           | UI/, Models/, Coordinator (state bridge)                                           |

`Coordinator` is the seam between platform events (hotkey, AX capture) and
engine work, while exposing a thin observable `CoordinatorState` to the UI.
The UI never talks to the engine directly.

---

## 4. Component reference (low-level)

Each component is small. The full surface area is summarized below — read the
source files for the actual implementations.

### 4.1 `BoloApp.swift`

The SwiftUI `@main`. Its only job is to install `AppDelegate` via
`@NSApplicationDelegateAdaptor` and provide an empty `SwiftUI.Settings`
scene (the SwiftUI scene-builder, not our `Settings` class — they collide
namespacing-wise; we qualify as `SwiftUI.Settings { EmptyView() }`).

There's no main window. `LSUIElement = true` in `Info.plist` keeps Bolo out
of the Dock and the standard menu bar.

### 4.2 `AppDelegate.swift` — the composition root

Marked `@MainActor`. Holds strong references to every long-lived object
the app needs:

- `statusItem: NSStatusItem?`
- `popoverController: PopoverController!`
- `hotkeyManager: HotkeyManager`
- `settingsWindow / onboardingWindow: NSWindow?`
- `coordinator: Coordinator?`
- `modelManager: ModelManager<Qwen3TTSModel>?`
- `downloadProgress: ModelDownloadProgress?`

`applicationDidFinishLaunching` does five things in order:

1. `Settings.shared.load()` — hydrate `@Published` properties from
   UserDefaults.
2. Create the menu bar status item with the SF Symbol `waveform` and the
   `togglePopover` action.
3. Build the engine pipeline:
   `ModelDownloadProgress → ModelManager<Qwen3TTSModel> → Qwen3TTSEngine
   → PlaybackController → Coordinator → coordinator.start()`. The
   `ModelManager` loader closure captures `progress` so first-run download
   updates the UI in real time.
4. Construct `PopoverController` with the `Settings`, `coordinator.state`,
   and an `onOpenSettings` closure.
5. If `!Settings.shared.hasCompletedOnboarding`, open the onboarding
   window via `showOnboarding()`.

It also owns two `@objc` callbacks: `togglePopover` (status item click)
and `openSettings` (gear icon from popover, also exposed as the
`onOpenSettings` closure to SwiftUI).

### 4.3 `Coordinator.swift`

Marked `@MainActor`. Holds:

- `state: CoordinatorState` (defined in `PopoverView.swift`) — the bridge
  to SwiftUI.
- `hotkey: HotkeyManager` (constructor-injected).
- `playback: PlaybackController` (constructor-injected).

`start()` registers the hotkey with a callback that calls
`handleHotkey()` via `MainActor.assumeIsolated`. The hotkey's underlying
`HotKey.keyDownHandler` is `(@Sendable () -> Void)?` — we know it fires
on the main thread but Swift 6 can't prove it, so the
`MainActor.assumeIsolated` is a deliberate trust boundary.

`handleHotkey`:
- Checks `PermissionsManager.isAccessibilityGranted`; if false, opens
  System Settings and returns.
- Calls `TextCaptureManager.captureSelectedText()`; if nil/empty, logs
  and returns.
- Updates `state.lastCapturedText`, calls `startPlayback`.

`startPlayback` reads `Settings.shared` and calls
`playback.play(...)` with a completion closure that flips
`state.isPlaying` back to false.

`togglePlayPause` and `stopPlayback` are exposed via closures on
`CoordinatorState` so the SwiftUI popover can call them without
importing AppKit.

### 4.4 `HotkeyManager.swift`

Marked `@MainActor`. Thin wrapper around the
[HotKey](https://github.com/soffes/HotKey) SPM package, which itself
wraps the Carbon `RegisterEventHotKey` API. Stores a `callback` so we
can call `fire()` from unit tests without actually pressing keys.

### 4.5 `PermissionsManager.swift` + `TextCaptureManager.swift`

`PermissionsManager` is an `enum` (no instances). Three static methods:

- `isAccessibilityGranted` — `AXIsProcessTrusted()`, no prompt.
- `requestAccessibility()` — `AXIsProcessTrustedWithOptions` with prompt
  flag. Triggers the one-time system prompt the user can only see once
  per install.
- `openAccessibilitySettings()` — deep-link via
  `x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility`.

Imports `ApplicationServices` with `@preconcurrency` because the C
global `kAXTrustedCheckOptionPrompt` is imported as a `var`, which trips
Swift 6 strict concurrency.

`TextCaptureManager` is also an `enum`. Two paths:

- **AX path** (`captureFromAccessibility`): Get the system-wide AX
  element, ask for `kAXFocusedUIElementAttribute`, then ask that
  element for `kAXSelectedTextAttribute`. Returns the string if
  non-empty, else nil. Works in Safari, Mail, Pages, TextEdit, Notes,
  most native + AX-compliant apps.
- **Clipboard fallback**: `NSPasteboard.general.string(forType: .string)`.
  Used when AX returns nothing (Terminal, some Electron apps, web
  views).

`captureSelectedText()` tries AX first, falls back to clipboard.

### 4.6 `TTSEngine.swift` — the protocol

```swift
protocol TTSEngine: Sendable {
    func synthesize(text: String, voice: VoiceID, speed: Speed) async throws
    func stop()
}
```

`VoiceID` is a `RawRepresentable<String>` opaque identifier. In the
Qwen3 path, the `rawValue` is a language string ("english", "chinese",
etc.). The `systemDefault` constant maps to "english".

`Speed` is a `Double` clamped to `[0.5, 2.0]` at init time. Speed
control is implemented in the audio graph via `AVAudioUnitVarispeed`,
not in the engine itself.

`TTSError` enumerates `modelNotLoaded`, `synthesisFailed(String)`,
`playbackFailed(String)`, `emptyText`.

### 4.7 `Qwen3TTSEngine.swift` — the production engine

A Swift `actor` (not a class) for two reasons:

1. `Qwen3TTSModel` from `speech-swift` is **not** `Sendable`. Wrapping
   it in an actor serializes access through the actor's executor.
2. The model is single-threaded internally — concurrent
   `synthesize` calls would corrupt MLX state. Actor serialization
   gives us this for free.

`init(modelProvider: @escaping @Sendable () async throws -> Qwen3TTSModel)`
takes a closure rather than the model directly. This decouples engine
lifecycle from model lifecycle — `ModelManager` owns the model;
`Qwen3TTSEngine` just asks for the current instance.

`synthesize` (the `TTSEngine` conformance) is `nonisolated` and
delegates to `_synthesize` on the actor. This keeps the protocol's
`async throws` shape without forcing the protocol itself to be
actor-bound.

The synthesize flow inside the actor:

1. `try await modelProvider()` — gets the loaded model (or triggers
   first-run download).
2. `Self.languageString(for: voice)` — maps the `VoiceID.rawValue` to
   one of Qwen3's 10 supported language strings, defaulting to
   "english" on unknown values.
3. `let samples: [Float] = model.synthesize(text:, language:)` — the
   one synchronous CPU-heavy call. Returns 24 kHz mono float samples.
4. `try await play(samples:, sampleRate:, speed:)` — wrap into
   `AVAudioPCMBuffer`, schedule, await completion.

The audio graph is created lazily on first synthesize and reused:

```
AVAudioPlayerNode → AVAudioUnitVarispeed → mainMixerNode
```

`Varispeed` does the speed shift; we set `rate = Float(speed.value)`
before each playback.

`stop()` is `nonisolated` and spawns an internal task to call `_stop`
on the actor (which stops `playerNode` and `engine`). This lets
callers fire-and-forget the stop signal.

### 4.8 `MockTTSEngine.swift`

`AVSpeechSynthesizer`-backed implementation of `TTSEngine`. Not used in
the production wiring anymore, but kept because:

- It's a useful reference for the protocol's shape.
- Tests can use it for fast, deterministic playback exercises (the
  built-in macOS voice doesn't download anything).
- If Qwen3 ever fails catastrophically, the app can fall back to
  AVSpeechSynthesizer with a one-line change in `AppDelegate`.

### 4.9 `ModelManager.swift`

A generic `actor ModelManager<Model>: @unchecked Sendable` (the
`@unchecked` is because the `Model` generic isn't required to be
`Sendable` — the actor's serial executor guarantees safety).

State machine:

```
                  ensureLoaded() (first call)
       ┌──────────┐                ┌──────────┐
       │ unloaded │ ───────────►   │ loading  │
       └──────────┘                └──────────┘
            ▲                            │
            │                            │ loader closure resolves
            │ unload() (manual or idle)  ▼
            │                       ┌──────────┐
            └────────────────────── │  loaded  │
                                    └──────────┘
                                         ▲
                                  (touch resets idle timer)
```

Public API:

- `ensureLoaded() async throws -> Model` — returns the cached model or
  triggers the loader closure. Resets the idle timer.
- `touch()` — resets the idle timer without loading.
- `unload()` — immediately frees the model.
- `state: State` — `unloaded | loading | loaded`.

Internals:

- `idleTask: Task<Void, Never>?` — a background task that sleeps for
  `idleTimeout` seconds then calls `unload()`. Cancelled and
  recreated on every `resetIdleTimer()`. Crucially, the
  `Task.sleep` is wrapped in `do/catch` (not `try?`) so a
  cancellation does NOT proceed to `unload()` — the original draft
  had this as `try? await Task.sleep` which masked the cancellation
  and broke the touch-resets-timer behavior. The bug was caught in
  Task 9's TDD pass.
- `loader: nonisolated(unsafe) () async throws -> Model` — stored
  closure, called only inside the actor's executor.
- `init(idleTimeout:loader:)` — `loader` is `sending` so the closure
  can move from `@MainActor` (where `AppDelegate` constructs it) into
  the actor's isolation domain.

### 4.10 `ModelDownloadProgress.swift`

A `@MainActor final class ObservableObject` with four `@Published`
fields: `progress: Double`, `label: String`, `error: String?`,
`isComplete: Bool`. SwiftUI's `OnboardingView` observes it and
animates the progress bar.

`AppDelegate.applicationDidFinishLaunching` wires
`Qwen3TTSModel.fromPretrained(progressHandler: ...)` to call
`progress.update(...)` and `progress.complete()` via
`Task { @MainActor in ... }`. The `progressHandler` closure is
`((Double, String) -> Void)?` and is called by speech-swift during
download with values like `(0.42, "Downloading model.safetensors")`.

### 4.11 `PlaybackController.swift`

A `@MainActor final class` that owns the current playback `Task` and
the engine instance. Two methods:

- `play(text:voice:speed:onComplete:)` — cancels any current task,
  spawns a new one that calls `engine.synthesize`. After synthesis
  completes (or errors), invokes `onComplete?()` and clears
  `currentTask` back on the main actor.
- `stop()` — calls `engine.stop()`, cancels `currentTask`, clears it.

`isPlaying: Bool { currentTask != nil }` is a derived state. The
boolean we surface to SwiftUI lives in `CoordinatorState.isPlaying`
instead, because SwiftUI needs an `@Published` for diffing — and
`Coordinator` knows when playback actually finishes via the
`onComplete` callback.

### 4.12 `Settings.swift`

A `@MainActor final class Settings: ObservableObject` with four
`@Published` properties:

| Property                  | Default     | Backed by UserDefaults key       |
| ------------------------- | ----------- | -------------------------------- |
| `selectedLanguage: String` | `"english"` | `bolo.language`                  |
| `speed: Speed`             | `Speed(1.0)` | `bolo.speed`                    |
| `launchAtLogin: Bool`      | `false`     | `bolo.launchAtLogin`             |
| `hasCompletedOnboarding`   | `false`     | `bolo.hasCompletedOnboarding`    |

Each `didSet` writes to `UserDefaults.standard`. `load()` hydrates
from UserDefaults at app launch. `reset()` is a test-only escape
hatch that clears all keys.

The singleton is `Settings.shared`. Used by `Coordinator`,
`PopoverView`, `SettingsView`, and `OnboardingView`.

### 4.13 `LaunchAtLogin.swift`

A tiny `enum` wrapping `SMAppService.mainApp`:

- `set(enabled:)` — calls `register()` or `unregister()`.
- `isEnabled: Bool` — checks `status == .enabled`.

Invoked from `SettingsView`'s `onChange(of: settings.launchAtLogin)`.

### 4.14 UI layer

**`PopoverController.swift`** — owns an `NSPopover` (`320×320`,
`.transient` behavior, animates). Hosts `PopoverView` via
`NSHostingController`. `show(relativeTo:)` shows below the status
item, `hide()` calls `performClose(nil)`. Also defines
`VisualEffectBackground` — a `NSViewRepresentable` wrapping
`NSVisualEffectView` with `material = .popover, blendingMode =
.behindWindow, state = .active`. This is the native vibrancy
backdrop reused across all popovers, settings, and onboarding.

**`PopoverView.swift`** — the SwiftUI content. Layout: header (icon +
"Bolo" + gear button) → text preview (scrollable, empty-state hint)
→ divider → transport controls (play/pause, stop) → divider →
speed slider (0.5–2.0, step 0.1). Observes `Settings` and the
`CoordinatorState` provided by `AppDelegate`. The `gear` button
calls the `onOpenSettings` closure (which `AppDelegate` wires to its
`openSettings()` method).

`CoordinatorState` lives in this file (alongside `PopoverView`)
because they're tightly coupled. It's `@MainActor`, has
`@Published lastCapturedText` and `@Published isPlaying`, plus
`togglePlayPause` and `stop` closure properties that `Coordinator`
populates in its initializer.

**`SettingsView.swift`** — a `TabView` with two tabs:

- **General**: launch-at-login `Toggle` (wired via
  `LaunchAtLogin.set`), language `Picker` (English only for v1),
  read-only hotkey display (⌘⇧R), and an accessibility status
  `GroupBox` with green check or orange triangle plus an "Open
  System Settings" button when permission isn't granted.
- **About**: app icon + name + version + privacy blurb.

460×320, native Form layout, no custom chrome.

**`OnboardingView.swift`** — a four-step state machine:

```
welcome → accessibility → modelDownload → ready
```

- `welcome`: hero waveform icon, intro copy, Continue button.
- `accessibility`: lock-shield icon, copy explaining why AX is
  needed, "Open System Settings" button. The "I've enabled it"
  button is disabled until `PermissionsManager.isAccessibilityGranted`
  flips true.
- `modelDownload`: ProgressView bound to `ModelDownloadProgress`.
  Calls `onStartDownload()` (which `AppDelegate` wires to
  `modelManager.ensureLoaded()`) on entry. Shows the current
  download label ("Downloading model.safetensors", etc.) under the
  bar. Retry button appears on error.
- `ready`: green checkmark, "Get Started" button → sets
  `Settings.shared.hasCompletedOnboarding = true` and closes the
  window.

Gated by `!Settings.shared.hasCompletedOnboarding` in `AppDelegate`.

---

## 5. Concurrency model

Bolo is built on Swift 6 strict concurrency. The whole codebase compiles
with `SWIFT_STRICT_CONCURRENCY: complete`.

### 5.1 Isolation choices

| Type / surface              | Isolation         | Why                                                        |
| --------------------------- | ----------------- | ---------------------------------------------------------- |
| `AppDelegate`               | `@MainActor`      | AppKit; touches NSStatusItem/NSWindow                      |
| `Coordinator`               | `@MainActor`      | Updates `CoordinatorState`, calls into MainActor PlaybackController |
| `CoordinatorState`          | `@MainActor`      | SwiftUI `ObservableObject`                                 |
| `Settings`                  | `@MainActor`      | SwiftUI `ObservableObject` + UserDefaults (also fine on background, but main is simpler) |
| `HotkeyManager`             | `@MainActor`      | HotKey package callback delivers on main                   |
| `PopoverController`         | `@MainActor`      | NSPopover                                                  |
| `PlaybackController`        | `@MainActor`      | Owns the engine reference and the in-flight `Task`         |
| `ModelDownloadProgress`     | `@MainActor`      | SwiftUI binding                                            |
| `ModelManager<Model>`       | `actor` (`@unchecked Sendable`) | Generic over non-Sendable model; serial executor protects access |
| `Qwen3TTSEngine`            | `actor`           | `Qwen3TTSModel` is not Sendable; concurrent synthesize would corrupt MLX |
| `TextCaptureManager`        | `enum` (static)   | Stateless                                                  |
| `PermissionsManager`        | `enum` (static)   | Stateless                                                  |
| `LaunchAtLogin`             | `enum` (static)   | Stateless                                                  |
| `TTSEngine` protocol        | `: Sendable`      | Allows storing across isolation domains                    |
| `MockTTSEngine`             | `NSObject @unchecked Sendable` | AVSpeechSynthesizer delegate pattern requires NSObject; thread-safety reasoned manually |

### 5.2 Why nonisolated on the engine

`Qwen3TTSEngine.synthesize` is `nonisolated` and delegates to a private
`_synthesize` on the actor. Reason: the `TTSEngine` protocol is `Sendable`
and its requirements are not actor-isolated. To conform an actor to such a
protocol, the conformance methods must be `nonisolated`. The bodies then
hop into the actor's isolation domain via the call to `_synthesize`.

Same pattern is used for `stop()`.

### 5.3 The MainActor.assumeIsolated trick

`HotkeyManager.callback` is `() -> Void` (no actor isolation). The HotKey
package's `keyDownHandler` is `(@Sendable () -> Void)?` — its
implementation invokes on the main thread (Carbon dispatches on main run
loop) but Swift can't verify this from the type signature.

In `Coordinator.start()`:

```swift
hotkey.register { [weak self] in
    MainActor.assumeIsolated { self?.handleHotkey() }
}
```

`MainActor.assumeIsolated` is a runtime check that crashes if you're
actually not on main, but compiles even though the surrounding closure
isn't `@MainActor`. It's the cleanest seam for a known-main external
callback.

### 5.4 Sending closures

`ModelManager.init(idleTimeout:loader:)` takes `loader: sending @escaping
() async throws -> Model`. `sending` is a Swift 6 keyword that says "I
guarantee this closure won't be referenced elsewhere after this call"
— it lets the closure cross from `@MainActor` (where `AppDelegate`
constructs it) into the actor's isolation domain without
`@Sendable` constraint on every captured value.

### 5.5 @preconcurrency imports

Two places use `@preconcurrency import`:

- `Qwen3TTS` in `AppDelegate` and tests — because `Qwen3TTSModel` lacks
  Sendable annotation
- `ApplicationServices` in `PermissionsManager` — because the C global
  `kAXTrustedCheckOptionPrompt` is imported as `var`

These are honest acknowledgments that an external library doesn't yet
declare its concurrency story explicitly. We reason about safety
ourselves and accept compiler-relaxed checking.

---

## 6. Model lifecycle

### 6.1 Memory profile

Bolo is "lightweight" in idle but heavy when active. Approximate RAM:

| State                          | RAM (approx) |
| ------------------------------ | ------------ |
| App just launched, no ⌘⇧R yet | ~30 MB       |
| First ⌘⇧R, model loading       | ~150 MB → ~800 MB peak during load |
| Model loaded, idle              | ~600–800 MB  |
| Model loaded, synthesizing      | ~800 MB to ~1.5 GB peak |
| 5 minutes after last ⌘⇧R       | back to ~30 MB |

The 5-minute idle unload is what makes Bolo "lightweight" in steady
state. A user who reads one paragraph and walks away returns to a
near-zero footprint within 5 minutes.

### 6.2 The lazy-load decision tree

```
ensureLoaded() called
├── model is non-nil ─────► resetIdleTimer(); return model
├── model is nil
│   ├── state := loading
│   ├── await loader()
│   │   ├── cached weights on disk ──► load into memory (~3 sec on M-series)
│   │   └── no cache ────────────────► download from HF (~30 sec @ 100 Mbps; ~500 MB)
│   │                                   ──► load into memory
│   ├── on success: cache model, state := loaded, resetIdleTimer, return
│   └── on throw: state := unloaded, propagate error
```

### 6.3 First-run download

On first ever ⌘⇧R (or during onboarding, whichever comes first), the
`fromPretrained` call triggers a Hugging Face download of:

- **Model**: `aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit/model.safetensors`
  — ~500 MB.
- **Tokenizer**: `Qwen/Qwen3-TTS-Tokenizer-12Hz` — small (~10 MB).

The cache lives at `~/Documents/huggingface/`. After the first download,
all subsequent launches are offline-only.

The `progressHandler` callback fires multiple times per file with
`(progress: Double, label: String)`. We forward these to
`ModelDownloadProgress` on the main actor. The onboarding view's
`ProgressView(value: downloadProgress.progress)` animates smoothly.

### 6.4 Idle unload

`resetIdleTimer()` cancels any pending unload task and schedules a new
one 5 minutes (`300` seconds) in the future. Each ⌘⇧R press resets the
timer.

When the timer fires, `unload()` sets `model = nil`. The Swift runtime
deallocates the underlying `Qwen3TTSModel`, which in turn releases its
MLX arrays and tokenizer. Next press triggers a fresh load — fast
this time (~3 sec) since the safetensors file is already on disk.

---

## 7. Audio pipeline

### 7.1 The chain

```
[Float] samples (24kHz mono)
    │
    ▼
AVAudioPCMBuffer (pcmFormatFloat32, sampleRate=24000, channels=1)
    │
    ▼
AVAudioPlayerNode  ─── scheduleBuffer(completionCallbackType: .dataPlayedBack)
    │
    ▼
AVAudioUnitVarispeed  ─── rate = Float(speed.value)   // 0.5x to 2.0x
    │
    ▼
mainMixerNode (default output device)
    │
    ▼
🔊 system audio
```

### 7.2 Why Varispeed and not the engine's speed param

The Qwen3 `synthesize` method has no speed parameter — it generates
audio at a fixed cadence (the natural speaking rate of the model).
There's no "regenerate the same text faster" without retraining.

`AVAudioUnitVarispeed` does time stretching with pitch correction in
the audio graph. It's not as good as a TTS model that natively
supports rate control (you lose some naturalness at extreme rates),
but for 0.75x–1.5x it's nearly imperceptible.

If a future engine (Sesame, Chatterbox) does support native speed
control, we'd add a method `func nativelySupportsSpeed() -> Bool` to
`TTSEngine` and skip the Varispeed when true.

### 7.3 Buffer scheduling and completion

```swift
await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
    player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
        cont.resume()
    }
}
```

`.dataPlayedBack` fires when the buffer has been physically played
through the output device — not when it's been queued, not when
synthesis is done. So `synthesize` doesn't return until you can stop
hearing audio.

The continuation is `Never`-throwing because completion callbacks
fire even on `stop()` (the player just stops emitting samples; the
callback still fires). This means `stop()` correctly causes the
synthesize call to return.

---

## 8. File system layout (runtime)

Where Bolo writes files on the user's disk:

| Path                                                                  | What                                                  |
| --------------------------------------------------------------------- | ----------------------------------------------------- |
| `~/Library/Preferences/com.virkhanna.bolo.plist`                      | Settings (language, speed, launch-at-login, etc.)     |
| `~/Documents/huggingface/aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit` | Model weights cache (managed by speech-swift)         |
| `~/Documents/huggingface/Qwen/Qwen3-TTS-Tokenizer-12Hz`               | Tokenizer cache                                       |

Nothing else.

To fully uninstall:

```bash
rm -rf ~/Documents/huggingface/aufklarer
rm -rf ~/Documents/huggingface/Qwen
defaults delete com.virkhanna.bolo
rm /Applications/Bolo.app
```

---

## 9. Privacy guarantees

Bolo's privacy posture is enforced by the codebase itself, not just by
policy:

| Guarantee                                  | Enforced by                                                                |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| All synthesis is on-device                 | `Qwen3TTSEngine` only calls `Qwen3TTSModel.synthesize` (local MLX). No network code path. |
| No telemetry, no analytics                 | There's no analytics SDK in the project. `grep` the codebase — no `URLSession.dataTask` outside of speech-swift's internal cache logic. |
| One-time model download is the only network call | `com.apple.security.network.client = true` is required only because speech-swift's `fromPretrained` issues HTTP requests. After the cache is warm, that code path is never entered again. Users can verify with Little Snitch. |
| Settings stay on-device                    | All four `@Published` properties write to `UserDefaults` — local plist on disk. No iCloud sync. |
| No accounts                                | The codebase has no auth code, no Keychain credential storage (other than the optional `bolo-notary` profile, which is developer-side, not user-side). |
| AX permission scope is read-only           | We only call `AXUIElementCopyAttributeValue` for the focused element's `kAXSelectedTextAttribute`. We never write, never enumerate the UI tree, never monitor key events. |

The `com.apple.security.app-sandbox` entitlement is **false**. This is
unavoidable for AX-capable apps — sandboxed apps cannot be trusted by
the AX system. Setapp accepts non-sandboxed apps as long as they're
Hardened Runtime + Developer ID signed + notarized, which we are.

---

## 10. Build, distribution, and the xcodegen choice

The project is **xcodegen-managed**. `project.yml` is the single source
of truth; `Bolo.xcodeproj`, `Bolo/Resources/Info.plist`, and
`Bolo/Bolo.entitlements` are all generated artifacts in `.gitignore`.

Why xcodegen and not a vanilla Xcode project:

- The whole build was bootstrapped by Claude Code subagents working
  headlessly. They can't drive Xcode's "New Project" wizard. xcodegen
  takes a declarative YAML and produces a real `.xcodeproj`.
- Adding SPM dependencies (HotKey, Qwen3Speech) is a single YAML line
  edit, not a series of Xcode UI clicks.
- The `.xcodeproj` is never in conflict with itself — every developer
  regenerates from YAML.
- Entitlements and Info.plist are versioned via `project.yml` `properties`
  blocks. Hand-editing the generated files would silently regress on
  the next `xcodegen generate`.

Cost: developers must run `xcodegen generate` after pulling. The
`README.md`'s build instructions make this explicit.

See [DEVELOPING.md](DEVELOPING.md) for the day-to-day workflow.
See [RELEASE.md](RELEASE.md) for signing + notarization.
See [SETAPP.md](SETAPP.md) for submission metadata.

---

## 11. Extension points

Three places the system was deliberately designed to extend.

### 11.1 Adding a second TTS engine

To add e.g. Sesame or Chatterbox:

1. Add the SPM package to `project.yml` under `packages:` and the
   `Bolo` target's `dependencies:`.
2. Create `Bolo/Engine/SesameTTSEngine.swift` conforming to `TTSEngine`.
3. In `AppDelegate.applicationDidFinishLaunching`, conditionally
   construct either engine (e.g. based on `Settings.shared.preferredEngine`).
4. The protocol's `Sendable` constraint and the `voice: VoiceID, speed:
   Speed` signature already accommodates engines with different voice
   models — `VoiceID.rawValue` is opaque, each engine maps it to its
   own world.

No other code changes needed. `Coordinator`, `PlaybackController`, UI:
all unchanged.

### 11.2 Adding a new language

Today, the language picker in `SettingsView` is hardcoded to
`Text("English").tag("english")`. To add another:

1. Add a new `Text("French").tag("french")` row to the Picker.
2. `VoiceCatalog` doesn't exist; `Qwen3TTSEngine.languageString(for:)`
   already accepts all 10 Qwen3 languages — see the switch statement
   for the supported set.
3. Manual QA: synthesize sample text in the new language, confirm
   voice quality.

### 11.3 Adding a configurable hotkey

Today, the hotkey is hardcoded to ⌘⇧R in `HotkeyManager.register`.
To make it configurable:

1. Add a `KeyCombo` struct to `Settings` (key + modifiers, persistable).
2. Update `HotkeyManager.register` to take a `KeyCombo` instead of
   hardcoding `.r, [.command, .shift]`.
3. Add a `KeyComboRecorder` SwiftUI view to `SettingsView.generalTab`
   that lets the user record a new combo.
4. On Settings change, call `hotkeyManager.unregister()` then
   `hotkeyManager.register(handler:)` again with the new combo.

The `HotKey` SPM package supports arbitrary key/modifier combinations.

---

## 12. Testing strategy

23 tests, organized roughly:

| Test target            | Lines | What it tests                                                   |
| ---------------------- | ----- | --------------------------------------------------------------- |
| `SmokeTests`           | small | One trivial XCTestCase that proves the test bundle loads        |
| `AppDelegateTests`     | small | Status item creation, accessibility description                 |
| `PopoverControllerTests` | small | Popover size, behavior, attach-to-view                         |
| `HotkeyManagerTests`   | small | Register stores callback, unregister clears                     |
| `PermissionsManagerTests` | small | API surface returns Bool                                      |
| `TextCaptureManagerTests` | small | Clipboard read happy path + nil path                          |
| `TTSEngineTests`       | small | VoiceID hashable, Speed clamps, MockEngine empty-text throws, Qwen3TTSEngine init, gated heavy synthesize test |
| `ModelManagerTests`    | small | State transitions: unloaded → loaded, idle timeout, touch resets idle |
| `SettingsTests`        | small | Default values, persistence, reset                              |

What we **don't** test automatically:

- Real Qwen3 synthesis. The heavy integration test is gated on
  `ProcessInfo.environment["BOLO_RUN_HEAVY_TESTS"] == "1"` — when
  set, it loads the model (~500 MB download on first run, ~3 sec
  thereafter) and exercises one synthesize+playback.
- Real AX text capture. The AX path requires Accessibility permission
  + a focused app — neither available in XCTest. Only the clipboard
  fallback is tested.
- The UI. SwiftUI views are not snapshot-tested. The visual surface
  is small and stable; we verify by running the app.

### Why not more UI tests

NSStatusItem and NSPopover are notoriously flaky to UI-test. The
ratio of "tests written" to "bugs caught" doesn't make sense at this
scale. Manual verification + the unit tests on the underlying state
(CoordinatorState, Settings) catch almost everything that matters.

---

## 13. Known compromises and v1.1+ roadmap

What we deliberately deferred:

- **Voice variety**: Qwen3-TTS exposes one voice per language. v1.1
  candidates: integrate Chatterbox (54 named voices, MIT) or add
  Qwen3's voice-cloning capability via uploaded reference audio.
- **Configurable hotkey**: hardcoded ⌘⇧R; the recorder UI is in
  the extension-points section above.
- **Reading queue**: each ⌘⇧R interrupts the current playback.
  v1.1 could queue selections in `Coordinator` and play sequentially.
- **Audio export**: no "save as .m4a" — the audio path is
  ephemeral. The `[Float]` samples flow straight into AVAudioEngine
  and are gone after playback.
- **Pronunciation overrides**: no user dictionary. Qwen3 mispronounces
  the occasional name; v1.1 could add a simple find-replace table
  applied before synthesis.
- **iOS / iPad version**: speech-swift's Qwen3TTS targets iOS too,
  but the AX text-capture story on iOS is entirely different (no
  cross-app selection capture). Would need a Share-extension-based
  alternative input.

---

## 14. Project lineage

Bolo started as a scope/plan iteration in conversation:

- **Scope** locked: [`SCOPE.md`](../SCOPE.md)
- **Plan** (15 tasks, all complete): [`docs/superpowers/plans/2026-05-26-bolo-menubar-tts.md`](superpowers/plans/2026-05-26-bolo-menubar-tts.md)
- **Subvocal**: a prior vibe-coded attempt at the same product space, archived at `~/Code/archive/Subvocal`. Used Kokoro TTS; the voices weren't natural enough. Bolo carries forward zero code from Subvocal — only the design intent.

The path from idea to feature-complete shipped in a single working session
(2026-05-26) via subagent-driven development. The plan is preserved as-is
for reference, including the "engine reality update" section that documents
the mid-execution pivot from a planned voice picker to language selection
once the real Qwen3 API was discovered.
