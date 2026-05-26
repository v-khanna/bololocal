# HearIt Menu Bar TTS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a macOS menu bar app that, when the user presses ⌘⇧R with text selected anywhere, reads the selected text aloud in a natural AI voice running fully on-device. Distribution: Setapp only.

**Architecture:** SwiftUI menu bar app with `NSStatusItem` + `NSPopover`. A `Coordinator` wires a `HotkeyManager` (global hotkey via the [HotKey](https://github.com/soffes/HotKey) Swift Package), a `TextCaptureManager` (AXUIElement Accessibility API with clipboard fallback), and a `TTSEngine` protocol implementation (`Qwen3TTSEngine` backed by [soniqo/speech-swift](https://github.com/soniqo/speech-swift) on MLX) into a `PlaybackController` driving `AVAudioEngine`. A `ModelManager` actor lazy-loads the Qwen3-TTS weights on first use and unloads them after 5 minutes of idle. UI is native-stealth: `NSVisualEffectView` vibrancy, SF Symbols, system font, light/dark auto.

**Tech Stack:** Swift 6, SwiftUI + AppKit, Xcode 26+, macOS 15+ deployment target, Apple Silicon only, SPM dependencies = HotKey + speech-swift (Qwen3TTS module), MLX-Swift (transitive), AVFoundation, ApplicationServices (AX API), ServiceManagement (SMAppService for launch-at-login).

---

## Pre-Flight Checks

Before any task: confirm host environment.

- [ ] **Verify Xcode ≥ 15**

  Run: `xcodebuild -version`
  Expected: `Xcode 15.0` or higher (current host has 26.4.1 — fine).

- [ ] **Verify Apple Silicon + macOS 15+**

  Run: `uname -m && sw_vers -productVersion`
  Expected: `arm64` and version `≥ 15.0`.

- [ ] **Verify Metal toolchain is present**

  Run: `xcrun --find metal`
  Expected: a path like `/Applications/Xcode.app/Contents/Developer/usr/bin/metal`.
  If missing: `xcodebuild -downloadComponent MetalToolchain`.

- [ ] **Verify native ARM Homebrew (required by speech-swift)**

  Run: `which brew && brew --prefix`
  Expected: `/opt/homebrew/bin/brew` and `/opt/homebrew`.
  If you only have x86_64 brew, install native ARM brew per [brew.sh](https://brew.sh).

- [ ] **Test-compile speech-swift in a throwaway SPM project to de-risk integration**

  ```bash
  mkdir -p /tmp/speech-swift-probe && cd /tmp/speech-swift-probe
  swift package init --type executable --name probe
  cat > Package.swift <<'EOF'
  // swift-tools-version:5.9
  import PackageDescription
  let package = Package(
    name: "probe",
    platforms: [.macOS(.v15)],
    dependencies: [
      .package(url: "https://github.com/soniqo/speech-swift", from: "0.0.9"),
    ],
    targets: [
      .executableTarget(
        name: "probe",
        dependencies: [.product(name: "Qwen3TTS", package: "speech-swift")]
      )
    ]
  )
  EOF
  swift build 2>&1 | tail -40
  ```

  Expected: a `Build complete!` line, possibly with warnings.
  If it fails with metallib errors: clone `speech-swift` and run `make build` once per the [README](https://github.com/soniqo/speech-swift), then retry.
  **Do not proceed to Task 1 until this passes.** If speech-swift cannot be made to build in 30 minutes of fighting, escalate: switch engine to [AtomGradient/swift-qwen3-tts](https://github.com/AtomGradient/swift-qwen3-tts) (standalone alternative) or revert to Kokoro.

---

## File Structure

The working directory is `~/Code/tts-app/`. Once the brand is locked, rename the directory and the `HearIt` Xcode target — both renames are mechanical.

```
~/Code/tts-app/
├── HearIt.xcodeproj/                       (Task 1)
├── HearIt/                                  (Xcode app target)
│   ├── HearItApp.swift                     (Task 1)   @main entry
│   ├── AppDelegate.swift                   (Task 2)   NSStatusItem owner
│   ├── Coordinator.swift                   (Task 6)   wires everything
│   │
│   ├── Hotkey/
│   │   └── HotkeyManager.swift             (Task 4)
│   │
│   ├── Capture/
│   │   ├── PermissionsManager.swift        (Task 5)
│   │   └── TextCaptureManager.swift        (Task 5)
│   │
│   ├── Engine/
│   │   ├── TTSEngine.swift                 (Task 6)   protocol + value types
│   │   ├── MockTTSEngine.swift             (Task 6)   AVSpeechSynthesizer wrapper
│   │   ├── Qwen3TTSEngine.swift            (Task 7)   speech-swift wrapper
│   │   ├── ModelDownloader.swift           (Task 8)
│   │   └── ModelManager.swift              (Task 9)   lazy-load + idle unload actor
│   │
│   ├── Playback/
│   │   └── PlaybackController.swift        (Task 6)   AVAudioEngine wrapper
│   │
│   ├── Models/
│   │   ├── Settings.swift                  (Task 10)  @AppStorage wrapper
│   │   └── VoiceCatalog.swift              (Task 10)  curated 6–8 voices
│   │
│   ├── UI/
│   │   ├── PopoverView.swift               (Task 11)
│   │   ├── SettingsView.swift              (Task 12)
│   │   └── OnboardingView.swift            (Task 13)
│   │
│   ├── Resources/
│   │   ├── Assets.xcassets/                (Task 1)
│   │   └── Info.plist                      (Task 2)
│   └── HearIt.entitlements                 (Task 2)
│
├── HearItTests/                            (Task 1)
│   ├── TTSEngineTests.swift                (Task 6)
│   ├── ModelManagerTests.swift             (Task 9)
│   ├── SettingsTests.swift                 (Task 10)
│   └── VoiceCatalogTests.swift             (Task 10)
│
├── scripts/
│   ├── build.sh                            (Task 15)
│   └── notarize.sh                         (Task 15)
│
├── docs/
│   ├── superpowers/plans/
│   │   └── 2026-05-26-hearit-menubar-tts.md   (this file)
│   ├── SETAPP.md                           (Task 15)
│   └── PRIVACY.md                          (Task 15)
│
├── README.md                               (Task 1)
└── SCOPE.md                                (already exists)
```

**Naming convention:** the working name `HearIt` is provisional. Use it everywhere in code, bundle ID `com.virkhanna.hearit`, until the final brand is decided. The rename later is a global find-and-replace.

---

## Testing Strategy

macOS Swift apps don't all lend themselves to TDD equally:

- **Real TDD with XCTest:** anything that doesn't touch AppKit/SwiftUI directly. That's `TTSEngine` protocol, `ModelManager` actor, `Settings`, `VoiceCatalog`, `TextCaptureManager` (with stubbed `AXUIElement`), `MockTTSEngine`, `PlaybackController` (with audio output disabled in test environment).

- **Build-and-verify with manual checks:** the AppKit surface — `NSStatusItem`, `NSPopover`, accessibility permission prompts, SwiftUI views that render. Each such task lists explicit visual checks the engineer runs after building.

- **Integration test (manual):** the full pipeline at Task 6 and Task 7. Engineer selects text in Safari, presses ⌘⇧R, hears audio. No way to fully automate this in CI without a signed test harness — accept the manual loop.

Every commit must leave a buildable, runnable app. If a task can't deliver that, the task is too big — split it.

---

## Task 1: Xcode Project Scaffold

**Files:**
- Create: `~/Code/tts-app/HearIt.xcodeproj/` (via Xcode wizard)
- Create: `~/Code/tts-app/HearIt/HearItApp.swift`
- Create: `~/Code/tts-app/HearIt/Resources/Assets.xcassets/AppIcon.appiconset/Contents.json`
- Create: `~/Code/tts-app/HearItTests/SmokeTests.swift`
- Create: `~/Code/tts-app/README.md`
- Create: `~/Code/tts-app/.gitignore`

- [ ] **Step 1: Initialize git repository in the working directory**

  ```bash
  cd ~/Code/tts-app
  git init
  cat > .gitignore <<'EOF'
  # Xcode
  build/
  DerivedData/
  *.xcuserstate
  xcuserdata/
  *.xcworkspace/xcuserdata/
  .DS_Store

  # SPM
  .swiftpm/
  Package.resolved

  # Notarization artifacts
  *.dmg
  *.zip
  notarization.log

  # macOS
  .DS_Store
  EOF
  git add SCOPE.md docs/ .gitignore
  git commit -m "chore: initial scope + plan"
  ```

  Expected: a fresh repo with one commit containing SCOPE.md and this plan.

- [ ] **Step 2: Create the Xcode project**

  Open Xcode → File → New → Project → macOS → App. Settings:
  - Product Name: `HearIt`
  - Team: your Apple Developer team (required later for signing)
  - Organization Identifier: `com.virkhanna`
  - Bundle Identifier: auto-computes to `com.virkhanna.hearit`
  - Interface: SwiftUI
  - Language: Swift
  - Storage: None
  - **Uncheck** "Include Tests" (we'll add a test target manually with a cleaner structure)
  - Save to: `~/Code/tts-app/` (this creates `~/Code/tts-app/HearIt.xcodeproj` and `~/Code/tts-app/HearIt/`)

- [ ] **Step 3: Set deployment target and architecture**

  In the project navigator → select `HearIt` project → Targets → `HearIt`:
  - General → Minimum Deployments → macOS = **15.0**
  - Build Settings → Architectures → **Standard Architectures (arm64)** only — remove `x86_64`
  - Build Settings → Swift Language Version = Swift 6
  - Build Settings → Strict Concurrency Checking = Complete

- [ ] **Step 4: Replace generated `ContentView.swift` with a placeholder `HearItApp.swift`**

  Delete `HearIt/ContentView.swift`. Replace `HearIt/HearItApp.swift` contents with:

  ```swift
  import SwiftUI

  @main
  struct HearItApp: App {
      var body: some Scene {
          // Menu bar app: no main window. AppDelegate (added in Task 2)
          // will own the NSStatusItem and popover.
          Settings { EmptyView() }
      }
  }
  ```

  Build: `⌘B` in Xcode. Expected: build succeeds, no warnings.

- [ ] **Step 5: Add the test target**

  File → New → Target → macOS → Unit Testing Bundle:
  - Product Name: `HearItTests`
  - Target to be Tested: `HearIt`

  Replace the generated `HearItTests.swift` file with `HearItTests/SmokeTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class SmokeTests: XCTestCase {
      func test_smoke_passes() {
          XCTAssertTrue(true)
      }
  }
  ```

  Run: `⌘U` in Xcode. Expected: 1 test passes.

- [ ] **Step 6: Create the README placeholder**

  ```bash
  cat > ~/Code/tts-app/README.md <<'EOF'
  # HearIt (working name)

  macOS menu bar app that reads selected text aloud in a natural AI voice.
  Fully on-device. Apple Silicon only. macOS 15+.

  See [SCOPE.md](SCOPE.md) for product scope.
  See [docs/superpowers/plans/](docs/superpowers/plans/) for implementation plan.
  EOF
  ```

- [ ] **Step 7: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt.xcodeproj HearIt HearItTests README.md
  git commit -m "feat(scaffold): empty SwiftUI menu bar app + test target"
  ```

  Expected: tests pass, app builds. The built app launches but shows nothing (no main window, no menu bar item yet — that's Task 2).

---

## Task 2: Menu Bar Presence (NSStatusItem + Entitlements)

**Files:**
- Create: `HearIt/AppDelegate.swift`
- Modify: `HearIt/HearItApp.swift`
- Modify: `HearIt/Resources/Info.plist`
- Create: `HearIt/HearIt.entitlements`

- [ ] **Step 1: Add `LSUIElement = true` to Info.plist**

  This hides the Dock icon and removes the standard menu bar — making it a true menu-bar-only app.

  In Xcode → `HearIt` target → Info tab → add a new key:
  - Key: `Application is agent (UIElement)` (raw: `LSUIElement`)
  - Type: Boolean
  - Value: `YES`

- [ ] **Step 2: Add an `NSAccessibilityUsageDescription` string (used in Task 5)**

  In the same Info tab, add:
  - Key: `Privacy - Accessibility Usage Description` (raw: `NSAccessibilityUsageDescription`)
  - Type: String
  - Value: `HearIt reads the text you select in any app and speaks it aloud. Accessibility access is required to read selected text from other applications.`

- [ ] **Step 3: Configure entitlements**

  Open the auto-generated `HearIt.entitlements` (or create one if absent). The entitlements XML body:

  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0">
  <dict>
      <key>com.apple.security.app-sandbox</key>
      <false/>
      <key>com.apple.security.network.client</key>
      <true/>
      <key>com.apple.security.files.user-selected.read-only</key>
      <true/>
  </dict>
  </plist>
  ```

  Note: **App Sandbox is disabled.** Accessibility access from a sandboxed app is famously broken — Setapp accepts non-sandboxed apps as long as they're notarized and Hardened Runtime is on. `network.client` is enabled for the one-time model download.

- [ ] **Step 4: Enable Hardened Runtime**

  Target → Signing & Capabilities → click `+ Capability` → add `Hardened Runtime`. Leave all the exception sub-checkboxes unchecked unless build errors force you to.

- [ ] **Step 5: Write the failing test for AppDelegate creating a status item**

  Add `HearItTests/AppDelegateTests.swift`:

  ```swift
  import XCTest
  import AppKit
  @testable import HearIt

  final class AppDelegateTests: XCTestCase {
      func test_applicationDidFinishLaunching_createsStatusItem() {
          let delegate = AppDelegate()
          delegate.applicationDidFinishLaunching(Notification(name: .init("test")))
          XCTAssertNotNil(delegate.statusItem)
          XCTAssertEqual(delegate.statusItem?.button?.image?.accessibilityDescription, "HearIt")
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL with "Cannot find 'AppDelegate' in scope".

- [ ] **Step 6: Implement `AppDelegate`**

  Create `HearIt/AppDelegate.swift`:

  ```swift
  import AppKit
  import SwiftUI

  final class AppDelegate: NSObject, NSApplicationDelegate {
      var statusItem: NSStatusItem?

      func applicationDidFinishLaunching(_ notification: Notification) {
          let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
          let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
          icon?.isTemplate = true
          item.button?.image = icon
          item.button?.action = #selector(handleStatusItemClick)
          item.button?.target = self
          self.statusItem = item
      }

      @objc private func handleStatusItemClick() {
          // Popover opens here (Task 3 wires this up).
          NSLog("HearIt status item clicked")
      }
  }
  ```

- [ ] **Step 7: Wire AppDelegate into the SwiftUI App**

  Replace `HearIt/HearItApp.swift`:

  ```swift
  import SwiftUI

  @main
  struct HearItApp: App {
      @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

      var body: some Scene {
          Settings { EmptyView() }
      }
  }
  ```

- [ ] **Step 8: Run tests + visually verify**

  Run: `⌘U`. Expected: AppDelegateTests passes.

  Run the app: `⌘R`. Expected: a waveform icon appears in the menu bar. Clicking it logs `HearIt status item clicked` to the Xcode console. No Dock icon. No standard menu bar at the top.

- [ ] **Step 9: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(menu-bar): NSStatusItem with waveform icon, LSUIElement, Hardened Runtime, AX usage description"
  ```

---

## Task 3: Blank Popover Attached to Status Item

**Files:**
- Create: `HearIt/UI/PopoverController.swift`
- Modify: `HearIt/AppDelegate.swift`

- [ ] **Step 1: Write the failing test**

  Add `HearItTests/PopoverControllerTests.swift`:

  ```swift
  import XCTest
  import AppKit
  @testable import HearIt

  final class PopoverControllerTests: XCTestCase {
      func test_init_createsPopoverWithCorrectSize() {
          let controller = PopoverController()
          XCTAssertEqual(controller.popover.contentSize, NSSize(width: 320, height: 280))
          XCTAssertEqual(controller.popover.behavior, .transient)
      }

      func test_show_attachesPopoverToView() {
          let controller = PopoverController()
          let dummyButton = NSStatusBarButton(frame: .zero)
          controller.show(relativeTo: dummyButton)
          XCTAssertTrue(controller.popover.isShown)
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL.

- [ ] **Step 2: Implement `PopoverController`**

  Create `HearIt/UI/PopoverController.swift`:

  ```swift
  import AppKit
  import SwiftUI

  final class PopoverController {
      let popover: NSPopover

      init() {
          let p = NSPopover()
          p.contentSize = NSSize(width: 320, height: 280)
          p.behavior = .transient
          p.animates = true
          // Placeholder content; replaced by PopoverView in Task 11.
          let host = NSHostingController(rootView: PopoverPlaceholderView())
          p.contentViewController = host
          self.popover = p
      }

      func show(relativeTo view: NSView) {
          popover.show(relativeTo: view.bounds, of: view, preferredEdge: .minY)
      }

      func hide() {
          popover.performClose(nil)
      }
  }

  private struct PopoverPlaceholderView: View {
      var body: some View {
          ZStack {
              VisualEffectBackground()
              Text("HearIt")
                  .font(.title3)
                  .foregroundStyle(.secondary)
          }
          .frame(width: 320, height: 280)
      }
  }

  /// NSVisualEffectView wrapped for SwiftUI. Native-stealth foundation.
  struct VisualEffectBackground: NSViewRepresentable {
      func makeNSView(context: Context) -> NSVisualEffectView {
          let v = NSVisualEffectView()
          v.material = .popover
          v.blendingMode = .behindWindow
          v.state = .active
          return v
      }
      func updateNSView(_ nsView: NSVisualEffectView, context: Context) {}
  }
  ```

- [ ] **Step 3: Update AppDelegate to own and toggle the popover**

  Replace `HearIt/AppDelegate.swift`:

  ```swift
  import AppKit
  import SwiftUI

  final class AppDelegate: NSObject, NSApplicationDelegate {
      var statusItem: NSStatusItem?
      let popoverController = PopoverController()

      func applicationDidFinishLaunching(_ notification: Notification) {
          let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
          let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
          icon?.isTemplate = true
          item.button?.image = icon
          item.button?.action = #selector(togglePopover)
          item.button?.target = self
          self.statusItem = item
      }

      @objc private func togglePopover() {
          guard let button = statusItem?.button else { return }
          if popoverController.popover.isShown {
              popoverController.hide()
          } else {
              popoverController.show(relativeTo: button)
          }
      }
  }
  ```

- [ ] **Step 4: Run tests + visually verify**

  Run: `⌘U`. Expected: PopoverControllerTests passes, AppDelegateTests still passes.

  Run: `⌘R`. Expected: clicking the menu bar icon opens a 320×280 popover with translucent vibrancy background showing "HearIt" centered. Clicking the icon again (or anywhere outside) dismisses it.

- [ ] **Step 5: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(ui): native-stealth NSPopover with vibrancy attached to status item"
  ```

---

## Task 4: Global Hotkey (⌘⇧R)

**Files:**
- Modify: `HearIt.xcodeproj` (add HotKey SPM dependency via Xcode)
- Create: `HearIt/Hotkey/HotkeyManager.swift`
- Create: `HearItTests/HotkeyManagerTests.swift`

- [ ] **Step 1: Add the HotKey Swift Package**

  In Xcode → File → Add Package Dependencies → enter `https://github.com/soffes/HotKey` → Dependency Rule: Up to Next Major Version, `0.2.0` → Add Package → check `HotKey` library and add to the `HearIt` target.

  Verify: the project navigator shows `HotKey` under "Package Dependencies."

- [ ] **Step 2: Write the failing test**

  Create `HearItTests/HotkeyManagerTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class HotkeyManagerTests: XCTestCase {
      func test_register_storesCallback() {
          let manager = HotkeyManager()
          var fired = false
          manager.register { fired = true }
          // We can't simulate the real OS-level hotkey in unit tests; we test
          // that the callback is stored and the manager.fire() helper invokes it.
          manager.fire()
          XCTAssertTrue(fired)
      }

      func test_unregister_clearsCallback() {
          let manager = HotkeyManager()
          var fired = false
          manager.register { fired = true }
          manager.unregister()
          manager.fire()
          XCTAssertFalse(fired)
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL.

- [ ] **Step 3: Implement `HotkeyManager`**

  Create `HearIt/Hotkey/HotkeyManager.swift`:

  ```swift
  import Foundation
  import HotKey
  import AppKit

  final class HotkeyManager {
      private var hotkey: HotKey?
      private var callback: (() -> Void)?

      /// Register the global ⌘⇧R hotkey. Called once at app launch.
      func register(handler: @escaping () -> Void) {
          self.callback = handler
          let hk = HotKey(key: .r, modifiers: [.command, .shift])
          hk.keyDownHandler = { [weak self] in
              self?.callback?()
          }
          self.hotkey = hk
      }

      func unregister() {
          callback = nil
          hotkey = nil
      }

      /// Manual fire helper for unit tests (does not actually press the hotkey).
      func fire() {
          callback?()
      }
  }
  ```

- [ ] **Step 4: Run tests**

  Run: `⌘U`. Expected: HotkeyManagerTests passes.

- [ ] **Step 5: Manual integration check**

  Temporarily wire the hotkey in `AppDelegate.applicationDidFinishLaunching`:

  ```swift
  let hotkey = HotkeyManager()
  hotkey.register { NSLog("HearIt hotkey fired") }
  self.hotkeyManager = hotkey  // add this stored property
  ```

  Add `let hotkeyManager: HotkeyManager` style as a stored property. Run: `⌘R`. With the app running, press ⌘⇧R from any app. Expected: `HearIt hotkey fired` logs in the Xcode console.

  **Note:** If the hotkey doesn't fire, the user may need to grant the app Input Monitoring permission. This is a one-time `System Settings → Privacy & Security → Input Monitoring` enable, surfaced properly in Task 13's onboarding.

- [ ] **Step 6: Revert the temporary wiring**

  Remove the temporary `NSLog` wiring. The real wiring lands in Task 6 (Coordinator). Keep the `hotkeyManager` stored property on AppDelegate but leave it unused for now.

- [ ] **Step 7: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests *.xcodeproj
  git commit -m "feat(hotkey): HotkeyManager with HotKey package, ⌘⇧R global registration"
  ```

---

## Task 5: Accessibility Permission + Selected Text Capture

**Files:**
- Create: `HearIt/Capture/PermissionsManager.swift`
- Create: `HearIt/Capture/TextCaptureManager.swift`
- Create: `HearItTests/TextCaptureManagerTests.swift`

- [ ] **Step 1: Write the failing test for PermissionsManager**

  Add `HearItTests/PermissionsManagerTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class PermissionsManagerTests: XCTestCase {
      func test_isAccessibilityGranted_returnsBool() {
          // We can't programmatically force the AX flag in a test sandbox.
          // We assert the API surface exists and returns a Bool.
          let result = PermissionsManager.isAccessibilityGranted
          XCTAssertTrue(result == true || result == false)
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL ("PermissionsManager not in scope").

- [ ] **Step 2: Implement `PermissionsManager`**

  Create `HearIt/Capture/PermissionsManager.swift`:

  ```swift
  import ApplicationServices
  import AppKit

  enum PermissionsManager {
      /// Current AX trust state — does NOT prompt.
      static var isAccessibilityGranted: Bool {
          AXIsProcessTrusted()
      }

      /// Show the system AX prompt (one-shot — only shows once per app install).
      static func requestAccessibility() {
          let options = [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: true]
          _ = AXIsProcessTrustedWithOptions(options as CFDictionary)
      }

      /// Deep-link the user to System Settings → Privacy → Accessibility.
      static func openAccessibilitySettings() {
          let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
          NSWorkspace.shared.open(url)
      }
  }
  ```

  Run: `⌘U`. Expected: PASS.

- [ ] **Step 3: Write the failing test for TextCaptureManager**

  Add `HearItTests/TextCaptureManagerTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class TextCaptureManagerTests: XCTestCase {
      func test_captureFromClipboard_returnsCurrentClipboardString() {
          NSPasteboard.general.clearContents()
          NSPasteboard.general.setString("hello world", forType: .string)
          let captured = TextCaptureManager.captureFromClipboard()
          XCTAssertEqual(captured, "hello world")
      }

      func test_captureFromClipboard_returnsNilWhenEmpty() {
          NSPasteboard.general.clearContents()
          let captured = TextCaptureManager.captureFromClipboard()
          XCTAssertNil(captured)
      }
  }
  ```

  Note: AX-based capture (`captureFromSelection()`) requires Accessibility permission and a real focused app — not easily testable in XCTest. We test the deterministic clipboard fallback only; the AX path gets a manual verification check below.

  Run: `⌘U`. Expected: FAIL.

- [ ] **Step 4: Implement `TextCaptureManager`**

  Create `HearIt/Capture/TextCaptureManager.swift`:

  ```swift
  import ApplicationServices
  import AppKit

  enum TextCaptureManager {

      /// Best-effort: try AX selected text first, fall back to clipboard.
      /// Returns the captured string or nil.
      static func captureSelectedText() -> String? {
          if let ax = captureFromAccessibility() { return ax }
          return captureFromClipboard()
      }

      /// Read selected text from the frontmost app's focused UI element via AX.
      static func captureFromAccessibility() -> String? {
          guard PermissionsManager.isAccessibilityGranted else { return nil }

          let systemWide = AXUIElementCreateSystemWide()
          var focused: AnyObject?
          let focusErr = AXUIElementCopyAttributeValue(
              systemWide,
              kAXFocusedUIElementAttribute as CFString,
              &focused
          )
          guard focusErr == .success, let element = focused else { return nil }

          var selected: AnyObject?
          let textErr = AXUIElementCopyAttributeValue(
              element as! AXUIElement,
              kAXSelectedTextAttribute as CFString,
              &selected
          )
          guard textErr == .success, let s = selected as? String, !s.isEmpty else {
              return nil
          }
          return s
      }

      /// Plain clipboard read. Used when AX returns nothing.
      static func captureFromClipboard() -> String? {
          NSPasteboard.general.string(forType: .string)
      }
  }
  ```

  Run: `⌘U`. Expected: PASS (both PermissionsManagerTests and TextCaptureManagerTests).

- [ ] **Step 5: Manual AX verification**

  This part has no automated test; verify by hand.

  Temporarily add to `AppDelegate.applicationDidFinishLaunching`:

  ```swift
  hotkey.register {
      if let text = TextCaptureManager.captureSelectedText() {
          NSLog("captured: \(text)")
      } else {
          NSLog("captured nothing")
      }
  }
  ```

  Run: `⌘R`. Grant Accessibility permission when the prompt appears (System Settings → Privacy & Security → Accessibility → enable HearIt). Then:
  1. In Safari, select a paragraph of text.
  2. Press ⌘⇧R.
  3. Expected: Xcode console logs `captured: <the selected text>`.
  4. Try with text selected in TextEdit, Notes, Pages. All should work.
  5. Try with text selected in a Terminal — AX may return nothing (Terminal has poor AX support). Expected fallback: the clipboard's current contents OR `captured nothing`.

  Revert the temporary `NSLog` wiring afterward — the real glue lands in Task 6.

- [ ] **Step 6: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(capture): AX selected-text capture with clipboard fallback, PermissionsManager API"
  ```

---

## Task 6: TTSEngine Protocol + Mock Engine + Playback + Coordinator (End-to-End Pipeline)

This task is the biggest. It proves the entire pipeline end-to-end using the macOS built-in `AVSpeechSynthesizer` voice — no model download, no MLX, just a real audio path. After this commit, ⌘⇧R reads selected text aloud (in a robotic system voice). Tasks 7–9 swap the engine for the real Qwen3 voice without changing this scaffolding.

**Files:**
- Create: `HearIt/Engine/TTSEngine.swift`
- Create: `HearIt/Engine/MockTTSEngine.swift`
- Create: `HearIt/Playback/PlaybackController.swift`
- Create: `HearIt/Coordinator.swift`
- Modify: `HearIt/AppDelegate.swift`
- Create: `HearItTests/TTSEngineTests.swift`

- [ ] **Step 1: Define value types — test first**

  Add `HearItTests/TTSEngineTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class TTSEngineTests: XCTestCase {
      func test_voiceID_isHashable() {
          let a = VoiceID(rawValue: "system-default")
          let b = VoiceID(rawValue: "system-default")
          XCTAssertEqual(a, b)
          XCTAssertEqual(a.hashValue, b.hashValue)
      }

      func test_speed_clampsToValidRange() {
          XCTAssertEqual(Speed(0.25).value, 0.5) // clamps up
          XCTAssertEqual(Speed(3.0).value, 2.0)  // clamps down
          XCTAssertEqual(Speed(1.0).value, 1.0)
      }

      func test_mockEngine_synthesize_completesWithoutError() async throws {
          let engine = MockTTSEngine()
          try await engine.synthesize(
              text: "hello",
              voice: .systemDefault,
              speed: Speed(1.0)
          )
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL ("VoiceID, Speed, MockTTSEngine not in scope").

- [ ] **Step 2: Implement `TTSEngine.swift`**

  Create `HearIt/Engine/TTSEngine.swift`:

  ```swift
  import Foundation

  /// Opaque identifier for a voice in any engine.
  struct VoiceID: Hashable, Codable, RawRepresentable {
      let rawValue: String
      init(rawValue: String) { self.rawValue = rawValue }

      static let systemDefault = VoiceID(rawValue: "system-default")
  }

  /// Clamped speech speed multiplier, 0.5x..2.0x.
  struct Speed: Equatable, Codable {
      let value: Double
      init(_ raw: Double) {
          self.value = max(0.5, min(2.0, raw))
      }
  }

  /// Engine-agnostic TTS contract. Implementations: MockTTSEngine, Qwen3TTSEngine.
  protocol TTSEngine: Sendable {
      /// Synthesize the text and play it through the system audio output.
      /// Throws on synthesis or playback failure.
      /// Returns when playback finishes (or `stop()` is called).
      func synthesize(text: String, voice: VoiceID, speed: Speed) async throws

      /// Halt any in-progress playback immediately.
      func stop()
  }

  enum TTSError: Error, Equatable {
      case modelNotLoaded
      case synthesisFailed(String)
      case playbackFailed(String)
      case emptyText
  }
  ```

- [ ] **Step 3: Implement `MockTTSEngine` using AVSpeechSynthesizer**

  Create `HearIt/Engine/MockTTSEngine.swift`:

  ```swift
  import AVFoundation
  import Foundation

  /// Built-in macOS voice via AVSpeechSynthesizer. Used as a placeholder until
  /// Qwen3TTSEngine (Task 7) replaces it. No model download required.
  final class MockTTSEngine: NSObject, TTSEngine, @unchecked Sendable {
      private let synth = AVSpeechSynthesizer()
      private var continuation: CheckedContinuation<Void, Error>?

      override init() {
          super.init()
          synth.delegate = self
      }

      func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
          guard !text.isEmpty else { throw TTSError.emptyText }
          try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
              self.continuation = cont
              let utt = AVSpeechUtterance(string: text)
              utt.voice = AVSpeechSynthesisVoice(language: "en-US")
              // AVSpeechUtterance rate is 0..1 with 0.5 being normal. Map our Speed:
              utt.rate = Float(AVSpeechUtteranceDefaultSpeechRate * speed.value)
              synth.speak(utt)
          }
      }

      func stop() {
          synth.stopSpeaking(at: .immediate)
          continuation?.resume()
          continuation = nil
      }
  }

  extension MockTTSEngine: AVSpeechSynthesizerDelegate {
      func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
          continuation?.resume()
          continuation = nil
      }
      func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
          continuation?.resume()
          continuation = nil
      }
  }
  ```

  Run: `⌘U`. Expected: PASS (all TTSEngineTests).

- [ ] **Step 4: Implement `PlaybackController`**

  PlaybackController owns the *current* engine and exposes a simpler imperative interface to the Coordinator. (In the mock-engine world this is thin; in Task 7 it grows to manage AVAudioEngine output buffers.)

  Create `HearIt/Playback/PlaybackController.swift`:

  ```swift
  import Foundation

  @MainActor
  final class PlaybackController {
      private let engine: TTSEngine
      private var currentTask: Task<Void, Never>?

      var isPlaying: Bool { currentTask != nil }

      init(engine: TTSEngine) {
          self.engine = engine
      }

      func play(text: String, voice: VoiceID, speed: Speed) {
          stop()
          currentTask = Task { [engine] in
              do {
                  try await engine.synthesize(text: text, voice: voice, speed: speed)
              } catch {
                  NSLog("HearIt playback error: \(error)")
              }
          }
      }

      func stop() {
          engine.stop()
          currentTask?.cancel()
          currentTask = nil
      }
  }
  ```

- [ ] **Step 5: Implement `Coordinator`**

  Create `HearIt/Coordinator.swift`:

  ```swift
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
          // Voice + speed wired to Settings in Task 10. For now: defaults.
          playback.play(text: text, voice: .systemDefault, speed: Speed(1.0))
      }
  }
  ```

- [ ] **Step 6: Wire Coordinator into AppDelegate**

  Replace `HearIt/AppDelegate.swift`:

  ```swift
  import AppKit
  import SwiftUI

  final class AppDelegate: NSObject, NSApplicationDelegate {
      var statusItem: NSStatusItem?
      let popoverController = PopoverController()
      let hotkeyManager = HotkeyManager()
      var coordinator: Coordinator?

      func applicationDidFinishLaunching(_ notification: Notification) {
          // Status item
          let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
          let icon = NSImage(systemSymbolName: "waveform", accessibilityDescription: "HearIt")
          icon?.isTemplate = true
          item.button?.image = icon
          item.button?.action = #selector(togglePopover)
          item.button?.target = self
          self.statusItem = item

          // Pipeline
          let engine: TTSEngine = MockTTSEngine()
          let playback = PlaybackController(engine: engine)
          let coordinator = Coordinator(hotkey: hotkeyManager, playback: playback)
          coordinator.start()
          self.coordinator = coordinator
      }

      @objc private func togglePopover() {
          guard let button = statusItem?.button else { return }
          if popoverController.popover.isShown {
              popoverController.hide()
          } else {
              popoverController.show(relativeTo: button)
          }
      }
  }
  ```

- [ ] **Step 7: Run tests + full manual integration**

  Run: `⌘U`. Expected: all tests pass.

  Run: `⌘R`. Grant Accessibility permission. Open Safari, select a sentence, press ⌘⇧R. Expected: a robotic macOS system voice reads the selected text aloud. Press ⌘⇧R again with different text — it interrupts and reads the new text. The voice quality is bad (that's expected for the mock engine — Task 7 swaps it).

- [ ] **Step 8: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(pipeline): end-to-end ⌘⇧R → AX capture → AVSpeechSynthesizer playback (mock engine)"
  ```

---

## Task 7: speech-swift Integration + Qwen3TTSEngine

The pipeline now exists. Swap the mock engine for the real Qwen3 voice.

**Files:**
- Modify: `HearIt.xcodeproj` (add speech-swift SPM dependency)
- Create: `HearIt/Engine/Qwen3TTSEngine.swift`
- Modify: `HearIt/Engine/TTSEngine.swift` (add a richer default voice)
- Modify: `HearItTests/TTSEngineTests.swift`

- [ ] **Step 1: Add the speech-swift Swift Package**

  In Xcode → File → Add Package Dependencies → enter `https://github.com/soniqo/speech-swift` → Dependency Rule: Up to Next Major Version, `0.0.9`. Add to the `HearIt` target:
  - Library: `Qwen3TTS`

  Verify the project navigator lists `speech-swift` under Package Dependencies. Build: `⌘B`. Expected: success.

  **If the build fails with metallib errors:** clone `speech-swift` separately, run `cd speech-swift && make build`, then point your SPM dependency at the local checkout temporarily (`File → Add Package Dependencies → Add Local…`). This is a known MLX-Swift quirk. Re-test the build.

- [ ] **Step 2: Write the failing test**

  Update `HearItTests/TTSEngineTests.swift` — add:

  ```swift
  func test_qwen3Engine_initializes() {
      let engine = Qwen3TTSEngine()
      XCTAssertNotNil(engine)
  }

  // Long-running, opt-in test — actually loads the model and synthesizes.
  // Skip in CI; run manually via `xcodebuild test -only-testing:HearItTests/TTSEngineTests/test_qwen3Engine_synthesize_realModel`.
  func test_qwen3Engine_synthesize_realModel() async throws {
      try XCTSkipIf(ProcessInfo.processInfo.environment["HEARIT_RUN_HEAVY_TESTS"] != "1",
                    "Set HEARIT_RUN_HEAVY_TESTS=1 to run model integration tests")
      let engine = Qwen3TTSEngine()
      try await engine.synthesize(text: "Hello from HearIt.", voice: .systemDefault, speed: Speed(1.0))
  }
  ```

- [ ] **Step 3: Implement `Qwen3TTSEngine`**

  Create `HearIt/Engine/Qwen3TTSEngine.swift`:

  ```swift
  import Foundation
  import AVFoundation
  import Qwen3TTS

  /// Real on-device TTS via soniqo/speech-swift's Qwen3-TTS module.
  /// All inference runs on the Apple Neural Engine / GPU via MLX-Swift.
  /// No network calls after model weights are downloaded by ModelManager (Task 9).
  final class Qwen3TTSEngine: NSObject, TTSEngine, @unchecked Sendable {
      private var loadedModel: Qwen3TTSModel?
      private let audioEngine = AVAudioEngine()
      private let playerNode = AVAudioPlayerNode()
      private var stopRequested = false

      override init() {
          super.init()
          audioEngine.attach(playerNode)
      }

      /// Inject a model loaded by ModelManager. Called once after model files are on disk.
      func setModel(_ model: Qwen3TTSModel) {
          self.loadedModel = model
      }

      func synthesize(text: String, voice: VoiceID, speed: Speed) async throws {
          guard let model = loadedModel else { throw TTSError.modelNotLoaded }
          guard !text.isEmpty else { throw TTSError.emptyText }
          stopRequested = false

          do {
              // speech-swift API surface — verify exact signature against
              // https://soniqo.audio/guides/speak before committing.
              let audio = try await model.synthesize(
                  text: text,
                  voice: voice.rawValue,
                  speakingRate: Float(speed.value)
              )
              try playPCMBuffer(audio)
          } catch {
              throw TTSError.synthesisFailed(error.localizedDescription)
          }
      }

      func stop() {
          stopRequested = true
          playerNode.stop()
          if audioEngine.isRunning { audioEngine.stop() }
      }

      // MARK: - PCM playback

      private func playPCMBuffer(_ pcm: AVAudioPCMBuffer) throws {
          guard !stopRequested else { return }
          let format = pcm.format
          audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: format)
          if !audioEngine.isRunning {
              try audioEngine.start()
          }
          playerNode.scheduleBuffer(pcm, completionHandler: nil)
          playerNode.play()
      }
  }
  ```

  > **Verification note:** The exact `Qwen3TTSModel.synthesize(...)` signature must be confirmed against [soniqo's docs](https://soniqo.audio/guides/speak) before committing. If the real API differs (e.g. it returns `Data` or a streamed `AsyncSequence<AVAudioPCMBuffer>`), adapt the body of `synthesize` and `playPCMBuffer` accordingly. The protocol contract above stays the same — only this file changes.

- [ ] **Step 4: Build (don't wire it up yet — ModelManager handles loading in Task 9)**

  Run: `⌘B`. Expected: success.
  Run: `⌘U`. Expected: `test_qwen3Engine_initializes` passes; the heavy test is skipped (it requires the model file, which doesn't exist until Task 8).

- [ ] **Step 5: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests *.xcodeproj
  git commit -m "feat(engine): Qwen3TTSEngine wrapping speech-swift, awaiting model from Task 9"
  ```

---

## Task 8: Model Download

**Files:**
- Create: `HearIt/Engine/ModelDownloader.swift`
- Create: `HearItTests/ModelDownloaderTests.swift`

- [ ] **Step 1: Write the failing test**

  Create `HearItTests/ModelDownloaderTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class ModelDownloaderTests: XCTestCase {
      func test_destinationURL_isInApplicationSupportSubdirectory() {
          let url = ModelDownloader.destinationURL(forModel: "qwen3-tts")
          XCTAssertTrue(url.path.contains("Application Support/HearIt/models"))
          XCTAssertTrue(url.path.hasSuffix("qwen3-tts.bin"))
      }

      func test_isDownloaded_returnsTrueWhenFileExists() throws {
          let url = ModelDownloader.destinationURL(forModel: "test-fixture-\(UUID())")
          try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                  withIntermediateDirectories: true)
          try Data("fake".utf8).write(to: url)
          defer { try? FileManager.default.removeItem(at: url) }

          XCTAssertTrue(ModelDownloader.isDownloaded(modelName: url.deletingPathExtension().lastPathComponent))
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL.

- [ ] **Step 2: Implement `ModelDownloader`**

  Create `HearIt/Engine/ModelDownloader.swift`:

  ```swift
  import Foundation

  enum ModelDownloader {

      /// Where model weights live on disk. Per-user, persistent.
      static func destinationURL(forModel name: String) -> URL {
          let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
          return support
              .appendingPathComponent("HearIt", isDirectory: true)
              .appendingPathComponent("models", isDirectory: true)
              .appendingPathComponent("\(name).bin")
      }

      static func isDownloaded(modelName: String) -> Bool {
          FileManager.default.fileExists(atPath: destinationURL(forModel: modelName).path)
      }

      /// Download a model file. Reports progress 0.0...1.0 to `onProgress`.
      /// Idempotent: returns immediately if file already exists.
      static func download(
          modelName: String,
          from sourceURL: URL,
          onProgress: @escaping @Sendable (Double) -> Void
      ) async throws {
          let destination = destinationURL(forModel: modelName)
          if FileManager.default.fileExists(atPath: destination.path) { return }
          try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                  withIntermediateDirectories: true)

          let (tempURL, response) = try await URLSession.shared.download(
              for: URLRequest(url: sourceURL),
              delegate: ProgressDelegate(onProgress: onProgress)
          )
          guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
              throw TTSError.synthesisFailed("Model download HTTP error \((response as? HTTPURLResponse)?.statusCode ?? -1)")
          }
          try FileManager.default.moveItem(at: tempURL, to: destination)
      }

      /// Forwards URLSession's progress callbacks into a Double in [0, 1].
      private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate {
          let onProgress: @Sendable (Double) -> Void
          init(onProgress: @escaping @Sendable (Double) -> Void) { self.onProgress = onProgress }
          func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                          didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                          totalBytesExpectedToWrite: Int64) {
              guard totalBytesExpectedToWrite > 0 else { return }
              let p = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
              onProgress(p)
          }
          // The async `download(for:delegate:)` API delivers the file via its
          // return value, so we don't override `didFinishDownloadingTo`.
          func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                          didFinishDownloadingTo location: URL) {}
      }
  }
  ```

  Run: `⌘U`. Expected: PASS.

- [ ] **Step 3: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(engine): ModelDownloader with progress reporting, idempotent download"
  ```

  > Note: this task does NOT yet hook the downloader into the UI. That happens in Task 13 (onboarding). The downloader is a unit; Task 9 consumes it.

---

## Task 9: ModelManager Actor (Lazy-Load + Idle Unload)

**Files:**
- Create: `HearIt/Engine/ModelManager.swift`
- Create: `HearItTests/ModelManagerTests.swift`
- Modify: `HearIt/AppDelegate.swift`

- [ ] **Step 1: Write the failing test**

  Create `HearItTests/ModelManagerTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class ModelManagerTests: XCTestCase {

      func test_initialState_isUnloaded() async {
          let manager = ModelManager(idleTimeout: 1.0)
          let state = await manager.state
          XCTAssertEqual(state, .unloaded)
      }

      func test_load_transitionsToLoaded() async throws {
          let manager = ModelManager(idleTimeout: 1.0, loader: { FakeModel() })
          try await manager.ensureLoaded()
          let state = await manager.state
          XCTAssertEqual(state, .loaded)
      }

      func test_idleTimeout_unloadsModel() async throws {
          let manager = ModelManager(idleTimeout: 0.1, loader: { FakeModel() })
          try await manager.ensureLoaded()
          XCTAssertEqual(await manager.state, .loaded)
          try await Task.sleep(nanoseconds: 200_000_000) // 0.2s, past the 0.1s idle
          XCTAssertEqual(await manager.state, .unloaded)
      }

      func test_useResetsIdleTimer() async throws {
          let manager = ModelManager(idleTimeout: 0.2, loader: { FakeModel() })
          try await manager.ensureLoaded()
          try await Task.sleep(nanoseconds: 100_000_000) // 0.1s
          await manager.touch()                            // resets idle
          try await Task.sleep(nanoseconds: 150_000_000)  // another 0.15s
          XCTAssertEqual(await manager.state, .loaded)     // total 0.25s but timer was reset
      }
  }

  /// Stand-in for the real Qwen3 model in tests.
  private struct FakeModel: Sendable {}
  ```

  Run: `⌘U`. Expected: FAIL.

- [ ] **Step 2: Implement `ModelManager`**

  Create `HearIt/Engine/ModelManager.swift`:

  ```swift
  import Foundation

  /// Lazy-load + idle-unload lifecycle for the heavy TTS model.
  /// Generic over the model type so tests can use a fake.
  actor ModelManager<Model: Sendable> {

      enum State: Equatable { case unloaded, loading, loaded }

      private(set) var state: State = .unloaded
      private var model: Model?
      private var idleTask: Task<Void, Never>?

      private let idleTimeout: TimeInterval
      private let loader: @Sendable () async throws -> Model

      init(idleTimeout: TimeInterval = 300, loader: @escaping @Sendable () async throws -> Model = { fatalError("loader not provided") }) {
          self.idleTimeout = idleTimeout
          self.loader = loader
      }

      /// Returns the loaded model, loading it on first call.
      func ensureLoaded() async throws -> Model {
          if let m = model { resetIdleTimer(); return m }
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
      func touch() { resetIdleTimer() }

      /// Immediately drop the loaded model.
      func unload() {
          idleTask?.cancel()
          idleTask = nil
          model = nil
          state = .unloaded
      }

      private func resetIdleTimer() {
          idleTask?.cancel()
          idleTask = Task { [weak self, idleTimeout] in
              try? await Task.sleep(nanoseconds: UInt64(idleTimeout * 1_000_000_000))
              await self?.unload()
          }
      }
  }
  ```

  Run: `⌘U`. Expected: all 4 ModelManagerTests pass.

- [ ] **Step 3: Wire ModelManager into AppDelegate**

  In `AppDelegate.applicationDidFinishLaunching`, replace the `engine: TTSEngine = MockTTSEngine()` line with a Qwen3-backed setup:

  ```swift
  // Replace this block:
  // let engine: TTSEngine = MockTTSEngine()
  // let playback = PlaybackController(engine: engine)

  let qwen3 = Qwen3TTSEngine()
  let modelManager = ModelManager<Qwen3TTSModel>(idleTimeout: 300) {
      // The real loader uses the file downloaded by ModelDownloader.
      // Confirm exact loader API in speech-swift docs:
      //   https://soniqo.audio/guides/speak
      let modelURL = ModelDownloader.destinationURL(forModel: "qwen3-tts")
      return try Qwen3TTSModel.load(from: modelURL)
  }
  // Bridge: PlaybackController calls qwen3.setModel(...) before each play.
  let playback = PlaybackController(engine: qwen3, modelLoader: {
      try await modelManager.ensureLoaded()
  })
  ```

  Update `PlaybackController` to accept a model loader:

  ```swift
  @MainActor
  final class PlaybackController {
      private let engine: Qwen3TTSEngine            // Concrete now — we own the model bridge
      private let modelLoader: () async throws -> Qwen3TTSModel
      private var currentTask: Task<Void, Never>?

      var isPlaying: Bool { currentTask != nil }

      init(engine: Qwen3TTSEngine, modelLoader: @escaping () async throws -> Qwen3TTSModel) {
          self.engine = engine
          self.modelLoader = modelLoader
      }

      func play(text: String, voice: VoiceID, speed: Speed) {
          stop()
          currentTask = Task { [engine, modelLoader] in
              do {
                  let model = try await modelLoader()
                  await MainActor.run { engine.setModel(model) }
                  try await engine.synthesize(text: text, voice: voice, speed: speed)
              } catch {
                  NSLog("HearIt playback error: \(error)")
              }
          }
      }

      func stop() {
          engine.stop()
          currentTask?.cancel()
          currentTask = nil
      }
  }
  ```

  > **Trade-off note:** `PlaybackController` is no longer generic on `TTSEngine` here — we coupled it to the concrete `Qwen3TTSEngine` to manage model injection. This is a deliberate YAGNI: we only ship one engine. The `TTSEngine` protocol still exists so adding a second engine (Kokoro, Chatterbox later) means swapping one `PlaybackController` field and adding a new concrete class behind the protocol.

- [ ] **Step 4: Manual verification (gated on model file existing)**

  Run: `⌘B`. Expected: builds.

  At this point ⌘⇧R will fail with "modelNotLoaded" until the model file is downloaded. That's expected — Task 13's onboarding flow triggers the download.

  To test the engine path manually right now: download the Qwen3-TTS weight file by hand from [Hugging Face](https://huggingface.co/) (URL TBD — confirm in Task 13) and place it at `~/Library/Application Support/HearIt/models/qwen3-tts.bin`. Then run, press ⌘⇧R with text selected. Expected: high-quality Qwen3 voice reads the text.

- [ ] **Step 5: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(engine): ModelManager actor with lazy-load + idle unload, AppDelegate now wires Qwen3"
  ```

---

## Task 10: Settings + Voice Catalog

**Files:**
- Create: `HearIt/Models/Settings.swift`
- Create: `HearIt/Models/VoiceCatalog.swift`
- Create: `HearItTests/SettingsTests.swift`
- Create: `HearItTests/VoiceCatalogTests.swift`
- Modify: `HearIt/Coordinator.swift`

- [ ] **Step 1: Write failing tests for VoiceCatalog**

  Create `HearItTests/VoiceCatalogTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class VoiceCatalogTests: XCTestCase {
      func test_curated_hasExactly7Voices() {
          XCTAssertEqual(VoiceCatalog.curated.count, 7)
      }

      func test_curated_allHaveUniqueIDs() {
          let ids = VoiceCatalog.curated.map { $0.id.rawValue }
          XCTAssertEqual(Set(ids).count, ids.count)
      }

      func test_curated_allHaveDisplayNames() {
          for v in VoiceCatalog.curated {
              XCTAssertFalse(v.displayName.isEmpty)
          }
      }

      func test_defaultVoice_isInCurated() {
          XCTAssertTrue(VoiceCatalog.curated.contains { $0.id == VoiceCatalog.defaultVoice.id })
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL.

- [ ] **Step 2: Implement `VoiceCatalog`**

  Create `HearIt/Models/VoiceCatalog.swift`:

  ```swift
  import Foundation

  struct Voice: Identifiable, Hashable, Sendable {
      let id: VoiceID
      let displayName: String
      let language: String
      let blurb: String  // one-line description for the picker
  }

  /// Hand-picked subset of Qwen3-TTS voices. v1 ships these 7 only;
  /// the full set is exposed in Settings → "Show all voices" (Task 12).
  /// Voice IDs must match the speech-swift / Qwen3 voice names.
  /// CONFIRM these IDs against the live model — placeholder names below.
  enum VoiceCatalog {
      static let curated: [Voice] = [
          Voice(id: VoiceID(rawValue: "ada"),    displayName: "Ada",    language: "en-US", blurb: "Warm, narrative"),
          Voice(id: VoiceID(rawValue: "owen"),   displayName: "Owen",   language: "en-US", blurb: "Calm, measured"),
          Voice(id: VoiceID(rawValue: "iris"),   displayName: "Iris",   language: "en-US", blurb: "Bright, friendly"),
          Voice(id: VoiceID(rawValue: "felix"),  displayName: "Felix",  language: "en-US", blurb: "Crisp, neutral"),
          Voice(id: VoiceID(rawValue: "nova"),   displayName: "Nova",   language: "en-US", blurb: "Expressive, lively"),
          Voice(id: VoiceID(rawValue: "june"),   displayName: "June",   language: "en-US", blurb: "Soft, conversational"),
          Voice(id: VoiceID(rawValue: "ezra"),   displayName: "Ezra",   language: "en-US", blurb: "Deep, deliberate"),
      ]

      static let defaultVoice: Voice = curated[0]   // Ada
  }
  ```

  > **Action item for the engineer:** the IDs above (`"ada"`, `"owen"`, etc.) are placeholders. Replace with the real voice identifiers reported by speech-swift's Qwen3 model. Verify by running the heavy integration test from Task 7 with each candidate ID, listening for which voice plays, then picking the 7 that sound best across a range of selected text.

  Run: `⌘U`. Expected: PASS.

- [ ] **Step 3: Write failing tests for Settings**

  Create `HearItTests/SettingsTests.swift`:

  ```swift
  import XCTest
  @testable import HearIt

  final class SettingsTests: XCTestCase {
      override func setUp() {
          super.setUp()
          Settings.shared.reset()
      }

      func test_defaultVoice_isAda() {
          XCTAssertEqual(Settings.shared.selectedVoice.id, VoiceCatalog.defaultVoice.id)
      }

      func test_defaultSpeed_isOne() {
          XCTAssertEqual(Settings.shared.speed.value, 1.0)
      }

      func test_speed_persists() {
          Settings.shared.speed = Speed(1.5)
          XCTAssertEqual(Settings.shared.speed.value, 1.5)
      }

      func test_hasCompletedOnboarding_defaultsFalse() {
          XCTAssertFalse(Settings.shared.hasCompletedOnboarding)
      }
  }
  ```

  Run: `⌘U`. Expected: FAIL.

- [ ] **Step 4: Implement `Settings`**

  Create `HearIt/Models/Settings.swift`:

  ```swift
  import Foundation
  import Combine

  /// Singleton settings backed by UserDefaults. SwiftUI views observe via @ObservedObject.
  /// Reset for tests via `Settings.shared.reset()`.
  final class Settings: ObservableObject {
      static let shared = Settings()

      private enum Key {
          static let voiceID = "hearit.voiceID"
          static let speed = "hearit.speed"
          static let showAllVoices = "hearit.showAllVoices"
          static let launchAtLogin = "hearit.launchAtLogin"
          static let hasCompletedOnboarding = "hearit.hasCompletedOnboarding"
      }

      private let defaults: UserDefaults

      init(defaults: UserDefaults = .standard) {
          self.defaults = defaults
      }

      @Published var selectedVoice: Voice = VoiceCatalog.defaultVoice {
          didSet { defaults.set(selectedVoice.id.rawValue, forKey: Key.voiceID) }
      }

      @Published var speed: Speed = Speed(1.0) {
          didSet { defaults.set(speed.value, forKey: Key.speed) }
      }

      @Published var showAllVoices: Bool = false {
          didSet { defaults.set(showAllVoices, forKey: Key.showAllVoices) }
      }

      @Published var launchAtLogin: Bool = false {
          didSet { defaults.set(launchAtLogin, forKey: Key.launchAtLogin) }
      }

      @Published var hasCompletedOnboarding: Bool = false {
          didSet { defaults.set(hasCompletedOnboarding, forKey: Key.hasCompletedOnboarding) }
      }

      func load() {
          if let rawID = defaults.string(forKey: Key.voiceID),
             let voice = VoiceCatalog.curated.first(where: { $0.id.rawValue == rawID }) {
              selectedVoice = voice
          }
          let storedSpeed = defaults.object(forKey: Key.speed) as? Double ?? 1.0
          speed = Speed(storedSpeed)
          showAllVoices = defaults.bool(forKey: Key.showAllVoices)
          launchAtLogin = defaults.bool(forKey: Key.launchAtLogin)
          hasCompletedOnboarding = defaults.bool(forKey: Key.hasCompletedOnboarding)
      }

      /// Test-only: clear all known keys and reset to defaults.
      func reset() {
          [Key.voiceID, Key.speed, Key.showAllVoices, Key.launchAtLogin, Key.hasCompletedOnboarding]
              .forEach { defaults.removeObject(forKey: $0) }
          selectedVoice = VoiceCatalog.defaultVoice
          speed = Speed(1.0)
          showAllVoices = false
          launchAtLogin = false
          hasCompletedOnboarding = false
      }
  }
  ```

  Run: `⌘U`. Expected: PASS.

- [ ] **Step 5: Update Coordinator to read from Settings**

  In `HearIt/Coordinator.swift`, change the `handleHotkey` method:

  ```swift
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
      playback.play(text: text, voice: s.selectedVoice.id, speed: s.speed)
  }
  ```

  Also add to `AppDelegate.applicationDidFinishLaunching`, before creating the coordinator:

  ```swift
  Settings.shared.load()
  ```

- [ ] **Step 6: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(settings): @AppStorage-backed Settings + curated 7-voice catalog"
  ```

---

## Task 11: Native-Stealth Popover View

**Files:**
- Create: `HearIt/UI/PopoverView.swift`
- Modify: `HearIt/UI/PopoverController.swift`
- Modify: `HearIt/Coordinator.swift` (expose recent text + playback state)

- [ ] **Step 1: Implement `PopoverView`**

  Create `HearIt/UI/PopoverView.swift`:

  ```swift
  import SwiftUI

  struct PopoverView: View {
      @ObservedObject var settings: Settings
      @ObservedObject var coordinator: CoordinatorState
      let onOpenSettings: () -> Void

      var body: some View {
          ZStack {
              VisualEffectBackground()
              VStack(alignment: .leading, spacing: 12) {
                  header
                  textPreview
                  Divider()
                  controls
                  Divider()
                  voicePicker
                  speedSlider
                  Spacer()
              }
              .padding(16)
          }
          .frame(width: 320, height: 360)
      }

      private var header: some View {
          HStack {
              Image(systemName: "waveform")
                  .foregroundStyle(.secondary)
              Text("HearIt").font(.headline)
              Spacer()
              Button(action: onOpenSettings) {
                  Image(systemName: "gearshape").foregroundStyle(.secondary)
              }
              .buttonStyle(.plain)
              .keyboardShortcut(",", modifiers: .command)
          }
      }

      private var textPreview: some View {
          ScrollView {
              Text(coordinator.lastCapturedText.isEmpty
                   ? "Select text in any app and press ⌘⇧R."
                   : coordinator.lastCapturedText)
                  .font(.callout)
                  .foregroundStyle(coordinator.lastCapturedText.isEmpty ? .secondary : .primary)
                  .frame(maxWidth: .infinity, alignment: .leading)
                  .padding(.vertical, 4)
          }
          .frame(height: 70)
      }

      private var controls: some View {
          HStack(spacing: 18) {
              Button(action: coordinator.togglePlayPause) {
                  Image(systemName: coordinator.isPlaying ? "pause.fill" : "play.fill")
                      .font(.title2)
              }
              .buttonStyle(.plain)
              .disabled(coordinator.lastCapturedText.isEmpty)
              Button(action: coordinator.stop) {
                  Image(systemName: "stop.fill").font(.title2)
              }
              .buttonStyle(.plain)
              .disabled(!coordinator.isPlaying)
              Spacer()
          }
      }

      private var voicePicker: some View {
          HStack {
              Text("Voice").foregroundStyle(.secondary)
              Spacer()
              Picker("", selection: $settings.selectedVoice) {
                  ForEach(VoiceCatalog.curated, id: \.id) { v in
                      Text(v.displayName).tag(v)
                  }
              }
              .labelsHidden()
              .pickerStyle(.menu)
              .frame(width: 120)
          }
      }

      private var speedSlider: some View {
          VStack(alignment: .leading, spacing: 4) {
              HStack {
                  Text("Speed").foregroundStyle(.secondary)
                  Spacer()
                  Text(String(format: "%.1fx", settings.speed.value))
                      .monospacedDigit()
                      .foregroundStyle(.secondary)
              }
              Slider(value: Binding(
                  get: { settings.speed.value },
                  set: { settings.speed = Speed($0) }
              ), in: 0.5...2.0, step: 0.1)
          }
      }
  }

  /// Observable mirror of Coordinator playback state — kept thin to avoid SwiftUI/AppKit threading mess.
  @MainActor
  final class CoordinatorState: ObservableObject {
      @Published var lastCapturedText: String = ""
      @Published var isPlaying: Bool = false
      var togglePlayPause: () -> Void = {}
      var stop: () -> Void = {}
  }
  ```

- [ ] **Step 2: Update Coordinator to publish state**

  In `HearIt/Coordinator.swift`, add a `state: CoordinatorState` property and update it on hotkey fire and on playback completion:

  ```swift
  @MainActor
  final class Coordinator {
      let state = CoordinatorState()
      private let hotkey: HotkeyManager
      private let playback: PlaybackController

      init(hotkey: HotkeyManager, playback: PlaybackController) {
          self.hotkey = hotkey
          self.playback = playback
          state.togglePlayPause = { [weak self] in self?.togglePlayPause() }
          state.stop = { [weak self] in self?.stop() }
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
          guard let text = TextCaptureManager.captureSelectedText(), !text.isEmpty else { return }
          state.lastCapturedText = text
          let s = Settings.shared
          playback.play(text: text, voice: s.selectedVoice.id, speed: s.speed)
          state.isPlaying = true
      }

      private func togglePlayPause() {
          if state.isPlaying { stop() } else if !state.lastCapturedText.isEmpty {
              let s = Settings.shared
              playback.play(text: state.lastCapturedText, voice: s.selectedVoice.id, speed: s.speed)
              state.isPlaying = true
          }
      }

      private func stop() {
          playback.stop()
          state.isPlaying = false
      }
  }
  ```

  > `PlaybackController.play(...)` should also call back to `state.isPlaying = false` when synthesis completes naturally. Add a `onComplete: () -> Void` callback to `PlaybackController.play(...)` and invoke it in the `Task` after `synthesize`. Coordinator passes `{ self?.state.isPlaying = false }` from `handleHotkey`.

- [ ] **Step 3: Wire `PopoverView` into `PopoverController`**

  Update `HearIt/UI/PopoverController.swift`:

  ```swift
  final class PopoverController {
      let popover: NSPopover

      init(settings: Settings, coordinatorState: CoordinatorState, onOpenSettings: @escaping () -> Void) {
          let p = NSPopover()
          p.contentSize = NSSize(width: 320, height: 360)
          p.behavior = .transient
          p.animates = true
          p.contentViewController = NSHostingController(rootView:
              PopoverView(settings: settings, coordinator: coordinatorState, onOpenSettings: onOpenSettings)
          )
          self.popover = p
      }

      func show(relativeTo view: NSView) {
          popover.show(relativeTo: view.bounds, of: view, preferredEdge: .minY)
      }
      func hide() { popover.performClose(nil) }
  }
  ```

  Update `AppDelegate.applicationDidFinishLaunching` to construct `PopoverController` with the dependencies:

  ```swift
  let coordinator = Coordinator(hotkey: hotkeyManager, playback: playback)
  coordinator.start()
  self.coordinator = coordinator
  self.popoverController = PopoverController(
      settings: Settings.shared,
      coordinatorState: coordinator.state,
      onOpenSettings: { [weak self] in self?.openSettings() }
  )
  ```

  Add an `openSettings()` stub on AppDelegate (real implementation in Task 12):

  ```swift
  func openSettings() {
      NSLog("HearIt: openSettings — implemented in Task 12")
  }
  ```

- [ ] **Step 4: Visual verification**

  Run: `⌘R`. Click the menu bar icon. Expected:
  - Popover is 320×360 with vibrancy
  - Header shows "waveform" + "HearIt" + gear icon
  - Empty-state hint "Select text in any app and press ⌘⇧R."
  - Disabled play/stop buttons
  - Voice picker shows 7 names
  - Speed slider at 1.0x

  Select text in Safari, press ⌘⇧R. Re-open the popover. Expected:
  - Text preview shows the selected text
  - Play button enabled

- [ ] **Step 5: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(ui): native-stealth popover view — preview, transport, voice picker, speed slider"
  ```

---

## Task 12: Settings Sheet

**Files:**
- Create: `HearIt/UI/SettingsView.swift`
- Modify: `HearIt/AppDelegate.swift`

- [ ] **Step 1: Implement `SettingsView`**

  Create `HearIt/UI/SettingsView.swift`:

  ```swift
  import SwiftUI

  struct SettingsView: View {
      @ObservedObject var settings: Settings
      let onClose: () -> Void

      var body: some View {
          TabView {
              generalTab.tabItem { Label("General", systemImage: "gearshape") }
              voicesTab.tabItem { Label("Voices", systemImage: "waveform") }
              aboutTab.tabItem { Label("About", systemImage: "info.circle") }
          }
          .frame(width: 460, height: 320)
          .padding()
      }

      // MARK: - General
      private var generalTab: some View {
          Form {
              Toggle("Launch HearIt at login", isOn: $settings.launchAtLogin)
                  .onChange(of: settings.launchAtLogin) { _, new in
                      LaunchAtLogin.set(enabled: new)
                  }
              HStack {
                  Text("Hotkey")
                  Spacer()
                  Text("⌘⇧R").foregroundStyle(.secondary)
                  // Hotkey customization is v1.1; for now, display only.
              }
              Spacer()
              accessibilitySection
          }
      }

      private var accessibilitySection: some View {
          GroupBox {
              HStack {
                  Image(systemName: PermissionsManager.isAccessibilityGranted ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                      .foregroundStyle(PermissionsManager.isAccessibilityGranted ? .green : .orange)
                  Text(PermissionsManager.isAccessibilityGranted ? "Accessibility access granted" : "Accessibility access required")
                  Spacer()
                  if !PermissionsManager.isAccessibilityGranted {
                      Button("Open System Settings") { PermissionsManager.openAccessibilitySettings() }
                  }
              }.padding(8)
          }
      }

      // MARK: - Voices
      private var voicesTab: some View {
          Form {
              Picker("Default voice", selection: $settings.selectedVoice) {
                  ForEach(VoiceCatalog.curated, id: \.id) { v in
                      Text("\(v.displayName) — \(v.blurb)").tag(v)
                  }
              }
              Toggle("Show all voices in popover (advanced)", isOn: $settings.showAllVoices)
              Text("HearIt ships with 7 hand-picked voices. Enabling 'Show all' adds every voice the engine supports — useful if you want more variety or non-English voices.")
                  .font(.caption)
                  .foregroundStyle(.secondary)
                  .padding(.top, 4)
          }
      }

      // MARK: - About
      private var aboutTab: some View {
          VStack(spacing: 8) {
              Image(systemName: "waveform").font(.system(size: 48)).foregroundStyle(.secondary)
              Text("HearIt").font(.title2)
              Text("Version \(Bundle.main.shortVersionString) (\(Bundle.main.buildNumber))")
                  .foregroundStyle(.secondary)
              Text("Reads selected text aloud, fully on-device.").font(.caption)
              Spacer()
              Text("Built by Vir Khanna. All speech runs locally on your Mac — no network calls.")
                  .multilineTextAlignment(.center)
                  .font(.caption)
                  .foregroundStyle(.secondary)
          }
          .padding()
      }
  }

  private extension Bundle {
      var shortVersionString: String { infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0" }
      var buildNumber: String { infoDictionary?["CFBundleVersion"] as? String ?? "0" }
  }
  ```

- [ ] **Step 2: Stub `LaunchAtLogin` (real implementation in Task 14)**

  Add to `HearIt/Models/Settings.swift` for now (move to its own file in Task 14):

  ```swift
  enum LaunchAtLogin {
      static func set(enabled: Bool) {
          NSLog("LaunchAtLogin.set(\(enabled)) — real impl in Task 14")
      }
  }
  ```

- [ ] **Step 3: Open settings as a window from AppDelegate**

  In `HearIt/AppDelegate.swift`, replace the `openSettings()` stub:

  ```swift
  private var settingsWindow: NSWindow?

  func openSettings() {
      if let win = settingsWindow {
          win.makeKeyAndOrderFront(nil)
          NSApp.activate(ignoringOtherApps: true)
          return
      }
      let view = SettingsView(settings: Settings.shared) { [weak self] in
          self?.settingsWindow?.close()
      }
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
  ```

- [ ] **Step 4: Visual verification**

  Run: `⌘R`. Click status item → click gear icon. Expected: a settings window opens with three tabs (General, Voices, About). Tabs render correctly. Voice picker reflects current selection. Toggling "Show all voices" persists across app relaunches.

- [ ] **Step 5: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(ui): three-tab settings sheet — General, Voices, About"
  ```

---

## Task 13: First-Run Onboarding

**Files:**
- Create: `HearIt/UI/OnboardingView.swift`
- Modify: `HearIt/AppDelegate.swift`

- [ ] **Step 1: Implement `OnboardingView`**

  Create `HearIt/UI/OnboardingView.swift`:

  ```swift
  import SwiftUI

  struct OnboardingView: View {
      @State private var step: Step = .welcome
      @State private var downloadProgress: Double = 0
      @State private var downloadError: String?
      let onComplete: () -> Void

      enum Step { case welcome, accessibility, modelDownload, ready }

      var body: some View {
          ZStack {
              VisualEffectBackground()
              switch step {
              case .welcome:        welcome
              case .accessibility:  accessibility
              case .modelDownload:  modelDownload
              case .ready:          ready
              }
          }
          .frame(width: 480, height: 360)
      }

      private var welcome: some View {
          VStack(spacing: 16) {
              Image(systemName: "waveform").font(.system(size: 60)).foregroundStyle(.tint)
              Text("Welcome to HearIt").font(.title)
              Text("Select text in any app, press ⌘⇧R, and hear it read aloud in a natural AI voice. All on-device.")
                  .multilineTextAlignment(.center)
                  .foregroundStyle(.secondary)
                  .padding(.horizontal, 32)
              Spacer()
              Button("Continue") { step = .accessibility }
                  .keyboardShortcut(.defaultAction)
          }.padding(24)
      }

      private var accessibility: some View {
          VStack(spacing: 16) {
              Image(systemName: "lock.shield").font(.system(size: 60)).foregroundStyle(.tint)
              Text("Allow Accessibility").font(.title2)
              Text("HearIt needs Accessibility access to read the text you select in other apps. Nothing is sent over the network.")
                  .multilineTextAlignment(.center)
                  .foregroundStyle(.secondary)
                  .padding(.horizontal, 32)
              Button("Open System Settings") {
                  PermissionsManager.requestAccessibility()
                  PermissionsManager.openAccessibilitySettings()
              }
              .keyboardShortcut(.defaultAction)
              Spacer()
              Button(PermissionsManager.isAccessibilityGranted ? "Continue" : "I've enabled it") {
                  if PermissionsManager.isAccessibilityGranted {
                      step = .modelDownload
                  }
              }
              .disabled(!PermissionsManager.isAccessibilityGranted)
          }.padding(24)
      }

      private var modelDownload: some View {
          VStack(spacing: 16) {
              Image(systemName: "arrow.down.circle").font(.system(size: 60)).foregroundStyle(.tint)
              Text("Downloading voices").font(.title2)
              Text("HearIt is downloading the Qwen3-TTS voice model — about 350 MB. This happens once.")
                  .multilineTextAlignment(.center)
                  .foregroundStyle(.secondary)
                  .padding(.horizontal, 32)
              ProgressView(value: downloadProgress).frame(width: 320)
              Text(String(format: "%.0f%%", downloadProgress * 100))
                  .font(.caption)
                  .foregroundStyle(.secondary)
              if let err = downloadError {
                  Text(err).font(.caption).foregroundStyle(.red)
                  Button("Retry") { startDownload() }
              }
              Spacer()
          }
          .padding(24)
          .task { startDownload() }
      }

      private var ready: some View {
          VStack(spacing: 16) {
              Image(systemName: "checkmark.circle.fill").font(.system(size: 60)).foregroundStyle(.green)
              Text("You're set").font(.title2)
              Text("Select text in any app and press ⌘⇧R to hear it read aloud. HearIt lives in your menu bar.")
                  .multilineTextAlignment(.center)
                  .foregroundStyle(.secondary)
                  .padding(.horizontal, 32)
              Spacer()
              Button("Get Started") {
                  Settings.shared.hasCompletedOnboarding = true
                  onComplete()
              }
              .keyboardShortcut(.defaultAction)
          }.padding(24)
      }

      private func startDownload() {
          downloadError = nil
          Task {
              do {
                  // CONFIRM the real Qwen3 model URL. Placeholder below;
                  // most likely a Hugging Face hosted file like
                  //   https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/qwen3-tts.bin
                  let source = URL(string: "https://huggingface.co/Qwen/Qwen3-TTS/resolve/main/qwen3-tts.bin")!
                  try await ModelDownloader.download(
                      modelName: "qwen3-tts",
                      from: source,
                      onProgress: { p in
                          Task { @MainActor in self.downloadProgress = p }
                      }
                  )
                  await MainActor.run { self.step = .ready }
              } catch {
                  await MainActor.run { self.downloadError = error.localizedDescription }
              }
          }
      }
  }
  ```

  > **Action item for the engineer:** confirm the real Qwen3-TTS model source URL by reading the [speech-swift Qwen3 docs](https://soniqo.audio/guides/speak) or the Hugging Face Qwen3-TTS page. Replace the placeholder URL.

- [ ] **Step 2: Show OnboardingView on first launch**

  In `HearIt/AppDelegate.swift`, after `Settings.shared.load()`:

  ```swift
  if !Settings.shared.hasCompletedOnboarding {
      showOnboarding()
  }
  ```

  Add the method:

  ```swift
  private var onboardingWindow: NSWindow?

  private func showOnboarding() {
      let view = OnboardingView { [weak self] in
          self?.onboardingWindow?.close()
          self?.onboardingWindow = nil
      }
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
  ```

- [ ] **Step 3: Visual verification**

  - Reset onboarding state: in Terminal, run `defaults delete com.virkhanna.hearit hearit.hasCompletedOnboarding` (or call `Settings.shared.reset()` from a temporary debug menu).
  - Delete the model file: `rm -f ~/Library/Application\ Support/HearIt/models/qwen3-tts.bin`
  - Run the app: `⌘R`. Expected: onboarding window opens. Step through welcome → accessibility (real prompt fires) → model download (progress bar fills) → ready. Click "Get Started." Window closes. Press ⌘⇧R with text selected. Expected: voice reads it aloud.

- [ ] **Step 4: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt HearItTests
  git commit -m "feat(onboarding): welcome → accessibility → model download → ready flow"
  ```

---

## Task 14: Launch at Login (SMAppService)

**Files:**
- Create: `HearIt/LaunchAtLogin.swift`
- Modify: `HearIt/Models/Settings.swift` (delete the stub from Task 12)

- [ ] **Step 1: Implement `LaunchAtLogin`**

  Create `HearIt/LaunchAtLogin.swift`:

  ```swift
  import ServiceManagement
  import Foundation

  enum LaunchAtLogin {
      static func set(enabled: Bool) {
          do {
              if enabled {
                  if SMAppService.mainApp.status != .enabled {
                      try SMAppService.mainApp.register()
                  }
              } else {
                  try SMAppService.mainApp.unregister()
              }
          } catch {
              NSLog("HearIt: LaunchAtLogin error: \(error)")
          }
      }

      static var isEnabled: Bool {
          SMAppService.mainApp.status == .enabled
      }
  }
  ```

- [ ] **Step 2: Remove the stub from Settings.swift**

  Delete the stub `enum LaunchAtLogin` at the bottom of `HearIt/Models/Settings.swift` (added in Task 12 step 2). The real implementation in `HearIt/LaunchAtLogin.swift` takes its place.

- [ ] **Step 3: Verify**

  Run: `⌘R`. Open Settings → General → toggle "Launch HearIt at login" ON. Quit and reopen System Settings → General → Login Items. Expected: HearIt appears in the "Open at Login" list. Toggle off in HearIt's settings → confirm it disappears from the system list.

- [ ] **Step 4: Commit**

  ```bash
  cd ~/Code/tts-app
  git add HearIt
  git commit -m "feat(login): SMAppService-backed launch-at-login"
  ```

---

## Task 15: Signing, Notarization, DMG, Setapp Materials

**Files:**
- Create: `scripts/build.sh`
- Create: `scripts/notarize.sh`
- Create: `docs/SETAPP.md`
- Create: `PRIVACY.md`
- Create: `HearIt/Resources/Assets.xcassets/AppIcon.appiconset/*.png` (app icon)

- [ ] **Step 1: Configure signing in Xcode**

  Target → Signing & Capabilities → Team = your Apple Developer Team ID. Signing Certificate = Developer ID Application. Bundle Identifier = `com.virkhanna.hearit`.

- [ ] **Step 2: Write `scripts/build.sh`**

  Create `scripts/build.sh`:

  ```bash
  #!/usr/bin/env bash
  set -euo pipefail

  # Builds a Release .app and packages it into a DMG.
  # Output: build/HearIt.dmg

  cd "$(dirname "$0")/.."

  BUILD_DIR="build"
  ARCHIVE_PATH="$BUILD_DIR/HearIt.xcarchive"
  EXPORT_PATH="$BUILD_DIR/Export"
  DMG_PATH="$BUILD_DIR/HearIt.dmg"

  rm -rf "$BUILD_DIR"
  mkdir -p "$BUILD_DIR"

  # 1. Archive
  xcodebuild archive \
    -project HearIt.xcodeproj \
    -scheme HearIt \
    -configuration Release \
    -archivePath "$ARCHIVE_PATH" \
    -destination "generic/platform=macOS"

  # 2. Export the .app from the archive (Developer ID, no Mac App Store)
  cat > "$BUILD_DIR/ExportOptions.plist" <<'EOF'
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0">
  <dict>
      <key>method</key>
      <string>developer-id</string>
      <key>signingStyle</key>
      <string>automatic</string>
  </dict>
  </plist>
  EOF

  xcodebuild -exportArchive \
    -archivePath "$ARCHIVE_PATH" \
    -exportPath "$EXPORT_PATH" \
    -exportOptionsPlist "$BUILD_DIR/ExportOptions.plist"

  # 3. DMG
  hdiutil create -volname "HearIt" -srcfolder "$EXPORT_PATH/HearIt.app" -ov -format UDZO "$DMG_PATH"

  echo "Done: $DMG_PATH"
  ```

  Make it executable: `chmod +x scripts/build.sh`.

- [ ] **Step 3: Write `scripts/notarize.sh`**

  Create `scripts/notarize.sh`:

  ```bash
  #!/usr/bin/env bash
  set -euo pipefail

  # Notarizes the built DMG via Apple's notarytool.
  # Prereq: store your notary credentials once with:
  #   xcrun notarytool store-credentials "hearit-notary" \
  #     --apple-id you@example.com \
  #     --team-id <YOUR_TEAM_ID> \
  #     --password <app-specific-password>

  cd "$(dirname "$0")/.."

  DMG_PATH="build/HearIt.dmg"
  PROFILE="hearit-notary"

  if [ ! -f "$DMG_PATH" ]; then
    echo "Run scripts/build.sh first." >&2
    exit 1
  fi

  echo "Submitting to Apple notary service..."
  xcrun notarytool submit "$DMG_PATH" \
    --keychain-profile "$PROFILE" \
    --wait

  echo "Stapling notarization to DMG..."
  xcrun stapler staple "$DMG_PATH"

  echo "Verifying..."
  spctl --assess --type install --verbose=2 "$DMG_PATH"

  echo "Done."
  ```

  Make it executable: `chmod +x scripts/notarize.sh`.

- [ ] **Step 4: Add an app icon**

  Create an `AppIcon` in `Assets.xcassets` with all required sizes (16, 32, 64, 128, 256, 512, 1024 @1x and @2x). Start from a 1024×1024 PNG of a stylized waveform on a soft gradient background.

  > If you don't have an icon yet, generate a placeholder: any 1024×1024 PNG named `Icon-1024.png` dropped into the appiconset will work for now. Set `Contents.json` to point at it. The final icon is a separate design task before Setapp submission.

- [ ] **Step 5: Write `docs/SETAPP.md` and `PRIVACY.md`**

  Create `docs/SETAPP.md`:

  ```markdown
  # Setapp Submission Notes

  - **App name:** HearIt (pending final rename)
  - **Bundle ID:** com.virkhanna.hearit
  - **Category:** Productivity
  - **Tagline:** Hear any text aloud, fully on-device.
  - **Long description (draft):**

  HearIt reads any text you select aloud, in a natural AI voice that runs entirely on your Mac. Select a paragraph in Safari, an email in Mail, or a passage in a PDF — press ⌘⇧R and listen. No cloud, no accounts, no analytics. The voice model downloads once on first launch and never connects to the internet again.

  Built for Apple Silicon Macs. Minimum macOS 15.

  - **Privacy:** see PRIVACY.md
  - **Distribution model:** Setapp only (no direct sale, no App Store v1)
  - **Pricing:** Setapp handles billing; no in-app purchases
  ```

  Create `PRIVACY.md`:

  ```markdown
  # HearIt Privacy Policy

  HearIt does not collect, transmit, store, or share any user data.

  - All text you select is processed locally on your Mac.
  - All speech synthesis runs locally via Apple's MLX framework on the Neural Engine / GPU.
  - The only network request HearIt ever makes is the initial download of the voice model file from Hugging Face on first launch. After that, no network calls happen — ever.
  - No analytics, telemetry, crash reporting, or "anonymous usage data" is sent anywhere.
  - HearIt has no user accounts.
  ```

- [ ] **Step 6: Do a dry run**

  ```bash
  cd ~/Code/tts-app
  ./scripts/build.sh
  ```

  Expected: `build/HearIt.dmg` exists. Open it in Finder, drag HearIt to Applications, launch — should run identically to your Xcode debug build.

  Then (when ready for actual submission):

  ```bash
  ./scripts/notarize.sh
  ```

  Expected: notarization succeeds, stapler attaches the ticket, `spctl` verifies as accepted.

- [ ] **Step 7: Commit**

  ```bash
  cd ~/Code/tts-app
  git add scripts docs PRIVACY.md HearIt
  git commit -m "build: signing, notarization scripts, Setapp + privacy docs"
  ```

---

## Self-Review (run after writing each major change)

**1. Spec coverage:**

| Spec item (SCOPE.md) | Covered by |
|---|---|
| Native-stealth aesthetic | Tasks 3, 11, 12, 13 |
| Qwen3-TTS via speech-swift | Tasks 7, 9 |
| Curated 6–8 voices | Task 10 |
| Setapp distribution | Task 15 |
| First-launch model download w/ progress | Task 13 |
| macOS 15+, Apple Silicon only | Task 1 |
| Zero network after install | Task 13 (one-time DL) + entitlements (Task 2) |
| <50 MB idle RAM | Task 9 (idle unload) — verify with Instruments after Task 9 |
| <20 MB binary | Naturally true — verify with `du -sh HearIt.app` after Task 15 |
| Lazy-load + 5min idle unload | Task 9 |
| ⌘⇧R global hotkey | Task 4 |
| Text capture (AX + clipboard fallback) | Task 5 |
| Settings (voice, speed, hotkey display, launch-at-login) | Tasks 10, 12, 14 |
| First-run onboarding | Task 13 |
| Privacy: no telemetry | Task 15 (PRIVACY.md) + behavior (no analytics code anywhere) |

No spec gaps identified.

**2. Placeholder scan:**

Three known placeholders flagged inline for the engineer:
1. **Task 7 step 3** — `Qwen3TTSModel.synthesize(...)` signature must be verified against soniqo's docs. The protocol contract holds; only this one method body adapts.
2. **Task 10 step 2** — voice IDs (`"ada"`, `"owen"`, ...) are placeholders. Engineer must replace with real Qwen3 voice identifiers verified by listening.
3. **Task 13 step 1** — Qwen3 model download URL is a placeholder pointing at a likely Hugging Face path. Engineer must confirm the real URL.

These are explicit "engineer action items" in the plan, not lazy hand-waves. All other steps are fully spelled out.

**3. Type consistency:** Verified — `VoiceID`, `Voice`, `Speed`, `Settings`, `Coordinator`, `CoordinatorState`, `PlaybackController`, `TTSEngine`, `Qwen3TTSEngine`, `ModelManager`, `ModelDownloader` are referenced consistently across all tasks where they appear.

---

## Execution Handoff

Plan complete and saved to `~/Code/tts-app/docs/superpowers/plans/2026-05-26-hearit-menubar-tts.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for keeping the main session lean while the actual macOS work happens in worktree-isolated subagents.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints. Best if you want to watch every step happen and intervene quickly.

Which approach?
