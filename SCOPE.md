# Bolo (working name) — Scope v0.1

> Working directory: `~/Code/tts-app`. Rename once the brand is decided (task #11).
> Status: scope finalized 2026-05-26. Ready to scaffold.

## What it is
A macOS menu bar app that reads selected text aloud in a natural AI voice. Press a global hotkey with text selected anywhere on macOS → the text is spoken in a high-quality AI voice running entirely on-device.

The "reverse Wispr Flow."

## What it is NOT (v1)
- Voice cloning
- Audio export / save-to-file
- Reading queue, library, or history
- iOS / iPad companion
- SSML markup, transcripts, browser extension
- Account system, sync, telemetry, analytics

## Hard constraints
| Constraint | Value |
|---|---|
| Network calls after install | **Zero** |
| Platform | Apple Silicon only (M1+) |
| macOS minimum | 15 Sequoia |
| App binary | < 20 MB |
| Idle RAM | < 50 MB (model unloaded when idle) |
| Engine runtime | Swift-native via MLX or CoreML — no Python sidecar |
| License of bundled model | MIT or Apache 2.0 only |

## Locked product decisions
- **Aesthetic:** native-stealth (NSPopover with vibrancy, SF Symbols, system font, light/dark auto)
- **Voice picker:** ~~curated 6–8 voices~~ → **no voice picker in v1.** Qwen3-TTS exposes language selection only, one voice per language (10 languages). Popover shows a language picker (English default). Voice variety comes in v1.1 via voice cloning. Updated 2026-05-26 after discovering the real Qwen3TTS API in speech-swift takes `language:` not `voice:`.
- **Distribution:** Setapp only (no direct download, no App Store v1)
- **First-run:** welcome → Accessibility permission → model download with progress bar → ready. ~30 sec to first read.
- **Engine:** Qwen3-TTS via [speech-swift](https://github.com/soniqo/speech-swift) (Apache 2.0, claimed highest quality in the toolkit, 10 languages, streaming). Kokoro available as "Fast mode" fallback.

## Engine landscape (verified 2026-05-26)
| Engine | Swift-native today? | License | Verdict |
|---|---|---|---|
| **Qwen3-TTS** | ✅ via speech-swift | Apache 2.0 | **Primary** |
| **Kokoro** | ✅ via speech-swift | Apache 2.0 | Fast-mode fallback |
| **Chatterbox** | ❌ Python-only (mlx-audio) | MIT | Future port if Qwen3 disappoints |
| **Sesame CSM-1B** | ⚠️ MLX-Python only | Apache 2.0 | Future port if Qwen3 disappoints |
| **Fish Speech S1** | ❌ No Swift port | Mixed | Skip |

## Architecture
```
┌─ App layer ────────── SwiftUI menu bar, NSStatusItem, NSPopover
├─ Hotkey layer ─────── HotKey Swift package (RegisterEventHotKey)
├─ Capture layer ────── AXUIElement (Accessibility API) + clipboard fallback
├─ Engine protocol ─── TTSEngine { synthesize(text) -> AudioStream }
│      ├─ Qwen3Engine (default, via speech-swift)
│      └─ KokoroEngine (Fast mode, via speech-swift)
├─ Playback layer ──── AVAudioEngine with pause/resume/stop
├─ Lifecycle ───────── ModelManager actor: lazy-load + 5min idle unload
└─ Persistence ──────── UserDefaults via @AppStorage
```

The `TTSEngine` protocol is the load-bearing abstraction — swapping engines later is one new class, not a rewrite.

## Build phases
1. **Foundation** — fresh Xcode app, NSStatusItem, blank popover, entitlements (Accessibility, Hardened Runtime)
2. **Hotkey + capture** — ⌘⇧R works globally, captured text logs to console, Accessibility permission flow
3. **TTS pipeline (real)** — speech-swift + Qwen3-TTS speaks a hardcoded string
4. **Glue** — captured text flows into Qwen3 → AVAudioEngine playback with pause/stop
5. **Native-stealth UI** — popover (text preview, transport controls, curated voice picker, speed slider), settings sheet (hotkey field, "show all voices" toggle, launch-at-login)
6. **Lifecycle** — lazy-load on first use, 5-min idle unload, launch-at-login
7. **Onboarding** — welcome → Accessibility deep-link → model download progress → ready
8. **Distribution** — Developer ID signing, notarization, DMG, Setapp marketing pack

## Deferred to v1.1+
- Voice cloning from user-supplied sample
- Reading queue (queue multiple selections)
- Per-app voice profiles
- Global pause/resume hotkey (separate from primary)
- Speed presets (currently continuous slider only)
- Pronunciation overrides / dictionary
- Chatterbox or Sesame engine swap

## Open items
- **Name** (task #11). Working name "Bolo" — final name deferred until product is tangible.
- **Domain + socials** registered after name is locked.
- Pricing within Setapp (Setapp sets revenue share; nothing to decide unilaterally).
