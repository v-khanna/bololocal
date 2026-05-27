# Bolo

> *Bolo* (बोलो) — Hindi imperative: **"speak."**

A macOS menu bar app that reads selected text aloud in a natural AI voice
running entirely on your Mac. Select text in any app, press ⌘⇧R, listen.

- **Fully on-device** — Qwen3-TTS via [speech-swift](https://github.com/soniqo/speech-swift)
  on the Apple Neural Engine. Exactly one network request, ever (the
  first-run model download).
- **Lightweight** — under 50 MB RAM when idle; the model unloads after
  5 minutes of inactivity.
- **Native-stealth** — invisible until you need it. Menu bar icon, no
  Dock presence, system-native vibrancy throughout.
- **Apple Silicon only**, macOS 15+.

## Documentation

| Doc | What's in it |
| --- | --- |
| [`SCOPE.md`](SCOPE.md) | Product scope and locked decisions |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **System design — high and low level.** Read this to understand how the app works end-to-end. |
| [`docs/DEVELOPING.md`](docs/DEVELOPING.md) | Practical build/test/debug guide |
| [`docs/RELEASE.md`](docs/RELEASE.md) | Signing, notarization, DMG packaging |
| [`docs/SETAPP.md`](docs/SETAPP.md) | Setapp submission metadata + screenshot checklist |
| [`PRIVACY.md`](PRIVACY.md) | Privacy policy (zero telemetry, zero accounts, zero post-install network calls) |
| [`docs/superpowers/plans/2026-05-26-bolo-menubar-tts.md`](docs/superpowers/plans/2026-05-26-bolo-menubar-tts.md) | The 15-task implementation plan, including the mid-execution Qwen3 API pivot |

## Quick start

```bash
brew install xcodegen
git clone <repo> ~/Code/bolo
cd ~/Code/bolo
xcodegen generate
xcodebuild -scheme Bolo -destination 'platform=macOS,arch=arm64' build
open build/Build/Products/Debug/Bolo.app
```

On first launch, walk through the onboarding flow (welcome → Accessibility
permission → ~500 MB model download → ready). After that, ⌘⇧R is instant.

For unsigned local builds, pass `CODE_SIGNING_REQUIRED=NO
CODE_SIGNING_ALLOWED=NO` to `xcodebuild`. For signed Developer ID builds,
fill in `DEVELOPMENT_TEAM` in `project.yml` (see [RELEASE.md](docs/RELEASE.md)).

## Status

Feature-complete v1. 17 commits, 23 passing unit tests, ready for
signing → notarization → Setapp submission. See `docs/RELEASE.md` for
the remaining manual steps.

## License

TBD. Source uses [HotKey](https://github.com/soffes/HotKey) (MIT) and
[soniqo/speech-swift](https://github.com/soniqo/speech-swift) (license
per their repo) as SPM dependencies.
