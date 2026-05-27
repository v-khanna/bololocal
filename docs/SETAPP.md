# Setapp Submission — Bolo

## App info

| Field | Value |
|---|---|
| App name | **Bolo** (working name — may change before submission) |
| Bundle ID | `com.virkhanna.bolo` |
| Category | Productivity (alt: Utilities) |
| Subcategory | Reading / Accessibility |
| Tagline | Hear any text aloud, fully on-device. |
| Pricing | Setapp handles billing; no in-app purchases |
| Platform | macOS 15+ Apple Silicon |

## Long description (draft)

Bolo reads any text you select aloud, in a natural AI voice that runs entirely on your Mac. Select a paragraph in Safari, an email in Mail, or a passage in a PDF — press ⌘⇧R and listen. No cloud, no accounts, no analytics.

The voice model downloads once on first launch (~500 MB) and never connects to the internet again. Built for Apple Silicon Macs running macOS 15 or later.

**Features**
- Press ⌘⇧R to read selected text in any app
- Natural AI voice powered by Qwen3-TTS, running on the Apple Neural Engine
- Lives in your menu bar — invisible until you need it
- Adjustable playback speed (0.5x–2.0x)
- Fully on-device, fully private — zero telemetry

## Screenshot checklist

Setapp requires the following screenshots (minimum 1280×800):

- [ ] Menu bar icon + open popover with sample text playing
- [ ] Settings window — General tab
- [ ] Settings window — About tab
- [ ] Onboarding screen — welcome
- [ ] Hero shot — text being read in Safari with Bolo popover visible

## App icon

Final icon is a TODO before submission. The asset catalog at `Bolo/Assets.xcassets/AppIcon.appiconset/` has the required slots set up — drop a 1024×1024 master PNG in and Xcode will downsample.

## Privacy

See [../PRIVACY.md](../PRIVACY.md).

## Distribution

- Setapp only for v1 (no direct download, no Mac App Store)
- Future: consider direct paid download at `bolo.app` for non-Setapp users
