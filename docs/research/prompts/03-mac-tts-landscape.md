# Deep Research Prompt: macOS Text-to-Speech Reader Landscape (2026)

You are mapping the full competitive landscape. Output should be a long-form markdown document with sources cited inline.

## Context

I'm building Bolo — a macOS menu bar AI TTS app that reads selected text aloud, fully on-device. I want a complete picture of every meaningful Mac-compatible competitor, including indie/open-source ones, so I understand what features are standard, what features are differentiating, and where the actual gaps are.

## What to investigate

### 1. Every macOS-compatible TTS reader app

Cover every category exhaustively:

**Major commercial**:
- Speechify (covered separately in prompt #2, brief mention only here)
- NaturalReader
- ElevenReader (ElevenLabs)
- Voice Dream Reader
- Voice Aloud Reader
- Read Aloud (the Chrome extension that has Mac users)
- Murf, WellSaid Labs, Play.ht — primarily B2B but check for consumer Mac apps

**Apple's built-in**:
- macOS "Speak Selection" + Siri voices + Premium voices
- VoiceOver (the screen reader) as TTS source
- The Read Aloud feature in Safari

**Setapp catalog**: Any TTS or audio-reading apps currently on Setapp? Be specific about which ones, their pricing, their differentiators.

**Open source / indie**:
- [Clicky](https://www.scriptbyai.com/ai-voice-companion-clicky/) — MIT Swift menu bar AI companion with TTS
- SuperCmd — launcher with read-aloud feature
- Subvocal — Kokoro-based, archived but a reference data point
- Trace (traceapp.info) — opposite direction (transcription) but similar UX pattern
- Any other indie macOS TTS apps on GitHub, Indie Hackers, Show HN history?

**Adjacent / partial-overlap**:
- Audiobook generators (Pocketsmith, LibriVox tools)
- Read-it-later apps with audio (Pocket TTS, Instapaper, Matter)
- Podcast/audio apps that accept pasted text
- Accessibility-focused tools (Speechify for accessibility, Voice Dream, NaturalReader pro)
- AI assistants with TTS as a side feature (ChatGPT Mac app, Claude desktop, Raycast AI)

### 2. For each app, document

In a single table per app:

- Name + link
- Mac support level (full native, Catalyst, Electron, web wrapper, browser extension only)
- Trigger mechanism (menu bar icon, hotkey, dock app, browser extension, contextual menu)
- TTS engine (cloud / on-device / which provider)
- Voice library size, language count
- Pricing (free, freemium, paid one-time, subscription tiers)
- Standout feature
- Top user complaint
- Last updated / actively maintained?

### 3. The standard feature set

After cataloging every app, identify: what features does EVERY meaningful TTS reader have? This is the baseline Bolo must hit.

### 4. The "rare and differentiating" features

What features do only the BEST or most distinctive apps have? Things like:
- Voice cloning
- Multi-voice dialogue rendering
- Smart pronunciation (names, acronyms, foreign words)
- Audio export
- Reading queues / history
- Cross-device sync
- Real-time tone adjustment
- Hands-free voice control

### 5. The unfilled gaps — where Bolo could uniquely win

Where are the gaps no one is currently filling? Some hypotheses to test against your research:

- "Fully on-device" — only a few players do this seriously
- "Apple-Silicon-native and fast" — many apps feel sluggish on Mac
- "Mac-feeling design" — most TTS apps look like cross-platform web wrappers
- "No subscription / one-time purchase / bundled in Setapp" — most are subscription
- "Privacy-first" — most send your selections to the cloud
- "Voice quality matching ElevenLabs but local" — currently nobody does this on Mac

Validate or reject each.

### 6. Pricing norms

For this category specifically:
- Average subscription price/month
- Average one-time price (if any)
- Free tier patterns — what's typically free vs paid?
- What does Setapp pay developers in similar categories?

### 7. Distribution channels

Where do these apps come from?
- Mac App Store
- Setapp
- Direct download (developer site)
- Browser extension marketplace
- Combination

What's the standard distribution pattern? Should Bolo be Setapp-only, Setapp + direct, MAS-also?

### 8. Positioning against each top-5 competitor

For each of the top 5 most-likely Bolo competitors, write a one-sentence positioning angle:

- "Bolo vs Speechify: ____"
- "Bolo vs NaturalReader: ____"
- "Bolo vs ElevenReader: ____"
- "Bolo vs Voice Dream: ____"
- "Bolo vs macOS built-in: ____"

## Output format

Long-form markdown:

```
# Mac TTS Reader Landscape 2026

## TL;DR
(5 bullets)

## App catalog
(one section per app or category, with tables)

## Standard feature set
(the baseline)

## Differentiating features
(what only the best have)

## Unfilled gaps
(where Bolo could win)

## Pricing norms
(numbers + table)

## Distribution channels
(strategy guidance)

## Positioning matrix
(Bolo vs each top-5 competitor)

## Sources
(all URLs)
```
