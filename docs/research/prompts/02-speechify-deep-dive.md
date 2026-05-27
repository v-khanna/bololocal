# Deep Research Prompt: Speechify Competitive Intelligence

You are doing competitive intelligence research. Output should be a long-form markdown document with sources cited inline.

## Context

I'm building Bolo — a macOS menu bar AI text-to-speech reader that runs fully on-device. Speechify is the dominant commercial competitor in this category. Before I launch, I want to deeply understand what they offer, where they're strong, where they're weak, and how to position against them.

## What to investigate

### 1. Product surface — the user-facing experience

- **macOS app**: Walk through the actual UX. What does the app look like? How do you trigger it (menu bar, dock app, browser extension, all of the above)?
- **Trigger pattern**: Can a user select text in Safari, Mail, a PDF, an arbitrary app, and have Speechify read it? What's the hotkey? Does it work cross-app?
- **Voice library**: How many voices? Which languages? Any "celebrity voices" (they had Snoop Dogg, Gwyneth Paltrow, etc. — what's the current lineup)?
- **Voice cloning**: Can users clone their own voice? What's the process and quality?
- **Reading features**: Speed control, pause/resume, skip, save-as-audio, queue, playlists?
- **iOS / iPad / Android / web parity**: How does the Mac experience compare?
- **Browser extension**: Chrome, Safari, Firefox? How does it work?
- **Recent feature releases** (last 6-12 months): what's new?

### 2. Pricing tiers

- Free tier — what's included, what's gated?
- Premium tier — pricing, features
- Pro tier — pricing, features
- Studio / Enterprise — pricing, features
- Annual vs monthly pricing
- Student / educational discounts
- Any one-time purchase option? (Doubt it but check.)
- How aggressive is the upsell pattern? (App Store / Reddit complaints about this)

### 3. Technology under the hood

- What TTS engines does Speechify actually use? Their own model? OpenAI? ElevenLabs API? Resemble? Microsoft? Google? (They've been opaque about this historically — check job listings, engineering blog posts, anything technical.)
- Is any synthesis on-device, or 100% cloud?
- Audio quality bar — how does it compare to ElevenLabs v3, OpenAI tts-1-hd, Cartesia Sonic?
- Latency — how fast does it start reading after you trigger?
- What model size class are their voices? (Hint: cloud means they can use huge models.)
- Any privacy posture statements about transcripts, audio, what they store?

### 4. Business and scale

- Approximate user base (DAU, MAU, paid subscribers — anything they've publicly disclosed)
- Most recent funding round, valuation, total raised
- Revenue if reported
- Founding team and where they came from
- Marketing positioning — what do they say about themselves? Who's the target?

### 5. User sentiment — where they're loved and hated

Scrape App Store reviews, Reddit r/Speechify and adjacent subs, Twitter/X complaints, Trustpilot, Quora, Hacker News threads. Document:

- Top 5 things users LOVE
- Top 5 things users COMPLAIN about
- Specific quotes (with sources) for the most common complaints
- Recurring themes (subscription frustration? voice quality? bugs? privacy?)

### 6. Privacy concerns

- What does Speechify do with the text you select?
- Is selected text sent to their servers? Stored? Logged?
- Audio output — generated server-side and streamed back? Cached?
- GDPR / privacy policy posture
- Has there been any data incident or controversy?

### 7. Their weaknesses — where Bolo could win

Synthesize: given everything above, what specific positioning angles would resonate with users who are currently on Speechify but unhappy? Examples to consider:

- Privacy ("fully on your Mac, nothing leaves")
- Pricing (Setapp bundled vs Speechify's subscription)
- Voice quality (if Chatterbox/Sesame/etc. truly beats their cloud)
- macOS-native feel (vs Speechify's cross-platform-but-feels-it design)
- Speed / latency
- Reliability when offline

### 8. The "moat" question

What stops a smaller competitor from taking share? Is Speechify's moat:

- Brand recognition?
- Voice library partnerships?
- Subscription lock-in?
- Distribution (paid acquisition spend, browser extension installs)?
- Cross-platform parity?

What's the realistic share a privacy-first, Mac-only, on-device, Setapp-distributed competitor could take?

### 9. Setapp presence

Is Speechify on Setapp? If yes, what's the integration / pricing there? If no, why might Setapp have wanted Bolo specifically?

## Output format

Long-form markdown. Structure:

```
# Speechify Competitive Brief

## TL;DR
(5 bullets)

## Product
(detailed UX walkthrough)

## Pricing
(table of tiers + features)

## Technology
(what we know about the stack)

## Business
(scale, funding, positioning)

## User sentiment
(loves + hates with quotes)

## Privacy posture
(what's their stance)

## Where Bolo can win
(3-5 specific positioning angles)

## Sources
(all URLs cited)
```

End with: "If I were Bolo's marketing copywriter, the single sentence I'd use to position against Speechify is: ____."
