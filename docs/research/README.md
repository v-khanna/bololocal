# Bolo — Deep Research

Prompts to run in your favorite deep-research tool (ChatGPT Deep Research, Perplexity Pro, Gemini Deep Research, Claude with web search, etc.). Each prompt is self-contained — you can paste it as-is and the tool won't need any other context.

## Workflow

1. Open a prompt in `prompts/`
2. Copy its contents into your deep research tool
3. Let it run (these can take 5–30 minutes for thorough output)
4. Save the resulting markdown into `results/` with the same filename
   - e.g. prompt `prompts/01-chatterbox-port-feasibility.md` → result `results/01-chatterbox-port-feasibility.md`
5. Next session, point me at `~/Code/bolo/docs/research/results/` and I'll read everything and update plans accordingly

## The five prompts

| # | File | What it answers | Priority |
|---|---|---|---|
| 01 | [chatterbox-port-feasibility](prompts/01-chatterbox-port-feasibility.md) | Has anyone already ported Chatterbox to MLX/Swift? What's the realistic effort to do it ourselves? | **P0 — gates the v2 engineering bet** |
| 02 | [speechify-deep-dive](prompts/02-speechify-deep-dive.md) | The dominant commercial competitor — how do we position against them? | P1 |
| 03 | [mac-tts-landscape](prompts/03-mac-tts-landscape.md) | Full competitive map. What does every other Mac TTS reader do? Where are the gaps? | P1 |
| 04 | [tts-model-soa-2026](prompts/04-tts-model-soa-2026.md) | What's the best local-deployable TTS model in 2026? Is Chatterbox still the right pick? | P1 |
| 05 | [setapp-positioning](prompts/05-setapp-positioning.md) | Is Setapp the right primary channel? What gets approved/featured? | P2 |

## Run order recommendation

**If you only run one**: run #1 (chatterbox-port-feasibility). It's the gate on the entire v2 engineering bet — if someone already ported it, we save a week of work. If MLX-Swift can't handle the architecture, we need a different engine.

**If you run two**: add #4 (tts-model-soa-2026) to confirm Chatterbox is actually the right model to port. Maybe something better has appeared in the last month.

**For the full picture**: all five. They take maybe 2–4 hours of total tool time (you can run multiple in parallel in different tabs).

## Format request

When pasting results back, just save them as raw markdown — no need to clean them up. I'll handle synthesis.

If the deep research tool gives you a PDF or web export, save as markdown if possible. Plain `.md` files are easiest for me to read across sessions.
