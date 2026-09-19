# Tring launch posts

Drafts, not committed to git history intent: post from personal accounts.
Link to use everywhere: https://github.com/anzal1/tring

---

## LinkedIn post

Two seconds of silence kills a phone call faster than any wrong answer.

I learned that running voice agents across hundreds of thousands of production telephony calls, in multiple languages, with callers who code-switch mid-sentence. The hard problems were never in the model. They live in the 300 milliseconds after someone stops talking.

Today I'm open-sourcing everything those calls taught me.

**Tring** (the sound of an arriving call) is a production-grade voice agent stack, MIT licensed:

🔔 One agent contract, three runtimes. Define an agent once in YAML and run it as a cascade (STT to LLM to TTS), as speech-to-speech, or as a hybrid. Switching architectures is a config change, not a migration.

🔔 Local-first, for real. faster-whisper, Ollama, and Kokoro give you a $0-per-minute pipeline with production turn-taking, not a demo. 18 providers ship in v0.2: AssemblyAI, Deepgram, OpenAI, Anthropic, Gemini, ElevenLabs, Cartesia, Sarvam, Ultravox, OpenAI Realtime, and more.

🔔 Dead air is structurally impossible. Every tool call's schema requires the model to declare what to say while the tool runs. A character-level parser streams speech to TTS while the tool call is still being generated. A playback ledger makes sure the model never references words a caller interrupted and never heard.

🔔 Honest costs. Every metered unit carries an estimated flag, cached tokens are tracked separately, and reports roll up to cost per outcome, not cost per minute. Optimizing the rate card is a rounding error next to optimizing the denominator.

pip install tring

Repo in the comments. If you're building voice agents, I'd genuinely love your issues, PRs, and war stories. Which provider should be next?

#VoiceAI #OpenSource #ConversationalAI #LLM #Python

(First comment: 🔗 https://github.com/anzal1/tring)

---

## X / Twitter thread

**1/**
I open-sourced the voice agent stack I wish existed a year ago.

tring: one agent contract, any runtime (cascade / speech-to-speech / hybrid), local-first, honest cost accounting.

pip install tring
https://github.com/anzal1/tring

🧵 the lessons it's built from:

**2/**
In a text agent, latency is an annoyance.

In a voice agent, latency is a conversational ERROR. Two seconds of silence and the caller assumes the line dropped, talks over the bot, and the turn is lost.

The interesting problems aren't in the model. They're in the 300ms after someone stops talking.

**3/**
Dead air during tool calls: don't patch it, make it impossible.

In tring, every tool call's schema REQUIRES three fields: waiting_message, spoken_mode, post_tool_response.

The model literally cannot call a tool without planning the silence around it.

**4/**
Waiting for complete JSON before speaking wastes an entire generation of latency.

tring's parser is a character-level state machine: it streams the "speak" field to TTS while the tool_call is still being generated. Handles escapes split across chunks. Falls back to plain speech instead of going silent.

**5/**
TTS generates faster than audio plays. So when a caller interrupts, your context contains words they never heard, and the model will confidently reference them.

tring keeps a playback ledger: heard vs unheard text, split at a word boundary, annotated only on genuine barge-ins.

**6/**
Multilingual models drift to English mid-conversation. The obvious fix (re-inject the language directive each turn) silently destroys your prompt cache.

tring folds the directive into one trailing message so the array stays byte-identical. There's a test that proves prefix stability.

**7/**
Cost: everyone optimizes the rate. Almost nobody measures the denominator.

tring meters every unit with an honest estimated flag and rolls costs up per dial, per connected call, per conversation, per OUTCOME. That last number is the only one your CFO cares about.

**8/**
It's local-first: faster-whisper + Ollama + Kokoro = $0/min with production turn-taking. 18 providers total (Anthropic, Gemini, Deepgram, AssemblyAI, ElevenLabs, Cartesia, Sarvam, Ultravox, OpenAI Realtime...).

MIT licensed. 196 tests. Stars and PRs very welcome 🔔
https://github.com/anzal1/tring

---

## Posting notes

- LinkedIn: put the repo link in the first comment, not the post body (LinkedIn suppresses external links in posts). Post Tuesday to Thursday morning your audience's time.
- X: pin the thread after posting. Reply to early comments fast; first-hour engagement decides reach.
- Both: attach the social card image (docs/assets/social-card.png) or a 20-second terminal capture of the console quickstart. Motion outperforms static.
- Space them: X thread day one, LinkedIn day two, so each gets its own engagement window.
