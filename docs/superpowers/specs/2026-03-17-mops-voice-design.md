# mops-voice Design Spec

**Date:** 2026-03-17
**Status:** Approved

## Overview

mops-voice is a voice-controlled CLI assistant named MOPS that controls digital fabrication machines through the MOPS MCP server. Inspired by TARS from Interstellar, it features configurable personality dials (humor, sarcasm, honesty) and speaks with a cloned voice using F5-TTS.

The user (Fran) talks to MOPS via push-to-talk, MOPS transcribes speech, reasons via Claude API with MOPS MCP tools, and responds with synthesized speech.

## Architecture

Monolith Python process. Main loop uses `asyncio`. Keyboard listener runs in a daemon thread via `pynput`. Audio recording uses blocking `sounddevice` in a thread, yielding to asyncio via `run_in_executor`. MCP client and Claude API use their async interfaces.

```
┌─────────────────────────────────────────────────┐
│                  mops-voice CLI                  │
│                                                  │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐  │
│  │  Audio In  │──▶│  whisper   │──▶│  Claude   │  │
│  │(sounddev.) │   │  .cpp     │   │  API      │  │
│  └───────────┘   └───────────┘   └─────┬─────┘  │
│                                        │         │
│                                   MCP Client     │
│                                   (stdio)        │
│                                        │         │
│  ┌───────────┐   ┌───────────┐   ┌─────▼─────┐  │
│  │  Speaker   │◀──│  F5-TTS   │◀──│ Response  │  │
│  │(sounddev.) │   │  (MLX)    │   │ + Tools   │  │
│  └───────────┘   └───────────┘   └───────────┘  │
│                                        │         │
│                                   ┌────▼────┐    │
│                                   │  MOPS   │    │
│                                   │  MCP    │    │
│                                   │ Server  │    │
│                                   └─────────┘    │
└─────────────────────────────────────────────────┘
```

### Main Loop

1. Wait for spacebar press
2. Record audio until spacebar release
3. Transcribe with whisper.cpp (CoreML, English only)
4. Validate transcription (skip if empty or < 3 characters)
5. Send to Claude API with conversation history + MOPS MCP tools (if available)
6. Execute any tool calls via MCP, feed results back until text-only response
7. Synthesize response text with F5-TTS (MOPS/TARS cloned voice)
8. Play audio through speakers
9. Loop back to 1

## Audio Input & Speech-to-Text

- **Recording:** `sounddevice`, 16kHz mono 16-bit PCM
- **Push-to-talk:** spacebar down = recording, spacebar up = stop. Key detection via `pynput` in a daemon thread
- **Transcription:** `pywhispercpp` with CoreML acceleration
- **Model:** `whisper-base.en` (English-only, ~140MB) as default. Configurable to `small.en` (~461MB) for higher accuracy
- **No VAD needed** — push-to-talk handles segmentation
- **Gibberish filtering:** Skip Claude call if transcript is empty or < 3 characters. Configurable blocklist for known whisper hallucinations ("Thank you for watching", etc.)

## LLM & MCP Integration

### Claude API

- `anthropic` async Python SDK with tool use
- Model: `claude-sonnet-4-20250514` (swappable via config)
- Conversation history maintained in memory, up to 50 message pairs. When limit is reached, oldest messages are pruned, preserving the system prompt and last 10 exchanges

### MCP Client

- `mcp` Python SDK in client mode, **stdio transport**
- On startup: spawn MOPS server as a child process (`node <mops_server_path>`) with stdin/stdout piped for MCP protocol
- Server path configurable via `config.json` `mops_server_path` field (default: `../mops/src/server.js` relative to mops-voice repo)
- CLI flags (`--mods-url`, `--headless`) forwarded to the subprocess as arguments
- Discover tools at startup, convert MCP tool schemas to Claude API tool format
- Tool call loop: Claude may chain multiple calls. Loop continues until text-only response
- Max 10 tool call iterations per turn (safety limit)
- **Subprocess lifecycle:** On `Ctrl+C` or exit, send SIGTERM to the child process. If subprocess crashes mid-session, log warning, switch to conversation-only mode

### MCP Failure Degradation

If the MCP server fails to start or crashes mid-session:
- Do NOT include MOPS tools in Claude API calls
- Only personality tools remain available
- MOPS tells the user: "I can't reach the machines right now, Fran. I'm in conversation-only mode."

### MOPS System Prompt

```
You are MOPS, a voice assistant for digital fabrication.
You address the user as Fran.
Personality settings: humor={humor}%, sarcasm={sarcasm}%, honesty={honesty}%.
Adjust your tone accordingly. Keep responses concise — they'll be spoken aloud.
You control fabrication machines through MOPS tools.
When a personality setting is changed, confirm it briefly in-character.
When asked about your settings, report them.
```

### Personality Tools

Two tools exposed to Claude alongside MOPS tools:

- `adjust_personality(dial, value)` — update a personality dial, persist to config
  - Valid dials: `humor`, `sarcasm`, `honesty`
  - Valid range: 0-100 (integer)
  - Invalid dial name or out-of-range value returns an error message to Claude
- `get_personality()` — return current dial values

Claude handles natural language variations ("be funnier", "set humor to 75%", "what's your sarcasm level?") and confirms changes in-character.

## Text-to-Speech & Voice Cloning

- **Engine:** F5-TTS via `f5-tts-mlx` (Apple Silicon optimized)
- **Voice cloning:** reference audio clip (5-15s of TARS voice) + reference transcript
  - Audio: `~/.mops-voice/tars_reference.wav` (WAV, 24kHz, mono, 16-bit PCM)
  - Transcript: `~/.mops-voice/tars_reference.txt` (exact text spoken in the reference clip)
- **On startup:** model + reference audio loaded into memory, stays resident
- **Playback:** `sounddevice`, blocks until complete
- **Estimated latency:** ~2-4s for ~20 words on M1 Pro

## Personality & Identity

- **Assistant name:** MOPS
- **User name:** Fran
- **Personality:** TARS-inspired — dry wit, deadpan, adjustable dials
- **Dials:** humor (default 75), sarcasm (default 50), honesty (default 90)
- **Persistence:** `~/.mops-voice/config.json` — always merged with defaults on load (missing keys fall back to defaults, so new config fields are forward-compatible)

```json
{
  "assistant_name": "MOPS",
  "user_name": "Fran",
  "personality": {
    "humor": 75,
    "sarcasm": 50,
    "honesty": 90
  },
  "mops_server_path": "../mops/src/server.js",
  "whisper_model": "base.en",
  "claude_model": "claude-sonnet-4-20250514"
}
```

## CLI Arguments

```bash
python -m mops_voice                          # defaults
python -m mops_voice --mods-url <url>         # forwarded to MOPS MCP server
python -m mops_voice --headless               # forwarded to MOPS MCP server (default)
python -m mops_voice --headed                 # show browser window (debug)
python -m mops_voice --whisper-model small.en # override whisper model
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mods-url` | (MOPS server default) | Mods CE deployment URL, forwarded to MOPS subprocess |
| `--headless` | on | Run Playwright browser headless, forwarded to MOPS subprocess |
| `--headed` | off | Show browser window for debugging |
| `--whisper-model` | `base.en` | Whisper model name |

## Verbose Terminal Output

Every pipeline stage prints in real time:

```
🎤 Recording... (spacebar held)
🎤 Recorded 2.3s audio
📝 Transcribing...
📝 "Mill the PCB traces for this board"
🤖 Calling Claude (sonnet)...
🔧 Tool call: get_server_status → OK
🔧 Tool call: load_program → "mill 2D PCB"
🔧 Tool call: trigger_action → "calculate"
🤖 Response: "I've loaded the PCB milling program and started calculating the traces."
🔊 Synthesizing speech (F5-TTS)...
🔊 Playing audio (3.1s)
⏱️  Total: 7.4s
```

Uses `rich` library for colored output and live-updating spinners.

## Project Structure

```
mops-voice/
├── mops_voice/
│   ├── __init__.py
│   ├── main.py              # Entry point, main loop, terminal output
│   ├── audio.py             # Mic recording, playback (sounddevice)
│   ├── transcribe.py        # whisper.cpp integration
│   ├── llm.py               # Claude API + MCP client + tool loop
│   ├── tts.py               # F5-TTS voice synthesis
│   ├── personality.py       # Dial management, config persistence
│   └── config.py            # Config loading, paths, defaults
├── pyproject.toml
├── README.md
└── .gitignore
```

### Dependencies

- `anthropic` — Claude API (async)
- `mcp` — MCP Python SDK (client mode, stdio transport)
- `pywhispercpp` — whisper.cpp Python bindings
- `f5-tts-mlx` — F5-TTS for Apple Silicon
- `sounddevice` — audio I/O (easy macOS install, numpy integration)
- `pynput` — keyboard listener for push-to-talk
- `rich` — pretty terminal output

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No mic detected | Error message at startup, exit |
| Empty recording (fast tap) | Skip, print "No audio captured" |
| Transcript empty or < 3 chars | Skip Claude call, print "Didn't catch that, Fran" |
| Known whisper hallucination | Skip Claude call (configurable blocklist) |
| `ANTHROPIC_API_KEY` missing | Error at startup with instructions |
| API rate limit / network error | MOPS says "I'm having trouble reaching my brain, Fran. Try again." |
| Tool call loop > 10 iterations | Break, respond with what we have |
| MOPS MCP server fails to start | Warn, switch to conversation-only mode (no MOPS tools sent to Claude) |
| MOPS MCP subprocess crashes | Log warning, switch to conversation-only mode |
| Tool call fails | Claude gets error, explains to Fran naturally |
| Reference audio missing | Error at startup with placement instructions |
| Reference transcript missing | Error at startup with placement instructions |
| F5-TTS fails | Fall back to text-only output, warn user |
| `Ctrl+C` or "goodbye"/"exit" | MOPS says farewell, SIGTERM child processes, exit |
| Invalid personality dial/value | Tool returns error, Claude communicates naturally |

## Memory Budget

Estimated memory usage on M1 Pro 16GB unified memory:

| Component | Estimated RAM |
|-----------|--------------|
| whisper-base.en model | ~140 MB |
| F5-TTS MLX model | ~500 MB - 1 GB |
| Chromium (Playwright, headless) | ~300-500 MB |
| Python runtime + deps | ~200 MB |
| macOS system overhead | ~4-6 GB |
| **Total estimated** | **~5-8 GB** |

Leaves ~8-11 GB headroom. If memory becomes tight:
- Swap `whisper-base.en` (140MB) instead of `small.en` (461MB) — this is already the default
- Consider loading/unloading whisper and TTS models on demand rather than keeping both resident (adds latency but saves ~500MB)

## Hardware & Constraints

- **Target:** Apple M1 Pro, 16GB unified memory
- **All local inference:** whisper.cpp (CoreML) + F5-TTS (MLX)
- **Remote:** Claude API only (requires internet + API key)
- **Estimated round-trip:** 5-10s depending on response length and tool calls
- **Language:** English only
