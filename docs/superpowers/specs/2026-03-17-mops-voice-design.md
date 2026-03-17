# mops-voice Design Spec

**Date:** 2026-03-17
**Status:** Approved

## Overview

mops-voice is a voice-controlled CLI assistant named MOPS that controls digital fabrication machines through the MOPS MCP server. Inspired by TARS from Interstellar, it features configurable personality dials (humor, sarcasm, honesty) and speaks with a cloned voice using F5-TTS.

The user (Fran) talks to MOPS via push-to-talk, MOPS transcribes speech, reasons via Claude API with MOPS MCP tools, and responds with synthesized speech.

## Architecture

Monolith Python process. Single main loop, all stages in-process.

```
┌─────────────────────────────────────────────────┐
│                  mops-voice CLI                  │
│                                                  │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐  │
│  │  Audio In  │──▶│  whisper   │──▶│  Claude   │  │
│  │ (PyAudio)  │   │  .cpp     │   │  API      │  │
│  └───────────┘   └───────────┘   └─────┬─────┘  │
│                                        │         │
│                                   MCP Client     │
│                                        │         │
│  ┌───────────┐   ┌───────────┐   ┌─────▼─────┐  │
│  │  Speaker   │◀──│  F5-TTS   │◀──│ Response  │  │
│  │ (PyAudio)  │   │  (MLX)    │   │ + Tools   │  │
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
4. Send to Claude API with conversation history + MOPS MCP tools
5. Execute any tool calls via MCP, feed results back until text-only response
6. Synthesize response text with F5-TTS (MOPS/TARS cloned voice)
7. Play audio through speakers
8. Loop back to 1

## Audio Input & Speech-to-Text

- **Recording:** PyAudio, 16kHz mono 16-bit PCM
- **Push-to-talk:** spacebar down = recording, spacebar up = stop. Key detection via `pynput`
- **Transcription:** `pywhispercpp` with CoreML acceleration
- **Model:** `whisper-small.en` (English-only, ~461MB). Can swap to `base.en` if speed preferred over accuracy
- **No VAD needed** — push-to-talk handles segmentation

## LLM & MCP Integration

### Claude API

- `anthropic` Python SDK with tool use
- Model: `claude-sonnet-4-20250514` (swappable)
- Conversation history maintained in memory for session duration

### MCP Client

- `mcp` Python SDK in client mode
- On startup: launch `node ../mops/src/server.js` as MCP subprocess
- Discover tools at startup, convert MCP tool schemas to Claude API tool format
- Tool call loop: Claude may chain multiple calls. Loop continues until text-only response
- Max 10 tool call iterations per turn (safety limit)

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
- `get_personality()` — return current dial values

Claude handles natural language variations ("be funnier", "set humor to 75%", "what's your sarcasm level?") and confirms changes in-character.

## Text-to-Speech & Voice Cloning

- **Engine:** F5-TTS via `f5-tts-mlx` (Apple Silicon optimized)
- **Voice cloning:** reference audio clip (5-15s of TARS voice) stored at `~/.mops-voice/tars_reference.wav`
- **On startup:** model + reference audio loaded into memory, stays resident
- **Playback:** PyAudio or `sounddevice`, blocks until complete
- **Estimated latency:** ~2-4s for ~20 words on M1 Pro

## Personality & Identity

- **Assistant name:** MOPS
- **User name:** Fran
- **Personality:** TARS-inspired — dry wit, deadpan, adjustable dials
- **Dials:** humor (default 75), sarcasm (default 50), honesty (default 90)
- **Persistence:** `~/.mops-voice/config.json`

```json
{
  "assistant_name": "MOPS",
  "user_name": "Fran",
  "personality": {
    "humor": 75,
    "sarcasm": 50,
    "honesty": 90
  }
}
```

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
│   ├── audio.py             # Mic recording, playback
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

- `anthropic` — Claude API
- `mcp` — MCP Python SDK (client mode)
- `pywhispercpp` — whisper.cpp Python bindings
- `f5-tts-mlx` — F5-TTS for Apple Silicon
- `pyaudio` or `sounddevice` — audio I/O
- `pynput` — keyboard listener for push-to-talk
- `rich` — pretty terminal output

### Entry Point

```bash
python -m mops_voice
python -m mops_voice --mods-url https://mods.beachlab.org
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No mic detected | Error message at startup, exit |
| Empty recording (fast tap) | Skip, print "No audio captured" |
| Empty/gibberish transcription | Skip Claude call, print "Didn't catch that, Fran" |
| `ANTHROPIC_API_KEY` missing | Error at startup with instructions |
| API rate limit / network error | MOPS says "I'm having trouble reaching my brain, Fran. Try again." |
| Tool call loop > 10 iterations | Break, respond with what we have |
| MOPS MCP server fails to start | Warn, continue in voice-only mode (no machine control) |
| Tool call fails | Claude gets error, explains to Fran naturally |
| Reference audio missing | Error at startup with placement instructions |
| F5-TTS fails | Fall back to text-only output, warn user |
| `Ctrl+C` or "goodbye"/"exit" | MOPS says farewell, cleans up, exits |

## Hardware & Constraints

- **Target:** Apple M1 Pro, 16GB unified memory
- **All local inference:** whisper.cpp (CoreML) + F5-TTS (MLX)
- **Remote:** Claude API only (requires internet + API key)
- **Estimated round-trip:** 5-10s depending on response length and tool calls
- **Language:** English only
