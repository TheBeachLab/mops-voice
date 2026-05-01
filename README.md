# mops-voice

Voice-controlled CLI assistant for digital fabrication. Talk to MOPS, control your machines.

## Demo

<video src="https://github.com/TheBeachLab/mops-voice/releases/download/demo/mops.mp4" controls width="720"></video>

## Pipeline

```
🎤 Mic → whisper.cpp → Claude (CLI or API) + MOPS MCP tools → TTS engine → 🔊 Speaker
```

## Setup

```bash
# Install
uv sync

# Configure
mkdir -p ~/.mops-voice
```

### LLM engine setup

**Claude CLI (default)** — uses Claude Code CLI, included with Max/Team plan:

```bash
# Install: https://docs.anthropic.com/en/docs/claude-code
```

**Direct Anthropic API (faster)** — bypasses CLI startup overhead (~1-2s vs ~7s per turn):

```json
{
  "llm_engine": "api",
  "claude_model": "haiku",
  "anthropic": {
    "api_key": "sk-ant-..."
  }
}
```

Get an API key at [console.anthropic.com](https://console.anthropic.com). Pay-per-token (haiku is fractions of a cent per voice turn).

**Direct OpenAI API (alternative)** — same low-latency profile, different vendor. Useful if you have OpenAI credits, want to compare model behaviour against Claude, or hit Anthropic rate limits:

```json
{
  "llm_engine": "openai",
  "openai": {
    "api_key": "sk-...",
    "model": "gpt-5-mini"
  }
}
```

Get an API key at [platform.openai.com](https://platform.openai.com/api-keys). Default model `gpt-5-mini` is the latency/cost analog of `haiku` for voice turns; aliases `mini` / `nano` / `5` resolve to `gpt-5-mini` / `gpt-5-nano` / `gpt-5`. Prompt caching is automatic on the OpenAI side for any prefix ≥1024 tokens since 2024-10 — no explicit markers needed. On API errors the runtime falls back to the Claude CLI engine, mirroring the Anthropic-API path.

### TTS engine setup

**F5-TTS (local, default on Apple Silicon)** — voice cloning via MLX. Skip this section on Linux/Windows/Intel Mac: the dependency is auto-skipped at install time and `voxtral` is selected as the default instead.

```bash
# Place reference audio and transcript (F5-only — Voxtral users skip this)
~/.mops-voice/tars_reference.wav   # 24kHz mono WAV, 5-15s
~/.mops-voice/tars_reference.txt   # exact text spoken in the audio (must match the WAV verbatim)
```

The `.txt` file is **mandatory for F5-TTS** — F5 conditions generation on `(reference_audio + transcript)` together, and startup raises `FileNotFoundError` if it's missing. Voxtral does pure zero-shot cloning over the audio bytes alone (the Mistral API has no reference-transcript field), so on the Voxtral path the `.txt` file is unused.

**Voxtral (cloud)** — Mistral API, no local model needed:

Add your API key to `~/.mops-voice/config.json`:

```json
{
  "tts_engine": "voxtral",
  "voxtral": {
    "api_key": "your-mistral-api-key",
    "voice": "en_paul_confident"
  }
}
```

Available voices: `en_paul_neutral`, `en_paul_confident`, `en_paul_happy`, `en_paul_excited`, `en_paul_cheerful`, `en_paul_angry`, `en_paul_frustrated`, `en_paul_sad`, `gb_oliver_neutral`, `gb_jane_sarcasm`.

## Usage

From inside the repo:

```bash
uv run mops-voice
```

Or install globally once, then call from anywhere:

```bash
uv tool install /path/to/mops-voice
mops-voice
```

Hold **spacebar** to talk. **Q** or **Ctrl+C** to exit.

**Bluetooth clicker (optional)** — a presentation remote like the Kensington 33062 can drive PTT from across the room. Defaults: **Page Down** toggles recording on/off, the lower **B** button cancels the current operation. Configure under `clicker` in `~/.mops-voice/config.json` (`enabled`, `trigger_key`, `cancel_key`, `mode: toggle|hold`). To map an unknown remote's buttons, run `uv run python scripts/clicker_test.py` and press each button to see the key name pynput reports.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-engine` | cli | LLM engine: `cli` (Claude Code CLI), `api` (direct Anthropic API), or `openai` (direct OpenAI API) |
| `--tts-engine` | f5 on Apple Silicon, else voxtral | TTS engine: `f5` (local MLX) or `voxtral` (Mistral API) |
| `--whisper-model NAME` | base.en | Whisper model |
| `--user NAME` | Fran | User name shown in greetings |
| `--mods-url URL` | modsproject.org | Mods CE deployment |
| `--headless` | off | Hide browser window |

Flags that express a preference (`--llm-engine`, `--tts-engine`, `--whisper-model`, `--user`) get saved to `~/.mops-voice/config.json` on use, so the next run defaults to whatever you last picked:

```bash
mops-voice --llm-engine api --tts-engine voxtral   # sets the defaults
mops-voice                                          # picks them up, no flags needed
```

### Voice-controllable settings

MOPS accepts natural-language config changes mid-session. Settings persist to `~/.mops-voice/config.json`.

- **Personality dials** (0-100): *"Set humor to 90"*, *"Be less sarcastic"*, *"What's your honesty level?"* — dials are `humor`, `sarcasm`, `honesty`.
- **Voxtral voice**: *"Use the angry voice"*, *"Be sarcastic"*, *"Switch to the British voice"* — swaps among the 10 Voxtral presets on the next spoken sentence.
- **Image roast**: *"Roast every image"*, *"Stop roasting"*, *"Roast less often"* — controls how often MOPS quips about a loaded PNG before cutting it. Default is 30% of the time.
- **LLM engine**: *"Switch to API engine"* / *"Switch to OpenAI"* / *"Use the CLI"* — flips between Claude CLI, direct Anthropic API, and direct OpenAI API mid-session.

API keys are deliberately NOT voice-changeable — edit `~/.mops-voice/config.json` to set those.

## Requirements

- macOS on Apple Silicon (M1/M2/M3) for the full local stack, or any Linux/Intel host with the Voxtral TTS engine (`tts_engine: voxtral` + Mistral API key). F5-TTS is Apple Silicon only.
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (for CLI engine), Anthropic API key (for `api` engine), or OpenAI API key (for `openai` engine)
- Node.js 18+ with `npx` on `PATH` — `npx` is not bundled with macOS or this repo; install Node from [nodejs.org](https://nodejs.org/) (or `brew install node`) and verify with `npx --version` before first run. The MOPS MCP server is then fetched on demand via `npx -y @thebeachlab/mops`. First run also needs `npx @thebeachlab/mops setup` once to install the Playwright Chromium browser.
- MOPS itself lives at [github.com/TheBeachLab/mops](https://github.com/TheBeachLab/mops) — see that repo for the tool list, server internals, and issue tracker.
- To pin to a local mops checkout instead, set `mops_server_command: "node /path/to/mops/src/server.js"` in `~/.mops-voice/config.json`.
