# mops-voice

Voice-controlled CLI assistant for digital fabrication. Talk to MOPS, control your machines.

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

**Direct API (faster)** — bypasses CLI startup overhead (~1-2s vs ~7s per turn):

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

### TTS engine setup

**F5-TTS (local, default)** — voice cloning via MLX, runs on Apple Silicon:

```bash
# Place reference audio and transcript
~/.mops-voice/tars_reference.wav   # 24kHz mono WAV, 5-15s
~/.mops-voice/tars_reference.txt   # exact text spoken in the audio
```

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

```bash
uv run python -m mops_voice
```

Hold **spacebar** to talk. **Ctrl+C** to exit.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-engine` | cli | LLM engine: `cli` (Claude Code CLI) or `api` (direct Anthropic API) |
| `--tts-engine` | f5 | TTS engine: `f5` (local MLX) or `voxtral` (Mistral API) |
| `--whisper-model NAME` | base.en | Whisper model |
| `--mods-url URL` | mods.beachlab.org | Mods CE deployment |
| `--headless` | off | Hide browser window |

### Personality

MOPS has adjustable personality dials (0-100):
- **humor** (default 75) — "Set humor to 90"
- **sarcasm** (default 50) — "Be less sarcastic"
- **honesty** (default 90) — "What's your honesty level?"

Settings persist in `~/.mops-voice/config.json`.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (for CLI engine) or Anthropic API key (for API engine)
- Node.js (for MOPS MCP server)
- [MOPS](../mops) MCP server in sibling directory
