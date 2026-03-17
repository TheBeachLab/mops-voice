# mops-voice

Voice-controlled CLI assistant for digital fabrication. Talk to MOPS, control your machines.

## Pipeline

```
🎤 Mic → whisper.cpp → Claude API + MOPS MCP tools → F5-TTS → 🔊 Speaker
```

## Setup

```bash
# Install
uv sync

# Configure
mkdir -p ~/.mops-voice
# Place reference audio: ~/.mops-voice/tars_reference.wav (24kHz mono WAV, 5-15s)
# Place transcript:      ~/.mops-voice/tars_reference.txt (exact text from audio)

# API key
export ANTHROPIC_API_KEY="your-key"
```

## Usage

```bash
uv run python -m mops_voice
```

Hold **spacebar** to talk. **Ctrl+C** to exit.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mods-url URL` | mods.beachlab.org | Mods CE deployment |
| `--headed` | off | Show browser window |
| `--whisper-model NAME` | base.en | Whisper model |

### Personality

MOPS has adjustable personality dials (0-100):
- **humor** (default 75) — "Set humor to 90"
- **sarcasm** (default 50) — "Be less sarcastic"
- **honesty** (default 90) — "What's your honesty level?"

Settings persist in `~/.mops-voice/config.json`.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Node.js (for MOPS MCP server)
- [MOPS](../mops) MCP server in sibling directory
