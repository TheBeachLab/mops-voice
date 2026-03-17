# mops-voice Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a voice-controlled CLI assistant (MOPS) that transcribes speech, reasons via Claude API with MOPS MCP tools for digital fabrication, and responds with a cloned TARS-like voice.

**Architecture:** Monolith async Python process. Push-to-talk spacebar triggers recording → whisper.cpp transcription → Claude API with MCP tool loop → F5-TTS voice synthesis → speaker playback. Keyboard listener in daemon thread, audio I/O via `run_in_executor`, MCP client and Claude API use native async.

**Tech Stack:** Python 3.14, uv, asyncio, sounddevice, pynput, pywhispercpp, anthropic (async), mcp (stdio client), f5-tts-mlx, rich

**Spec:** `docs/superpowers/specs/2026-03-17-mops-voice-design.md`

---

## File Map

```
mops-voice/
├── mops_voice/
│   ├── __init__.py              # Package init, version
│   ├── __main__.py              # Entry: python -m mops_voice
│   ├── config.py                # Config loading, merging with defaults, paths
│   ├── personality.py           # Personality dials, validation, Claude tool schemas
│   ├── audio.py                 # Mic recording, speaker playback (sounddevice)
│   ├── transcribe.py            # whisper.cpp wrapper
│   ├── tts.py                   # F5-TTS voice synthesis wrapper
│   ├── llm.py                   # Claude API async client, MCP client, tool loop
│   └── main.py                  # Main async loop, verbose output, orchestration
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_personality.py
│   ├── test_transcribe.py
│   ├── test_llm.py
│   └── test_main.py
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `mops_voice/__init__.py`
- Create: `mops_voice/__main__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "mops-voice"
version = "0.1.0"
description = "Voice-controlled CLI assistant for digital fabrication via MOPS MCP"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.52",
    "mcp>=1.0,<2",
    "pywhispercpp>=1.3",
    "f5-tts-mlx>=0.4",
    "sounddevice>=0.5",
    "soundfile>=0.13",
    "pynput>=1.8",
    "rich>=13.0",
    "numpy>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/
.env
```

- [ ] **Step 3: Create mops_voice/__init__.py**

```python
"""mops-voice: Voice-controlled CLI assistant for digital fabrication."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create mops_voice/__main__.py**

Minimal stub that will be filled in Task 8:

```python
"""Entry point for python -m mops_voice."""

import asyncio
from mops_voice.main import run


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create tests/__init__.py**

Empty file.

- [ ] **Step 6: Initialize project with uv**

Run: `cd /Users/Papi/Repositories/mops-voice && uv sync --dev`
Expected: virtual environment created, all dependencies installed.

Note: If any dependency fails on Python 3.14, pin `requires-python = ">=3.12,<3.14"` and use `uv python install 3.13` then `uv sync`.

- [ ] **Step 7: Verify pytest runs**

Run: `uv run pytest --co`
Expected: "no tests ran" (no test files yet), but pytest itself loads without error.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore mops_voice/ tests/ uv.lock
git commit -m "feat: project scaffolding with uv and dependencies"
```

---

## Task 2: Config Module

**Files:**
- Create: `mops_voice/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import json
from pathlib import Path

from mops_voice.config import load_config, save_config, DEFAULT_CONFIG, CONFIG_DIR


def test_default_config_has_required_keys():
    assert "assistant_name" in DEFAULT_CONFIG
    assert "user_name" in DEFAULT_CONFIG
    assert "personality" in DEFAULT_CONFIG
    assert "mops_server_path" in DEFAULT_CONFIG
    assert "whisper_model" in DEFAULT_CONFIG
    assert "claude_model" in DEFAULT_CONFIG


def test_load_config_returns_defaults_when_no_file(tmp_path):
    config = load_config(tmp_path / "config.json")
    assert config == DEFAULT_CONFIG


def test_load_config_merges_with_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"user_name": "Alice"}))
    config = load_config(config_path)
    assert config["user_name"] == "Alice"
    assert config["assistant_name"] == "MOPS"  # default preserved
    assert config["personality"]["humor"] == 75  # nested default preserved


def test_load_config_merges_partial_personality(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"personality": {"humor": 90}}))
    config = load_config(config_path)
    assert config["personality"]["humor"] == 90
    assert config["personality"]["sarcasm"] == 50  # default preserved


def test_save_config_writes_json(tmp_path):
    config_path = tmp_path / "config.json"
    save_config(config_path, DEFAULT_CONFIG)
    loaded = json.loads(config_path.read_text())
    assert loaded == DEFAULT_CONFIG


def test_config_dir_is_dot_mops_voice():
    assert CONFIG_DIR == Path.home() / ".mops-voice"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `cannot import name 'load_config' from 'mops_voice.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# mops_voice/config.py
"""Config loading with merge-with-defaults strategy."""

import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".mops-voice"

DEFAULT_CONFIG = {
    "assistant_name": "MOPS",
    "user_name": "Fran",
    "personality": {
        "humor": 75,
        "sarcasm": 50,
        "honesty": 90,
    },
    "mops_server_path": "../mops/src/server.js",
    "whisper_model": "base.en",
    "claude_model": "claude-sonnet-4-20250514",
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Merge overrides into defaults, recursing into nested dicts."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict:
    """Load config from path, merging with defaults. Missing keys get defaults."""
    import copy
    if path is None:
        path = CONFIG_DIR / "config.json"
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path) as f:
        user_config = json.load(f)
    return _deep_merge(DEFAULT_CONFIG, user_config)


def save_config(path: Path | None, config: dict) -> None:
    """Save config to path, creating parent directories if needed."""
    if path is None:
        path = CONFIG_DIR / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mops_voice/config.py tests/test_config.py
git commit -m "feat: config module with deep-merge defaults strategy"
```

---

## Task 3: Personality Module

**Files:**
- Create: `mops_voice/personality.py`
- Create: `tests/test_personality.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_personality.py
import json
from pathlib import Path

import pytest

from mops_voice.personality import (
    VALID_DIALS,
    validate_dial,
    adjust_personality,
    get_personality,
    personality_tool_schemas,
)
from mops_voice.config import load_config, save_config, DEFAULT_CONFIG


def test_valid_dials():
    assert VALID_DIALS == {"humor", "sarcasm", "honesty"}


def test_validate_dial_accepts_valid():
    assert validate_dial("humor", 75) is None


def test_validate_dial_rejects_invalid_name():
    error = validate_dial("creativity", 50)
    assert "Invalid dial" in error


def test_validate_dial_rejects_out_of_range():
    error = validate_dial("humor", 150)
    assert "range" in error.lower()
    error = validate_dial("humor", -5)
    assert "range" in error.lower()


def test_validate_dial_rejects_non_integer():
    error = validate_dial("humor", 50.5)
    assert error is not None


def test_adjust_personality_updates_and_saves(tmp_path):
    config_path = tmp_path / "config.json"
    save_config(config_path, DEFAULT_CONFIG)
    config = load_config(config_path)
    result = adjust_personality(config, config_path, "humor", 90)
    assert result["humor"] == 90
    # Verify persisted
    reloaded = load_config(config_path)
    assert reloaded["personality"]["humor"] == 90


def test_adjust_personality_returns_error_for_invalid(tmp_path):
    config_path = tmp_path / "config.json"
    config = DEFAULT_CONFIG.copy()
    result = adjust_personality(config, config_path, "humor", 200)
    assert isinstance(result, str)  # error message


def test_get_personality():
    config = {"personality": {"humor": 75, "sarcasm": 50, "honesty": 90}}
    result = get_personality(config)
    assert result == {"humor": 75, "sarcasm": 50, "honesty": 90}


def test_personality_tool_schemas_are_valid():
    schemas = personality_tool_schemas()
    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"adjust_personality", "get_personality"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_personality.py -v`
Expected: FAIL — `cannot import name 'VALID_DIALS'`

- [ ] **Step 3: Write minimal implementation**

```python
# mops_voice/personality.py
"""Personality dial management — TARS-style configurable personality."""

from pathlib import Path

from mops_voice.config import save_config

VALID_DIALS = {"humor", "sarcasm", "honesty"}


def validate_dial(dial: str, value: int) -> str | None:
    """Return error message if invalid, None if valid."""
    if dial not in VALID_DIALS:
        return f"Invalid dial '{dial}'. Valid dials: {', '.join(sorted(VALID_DIALS))}"
    if not isinstance(value, int) or isinstance(value, bool):
        return f"Value must be an integer, got {type(value).__name__}"
    if value < 0 or value > 100:
        return f"Value must be in range 0-100, got {value}"
    return None


def adjust_personality(
    config: dict, config_path: Path, dial: str, value: int
) -> dict | str:
    """Adjust a personality dial. Returns updated dials dict, or error string."""
    error = validate_dial(dial, value)
    if error:
        return error
    config["personality"][dial] = value
    save_config(config_path, config)
    return config["personality"].copy()


def get_personality(config: dict) -> dict:
    """Return current personality dial values."""
    return config["personality"].copy()


def personality_tool_schemas() -> list[dict]:
    """Return Claude API tool schemas for personality tools."""
    return [
        {
            "name": "adjust_personality",
            "description": (
                "Adjust a personality dial. Valid dials: humor, sarcasm, honesty. "
                "Values 0-100. Persists across sessions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "dial": {
                        "type": "string",
                        "enum": sorted(VALID_DIALS),
                        "description": "The personality dial to adjust",
                    },
                    "value": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "New value for the dial (0-100)",
                    },
                },
                "required": ["dial", "value"],
            },
        },
        {
            "name": "get_personality",
            "description": "Get current personality dial values (humor, sarcasm, honesty).",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_personality.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mops_voice/personality.py tests/test_personality.py
git commit -m "feat: personality dials with validation and Claude tool schemas"
```

---

## Task 4: Audio Module

**Files:**
- Create: `mops_voice/audio.py`
- Create: `tests/test_audio.py` (limited — hardware-dependent)

- [ ] **Step 1: Write the tests**

Audio is hardware-dependent, so tests focus on the interface contract and data format:

```python
# tests/test_audio.py
import numpy as np

from mops_voice.audio import SAMPLE_RATE, CHANNELS, audio_to_wav_bytes


def test_sample_rate_is_16k():
    assert SAMPLE_RATE == 16000


def test_channels_is_mono():
    assert CHANNELS == 1


def test_audio_to_wav_bytes_returns_bytes():
    # 1 second of silence
    audio_data = np.zeros((16000, 1), dtype=np.int16)
    wav_bytes = audio_to_wav_bytes(audio_data)
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 44  # WAV header is 44 bytes
    assert wav_bytes[:4] == b"RIFF"


def test_audio_to_wav_bytes_empty_audio():
    audio_data = np.zeros((0, 1), dtype=np.int16)
    wav_bytes = audio_to_wav_bytes(audio_data)
    assert wav_bytes is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL — `cannot import name 'SAMPLE_RATE'`

- [ ] **Step 3: Write implementation**

```python
# mops_voice/audio.py
"""Audio recording and playback via sounddevice."""

import io
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def audio_to_wav_bytes(audio_data: np.ndarray) -> bytes | None:
    """Convert numpy audio array to WAV bytes. Returns None if empty."""
    if audio_data.size == 0:
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    return buf.getvalue()


def record_until_release(stop_event) -> np.ndarray | None:
    """Record audio until stop_event is set. Returns numpy array or None."""
    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            pass  # ignore underflow warnings during recording
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    ):
        stop_event.wait()

    if not frames:
        return None
    return np.concatenate(frames, axis=0)


def play_audio(audio_data: np.ndarray, sample_rate: int = 24000) -> None:
    """Play audio through speakers. Blocks until complete."""
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add mops_voice/audio.py tests/test_audio.py
git commit -m "feat: audio recording and playback module with sounddevice"
```

---

## Task 5: Transcription Module

**Files:**
- Create: `mops_voice/transcribe.py`
- Create: `tests/test_transcribe.py`

- [ ] **Step 1: Write the tests**

```python
# tests/test_transcribe.py
from mops_voice.transcribe import is_gibberish, HALLUCINATION_BLOCKLIST


def test_is_gibberish_empty():
    assert is_gibberish("") is True


def test_is_gibberish_short():
    assert is_gibberish("hi") is True  # < 3 chars


def test_is_gibberish_valid():
    assert is_gibberish("mill the PCB traces") is False


def test_is_gibberish_hallucination():
    assert is_gibberish("Thank you for watching") is True


def test_hallucination_blocklist_is_lowercase():
    for entry in HALLUCINATION_BLOCKLIST:
        assert entry == entry.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `cannot import name 'is_gibberish'`

- [ ] **Step 3: Write implementation**

```python
# mops_voice/transcribe.py
"""Speech-to-text via whisper.cpp with CoreML acceleration."""

from pywhispercpp.model import Model

HALLUCINATION_BLOCKLIST = [
    "thank you for watching",
    "thanks for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
]


def is_gibberish(text: str) -> bool:
    """Return True if transcription should be skipped."""
    text = text.strip()
    if len(text) < 3:
        return True
    if text.lower() in HALLUCINATION_BLOCKLIST:
        return True
    return False


class Transcriber:
    """Whisper.cpp wrapper. Load once, transcribe many."""

    def __init__(self, model_name: str = "base.en"):
        self.model = Model(model_name)

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text."""
        # pywhispercpp expects a file path or ndarray
        # Write to temp file for simplicity
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        try:
            segments = self.model.transcribe(tmp_path)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            os.unlink(tmp_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: All 5 tests PASS (these tests don't instantiate the model)

- [ ] **Step 5: Commit**

```bash
git add mops_voice/transcribe.py tests/test_transcribe.py
git commit -m "feat: transcription module with gibberish filtering"
```

---

## Task 6: TTS Module

**Files:**
- Create: `mops_voice/tts.py`

- [ ] **Step 1: Write implementation**

TTS depends on f5-tts-mlx which requires MLX and Apple Silicon. Tests would need the model loaded. Write the wrapper, test manually.

```python
# mops_voice/tts.py
"""Text-to-speech via F5-TTS with MLX acceleration and voice cloning."""

import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf


class Synthesizer:
    """F5-TTS wrapper. Load once, synthesize many."""

    SAMPLE_RATE = 24000  # F5-TTS default output rate

    def __init__(self, ref_audio_path: Path, ref_text_path: Path):
        from f5_tts_mlx.generate import generate

        self._generate = generate
        self.ref_audio_path = str(ref_audio_path)

        if not Path(self.ref_audio_path).exists():
            raise FileNotFoundError(
                f"Reference audio not found: {ref_audio_path}\n"
                "Place a 5-15s WAV clip (24kHz mono 16-bit) of the target voice there."
            )

        if not ref_text_path.exists():
            raise FileNotFoundError(
                f"Reference transcript not found: {ref_text_path}\n"
                "Create it with the exact text spoken in the reference audio."
            )
        self.ref_text = ref_text_path.read_text().strip()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio. Returns (audio_array, sample_rate).

        f5_tts_mlx.generate() writes to a file (side-effect only).
        We write to a temp file, then read it back as a numpy array.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            self._generate(
                generation_text=text,
                ref_audio_path=self.ref_audio_path,
                ref_audio_text=self.ref_text,
                output_path=tmp_path,
            )
            audio_data, sample_rate = sf.read(tmp_path, dtype="float32")
            return audio_data, sample_rate
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
```

Note: Add `soundfile` to dependencies. The exact `f5_tts_mlx` API may differ — verify during implementation by checking `uv run python -c "from f5_tts_mlx.generate import generate; help(generate)"`. Adjust parameter names if needed. The key insight is that `generate()` returns `None` and writes to `output_path`.

- [ ] **Step 2: Commit**

```bash
git add mops_voice/tts.py
git commit -m "feat: TTS module wrapping F5-TTS MLX with voice cloning"
```

---

## Task 7: LLM & MCP Module

**Files:**
- Create: `mops_voice/llm.py`
- Create: `tests/test_llm.py`

This is the most complex module — Claude API async client, MCP subprocess management, tool loop.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
import pytest

from mops_voice.llm import (
    build_system_prompt,
    MAX_TOOL_ITERATIONS,
    handle_personality_tool,
)
from mops_voice.config import DEFAULT_CONFIG


def test_build_system_prompt_includes_personality():
    config = DEFAULT_CONFIG.copy()
    prompt = build_system_prompt(config)
    assert "MOPS" in prompt
    assert "Fran" in prompt
    assert "humor=75%" in prompt
    assert "sarcasm=50%" in prompt
    assert "honesty=90%" in prompt


def test_build_system_prompt_updates_with_changed_dials():
    config = DEFAULT_CONFIG.copy()
    config["personality"]["humor"] = 10
    prompt = build_system_prompt(config)
    assert "humor=10%" in prompt


def test_max_tool_iterations():
    assert MAX_TOOL_ITERATIONS == 10


def test_handle_personality_tool_adjust(tmp_path):
    config = DEFAULT_CONFIG.copy()
    config_path = tmp_path / "config.json"
    result = handle_personality_tool(
        "adjust_personality",
        {"dial": "humor", "value": 90},
        config,
        config_path,
    )
    assert "90" in result
    assert config["personality"]["humor"] == 90


def test_handle_personality_tool_get():
    config = DEFAULT_CONFIG.copy()
    result = handle_personality_tool(
        "get_personality",
        {},
        config,
        None,
    )
    assert "humor" in result
    assert "75" in result


def test_handle_personality_tool_invalid():
    config = DEFAULT_CONFIG.copy()
    result = handle_personality_tool(
        "adjust_personality",
        {"dial": "humor", "value": 200},
        config,
        None,
    )
    assert "range" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — `cannot import name 'build_system_prompt'`

- [ ] **Step 3: Write implementation**

```python
# mops_voice/llm.py
"""Claude API async client with MCP integration and tool loop."""

import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mops_voice.personality import (
    adjust_personality,
    get_personality,
    personality_tool_schemas,
)

MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT_TEMPLATE = """\
You are {assistant_name}, a voice assistant for digital fabrication.
You address the user as {user_name}.
Personality settings: humor={humor}%, sarcasm={sarcasm}%, honesty={honesty}%.
Adjust your tone accordingly. Keep responses concise — they'll be spoken aloud.
You control fabrication machines through MOPS tools.
When a personality setting is changed, confirm it briefly in-character.
When asked about your settings, report them.\
"""


def build_system_prompt(config: dict) -> str:
    """Build system prompt with current personality settings."""
    p = config["personality"]
    return SYSTEM_PROMPT_TEMPLATE.format(
        assistant_name=config["assistant_name"],
        user_name=config["user_name"],
        humor=p["humor"],
        sarcasm=p["sarcasm"],
        honesty=p["honesty"],
    )


def handle_personality_tool(
    tool_name: str,
    tool_input: dict,
    config: dict,
    config_path: Path | None,
) -> str:
    """Handle a personality tool call. Returns result string for Claude."""
    if tool_name == "adjust_personality":
        result = adjust_personality(
            config, config_path, tool_input["dial"], tool_input["value"]
        )
        if isinstance(result, str):
            return result  # error message
        return json.dumps(result)
    elif tool_name == "get_personality":
        return json.dumps(get_personality(config))
    return f"Unknown personality tool: {tool_name}"


class MopsLLM:
    """Manages Claude API calls and MCP tool execution."""

    def __init__(self, config: dict, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
        self.messages: list[dict] = []
        self.max_messages = 100  # 50 pairs

    @staticmethod
    def check_api_key() -> bool:
        """Check if ANTHROPIC_API_KEY is set. Call at startup."""
        import os
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.mcp_session: ClientSession | None = None
        self.mcp_tools: list[dict] = []
        self._mcp_cm = None
        self._mcp_session_cm = None

    async def connect_mcp(self, server_path: str, extra_args: list[str] | None = None) -> bool:
        """Connect to the MOPS MCP server. Returns True on success."""
        try:
            args = [server_path] + (extra_args or [])
            server_params = StdioServerParameters(
                command="node",
                args=args,
            )
            # Keep context managers alive for session lifetime
            self._mcp_cm = stdio_client(server_params)
            read, write = await self._mcp_cm.__aenter__()
            self._mcp_session_cm = ClientSession(read, write)
            self.mcp_session = await self._mcp_session_cm.__aenter__()
            await self.mcp_session.initialize()

            # Discover tools
            tools_result = await self.mcp_session.list_tools()
            self.mcp_tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_result.tools
            ]
            return True
        except Exception:
            self.mcp_session = None
            self.mcp_tools = []
            return False

    async def disconnect_mcp(self):
        """Shut down MCP session and subprocess."""
        if self._mcp_session_cm:
            try:
                await self._mcp_session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._mcp_cm:
            try:
                await self._mcp_cm.__aexit__(None, None, None)
            except Exception:
                pass

    def _get_tools(self) -> list[dict]:
        """Get all tools: MCP tools + personality tools."""
        return self.mcp_tools + personality_tool_schemas()

    def _prune_messages(self):
        """Prune old messages, keeping last 20 (10 pairs)."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-20:]

    async def chat(self, user_text: str, on_tool_call=None) -> str:
        """Send user text, handle tool loop, return final text response.

        on_tool_call: optional callback(tool_name, result_summary) for verbose output.
        """
        self.messages.append({"role": "user", "content": user_text})
        self._prune_messages()

        tools = self._get_tools()

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = await self.client.messages.create(
                    model=self.config["claude_model"],
                    max_tokens=1024,
                    system=build_system_prompt(self.config),
                    messages=self.messages,
                    tools=tools if tools else None,
                )
            except Exception:
                # API error, rate limit, or network issue
                return "I'm having trouble reaching my brain, Fran. Try again."

            # Collect text and tool_use blocks
            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                # No more tool calls — we have the final response
                final_text = " ".join(text_parts).strip()
                self.messages.append({"role": "assistant", "content": response.content})
                return final_text

            # Process tool calls
            self.messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input

                # Check if it's a personality tool
                if tool_name in ("adjust_personality", "get_personality"):
                    result = handle_personality_tool(
                        tool_name, tool_input, self.config, self.config_path
                    )
                elif self.mcp_session:
                    # Call MCP tool
                    try:
                        mcp_result = await self.mcp_session.call_tool(
                            tool_name, tool_input
                        )
                        result = (
                            mcp_result.content[0].text
                            if mcp_result.content
                            else "OK"
                        )
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = "Tool unavailable — MCP server not connected."

                if on_tool_call:
                    # Summarize result for verbose output
                    summary = result[:80] + "..." if len(result) > 80 else result
                    on_tool_call(tool_name, summary)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})

        # Exceeded max iterations
        return "I got a bit carried away with the tools there, Fran. Let me try again."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v`
Expected: All 6 tests PASS (these test pure functions, not async/network)

- [ ] **Step 5: Commit**

```bash
git add mops_voice/llm.py tests/test_llm.py
git commit -m "feat: LLM module with Claude API, MCP client, and tool loop"
```

---

## Task 8: Main Loop & CLI

**Files:**
- Create: `mops_voice/main.py`
- Modify: `mops_voice/__main__.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the tests**

```python
# tests/test_main.py
import argparse

from mops_voice.main import parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.headed is False
    assert args.mods_url is None
    assert args.whisper_model is None


def test_parse_args_headed():
    args = parse_args(["--headed"])
    assert args.headed is True


def test_parse_args_mods_url():
    args = parse_args(["--mods-url", "http://localhost:8080"])
    assert args.mods_url == "http://localhost:8080"


def test_parse_args_whisper_model():
    args = parse_args(["--whisper-model", "small.en"])
    assert args.whisper_model == "small.en"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL — `cannot import name 'parse_args'`

- [ ] **Step 3: Write implementation**

```python
# mops_voice/main.py
"""Main loop: push-to-talk → transcribe → Claude → TTS → playback."""

import argparse
import asyncio
import sys
import threading
from pathlib import Path

from rich.console import Console

from mops_voice.config import load_config, CONFIG_DIR
from mops_voice.audio import record_until_release, play_audio, audio_to_wav_bytes
from mops_voice.transcribe import Transcriber, is_gibberish
from mops_voice.tts import Synthesizer
from mops_voice.llm import MopsLLM

console = Console()

EXIT_PHRASES = {"goodbye", "exit", "quit", "bye"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOPS voice assistant")
    parser.add_argument(
        "--mods-url", type=str, default=None, help="Mods CE URL (forwarded to MOPS server)"
    )
    parser.add_argument(
        "--headed", action="store_true", default=False, help="Show browser window (default: headless)"
    )
    parser.add_argument(
        "--whisper-model", type=str, default=None, help="Whisper model name"
    )
    return parser.parse_args(argv)


async def run(argv: list[str] | None = None):
    args = parse_args(argv)
    config = load_config()
    config_path = CONFIG_DIR / "config.json"

    # Override config from CLI args
    if args.whisper_model:
        config["whisper_model"] = args.whisper_model

    # --- Startup checks ---
    import sounddevice as sd

    # Check mic
    try:
        sd.query_devices(kind="input")
    except sd.PortAudioError:
        console.print("[red]No microphone detected. Cannot start.[/red]")
        return

    # Check API key
    if not MopsLLM.check_api_key():
        console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
        console.print("Export it: export ANTHROPIC_API_KEY='your-key'")
        return

    console.print("[bold cyan]MOPS Voice Assistant[/bold cyan]")
    console.print(f"Assistant: {config['assistant_name']}  |  User: {config['user_name']}")
    p = config["personality"]
    console.print(
        f"Personality: humor={p['humor']}% sarcasm={p['sarcasm']}% honesty={p['honesty']}%"
    )
    console.print()

    # --- Initialize components ---

    # Transcriber
    console.print("📝 Loading whisper model...", end=" ")
    try:
        transcriber = Transcriber(config["whisper_model"])
        console.print("[green]OK[/green]")
    except Exception as e:
        console.print(f"[red]FAILED: {e}[/red]")
        return

    # TTS
    ref_audio = CONFIG_DIR / "tars_reference.wav"
    ref_text = CONFIG_DIR / "tars_reference.txt"
    synthesizer = None
    console.print("🔊 Loading TTS model...", end=" ")
    try:
        synthesizer = Synthesizer(ref_audio, ref_text)
        console.print("[green]OK[/green]")
    except FileNotFoundError as e:
        console.print(f"[yellow]WARN: {e}[/yellow]")
        console.print("[yellow]   TTS disabled — text-only mode[/yellow]")
    except Exception as e:
        console.print(f"[yellow]WARN: TTS failed: {e}[/yellow]")
        console.print("[yellow]   TTS disabled — text-only mode[/yellow]")

    # LLM + MCP
    llm = MopsLLM(config, config_path)

    # Build MOPS server args
    mcp_args = []
    if args.headed:
        pass  # default is headed for MOPS server
    else:
        mcp_args.append("--headless")
    if args.mods_url:
        mcp_args.extend(["--mods-url", args.mods_url])

    server_path = config["mops_server_path"]
    # Resolve relative path from mops-voice repo root
    if not Path(server_path).is_absolute():
        server_path = str((Path(__file__).parent.parent / server_path).resolve())

    console.print("🤖 Connecting to MOPS MCP server...", end=" ")
    mcp_connected = await llm.connect_mcp(server_path, mcp_args)
    if mcp_connected:
        console.print(f"[green]OK ({len(llm.mcp_tools)} tools)[/green]")
    else:
        console.print("[yellow]WARN: Could not connect[/yellow]")
        console.print("[yellow]   Conversation-only mode (no machine control)[/yellow]")

    console.print()
    console.print("[bold]Hold SPACEBAR to talk. Ctrl+C to exit.[/bold]")
    console.print()

    # --- Keyboard listener ---
    from pynput import keyboard

    recording = False
    stop_event = threading.Event()
    space_held = False

    def on_press(key):
        nonlocal recording, space_held
        if key == keyboard.Key.space and not space_held:
            space_held = True
            if not recording:
                recording = True
                stop_event.clear()

    def on_release(key):
        nonlocal recording, space_held
        if key == keyboard.Key.space:
            space_held = False
            if recording:
                recording = False
                stop_event.set()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    # --- Main loop ---
    loop = asyncio.get_running_loop()

    try:
        while True:
            # Wait for spacebar press
            while not recording:
                await asyncio.sleep(0.05)

            console.print("🎤 [bold red]Recording...[/bold red]", end=" ")

            # Record in thread
            audio_data = await loop.run_in_executor(
                None, record_until_release, stop_event
            )

            if audio_data is None or audio_data.size == 0:
                console.print("[yellow]No audio captured[/yellow]")
                continue

            duration = len(audio_data) / 16000
            console.print(f"[green]{duration:.1f}s[/green]")

            # Transcribe
            wav_bytes = audio_to_wav_bytes(audio_data)
            if wav_bytes is None:
                console.print("[yellow]No audio captured[/yellow]")
                continue

            console.print("📝 Transcribing...", end=" ")
            text = await loop.run_in_executor(
                None, transcriber.transcribe, wav_bytes
            )
            console.print(f'[cyan]"{text}"[/cyan]')

            if is_gibberish(text):
                console.print("[yellow]Didn't catch that, Fran.[/yellow]")
                continue

            # Check for exit
            if text.strip().lower().rstrip(".!") in EXIT_PHRASES:
                console.print("🤖 ", end="")
                farewell = await llm.chat("Fran is leaving. Say a brief goodbye.")
                console.print(f"[green]{farewell}[/green]")
                if synthesizer:
                    try:
                        audio, sr = await loop.run_in_executor(
                            None, synthesizer.synthesize, farewell
                        )
                        await loop.run_in_executor(None, play_audio, audio, sr)
                    except Exception:
                        pass
                break

            # Claude API with tool loop
            import time

            t0 = time.monotonic()
            console.print(f"🤖 Calling Claude ({config['claude_model'].split('-')[1]})...")

            def on_tool_call(name, summary):
                console.print(f"  🔧 Tool call: {name} → {summary}")

            response_text = await llm.chat(text, on_tool_call=on_tool_call)
            console.print(f"🤖 Response: [green]{response_text}[/green]")

            # TTS
            if synthesizer:
                console.print("🔊 Synthesizing speech...", end=" ")
                try:
                    audio, sr = await loop.run_in_executor(
                        None, synthesizer.synthesize, response_text
                    )
                    audio_duration = len(audio) / sr
                    console.print(f"[green]{audio_duration:.1f}s[/green]")
                    console.print("🔊 Playing...", end=" ")
                    await loop.run_in_executor(None, play_audio, audio, sr)
                    console.print("[green]done[/green]")
                except Exception as e:
                    console.print(f"[red]TTS error: {e}[/red]")

            elapsed = time.monotonic() - t0
            console.print(f"⏱️  Total: {elapsed:.1f}s")
            console.print()

    except KeyboardInterrupt:
        console.print("\n[bold]Shutting down...[/bold]")
    finally:
        listener.stop()
        await llm.disconnect_mcp()
        console.print("[bold cyan]MOPS out.[/bold cyan]")
```

- [ ] **Step 4: Update __main__.py**

```python
"""Entry point for python -m mops_voice."""

import asyncio
from mops_voice.main import run


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_main.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add mops_voice/main.py mops_voice/__main__.py tests/test_main.py
git commit -m "feat: main loop with push-to-talk, verbose output, and graceful shutdown"
```

---

## Task 9: Manual Integration Test

**No files created** — this is a verification task.

- [ ] **Step 1: Create config directory and reference files**

```bash
mkdir -p ~/.mops-voice
```

If you have the TARS reference audio, place it at `~/.mops-voice/tars_reference.wav` with its transcript at `~/.mops-voice/tars_reference.txt`. If not, MOPS will run in text-only mode.

- [ ] **Step 2: Set API key**

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

- [ ] **Step 3: Run the application**

```bash
cd /Users/Papi/Repositories/mops-voice
uv run python -m mops_voice
```

Expected startup output:
```
MOPS Voice Assistant
Assistant: MOPS  |  User: Fran
Personality: humor=75% sarcasm=50% honesty=90%

📝 Loading whisper model... OK
🔊 Loading TTS model... WARN: ...
🤖 Connecting to MOPS MCP server... OK (13 tools)

Hold SPACEBAR to talk. Ctrl+C to exit.
```

- [ ] **Step 4: Test voice interaction**

Hold spacebar, say "Hello MOPS", release. Verify:
- Recording indicator appears
- Transcription shown
- Claude response shown
- Tool calls (if any) shown
- TTS plays (if reference audio available)

- [ ] **Step 5: Test personality dials**

Say "Set humor to 100". Verify:
- Tool call `adjust_personality` shown
- MOPS confirms in-character
- Check `~/.mops-voice/config.json` was updated

- [ ] **Step 6: Test exit**

Say "Goodbye". Verify MOPS says farewell and exits cleanly.

- [ ] **Step 7: Commit any fixes**

```bash
git add -u
git commit -m "fix: integration test adjustments"
```

---

## Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
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
# Place reference audio: ~/.mops-voice/tars_reference.wav (24kHz mono WAV)
# Place transcript:      ~/.mops-voice/tars_reference.txt

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage instructions"
```
