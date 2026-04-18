"""Config loading with merge-with-defaults strategy."""

import copy
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
    "llm_engine": "cli",
    "claude_model": "sonnet",
    "anthropic": {
        "api_key": "",
    },
    "tts_engine": "f5",
    "voxtral": {
        "api_key": "",
        "voice": "en_paul_confident",
    },
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
    if path is None:
        path = CONFIG_DIR / "config.json"
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path) as f:
        user_config = json.load(f)
    return _deep_merge(copy.deepcopy(DEFAULT_CONFIG), user_config)


def save_config(path: Path | None, config: dict) -> None:
    """Save config to path, creating parent directories if needed."""
    if path is None:
        path = CONFIG_DIR / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
