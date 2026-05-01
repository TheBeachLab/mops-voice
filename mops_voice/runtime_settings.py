"""Voice-mutable session settings: Voxtral voice, image-roast probability, LLM engine.

Mirrors the personality.py pattern: each setter validates, mutates the
config dict in place, persists to disk, and returns either a dict
snapshot of the new state or an error string the LLM can speak back.

API keys deliberately are NOT mutable from voice — they live in
~/.mops-voice/config.json alongside the other secrets.
"""

from __future__ import annotations

from pathlib import Path

from mops_voice.config import save_config

VOXTRAL_VOICES = [
    "en_paul_neutral",
    "en_paul_confident",
    "en_paul_happy",
    "en_paul_excited",
    "en_paul_cheerful",
    "en_paul_angry",
    "en_paul_frustrated",
    "en_paul_sad",
    "gb_oliver_neutral",
    "gb_jane_sarcasm",
]

VALID_LLM_ENGINES = ("cli", "api", "openai")


def set_voxtral_voice(config: dict, config_path: Path, voice: str) -> dict | str:
    if voice not in VOXTRAL_VOICES:
        return (
            f"Invalid voice '{voice}'. Choose from: {', '.join(VOXTRAL_VOICES)}."
        )
    config.setdefault("voxtral", {})["voice"] = voice
    save_config(config_path, config)
    return {"voice": voice}


def set_image_roast(config: dict, config_path: Path, probability: float) -> dict | str:
    try:
        prob = float(probability)
    except (TypeError, ValueError):
        return f"Probability must be a number between 0 and 1, got {probability!r}."
    if prob < 0 or prob > 1:
        return f"Probability must be between 0 and 1, got {prob}."
    config.setdefault("image_roast", {})["probability"] = prob
    save_config(config_path, config)
    return {"probability": prob}


def set_llm_engine(config: dict, config_path: Path, engine: str) -> dict | str:
    if engine not in VALID_LLM_ENGINES:
        return f"Invalid engine '{engine}'. Choose from: {', '.join(VALID_LLM_ENGINES)}."
    if engine == "api":
        key = (config.get("anthropic") or {}).get("api_key", "")
        if not key:
            return (
                "Cannot switch to api engine: anthropic.api_key is empty. "
                "Set it in ~/.mops-voice/config.json first."
            )
    if engine == "openai":
        key = (config.get("openai") or {}).get("api_key", "")
        if not key:
            return (
                "Cannot switch to openai engine: openai.api_key is empty. "
                "Set it in ~/.mops-voice/config.json first."
            )
    config["llm_engine"] = engine
    save_config(config_path, config)
    return {"engine": engine}


def get_voice_settings(config: dict) -> dict:
    return {
        "tts_engine": config.get("tts_engine"),
        "voxtral_voice": (config.get("voxtral") or {}).get("voice"),
        "llm_engine": config.get("llm_engine"),
        "image_roast_probability": (config.get("image_roast") or {}).get("probability"),
    }


def runtime_settings_tool_schemas() -> list[dict]:
    """Tool schemas for the Anthropic API tool list."""
    return [
        {
            "name": "set_voxtral_voice",
            "description": (
                "Change the Voxtral TTS voice mid-session. Takes effect on the "
                "next spoken sentence. Only meaningful when tts_engine=voxtral."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "voice": {
                        "type": "string",
                        "enum": VOXTRAL_VOICES,
                        "description": "One of the 10 Voxtral preset voices",
                    },
                },
                "required": ["voice"],
            },
        },
        {
            "name": "set_image_roast",
            "description": (
                "Set the probability (0.0–1.0) that a loaded image is shown to "
                "you for an in-character quip before the workflow continues. "
                "0 disables roasting; 1 roasts every cut."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "probability": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Probability between 0 and 1",
                    },
                },
                "required": ["probability"],
            },
        },
        {
            "name": "set_llm_engine",
            "description": (
                "Switch the LLM engine. Options: 'cli' (Claude Code CLI, "
                "slower but no API key), 'api' (direct Anthropic API, "
                "faster, requires anthropic.api_key), 'openai' (direct "
                "OpenAI API, requires openai.api_key). Takes effect next turn."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string",
                        "enum": list(VALID_LLM_ENGINES),
                        "description": "'cli', 'api', or 'openai'",
                    },
                },
                "required": ["engine"],
            },
        },
        {
            "name": "get_voice_settings",
            "description": (
                "Return current runtime settings: tts_engine, voxtral voice, "
                "llm_engine, image_roast probability."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
