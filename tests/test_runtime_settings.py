"""Tests for runtime_settings: voice-driven mutation of session config."""

import json
from pathlib import Path

import pytest

from mops_voice.runtime_settings import (
    VOXTRAL_VOICES,
    set_image_roast,
    set_llm_engine,
    set_voxtral_voice,
)


@pytest.fixture
def cfg(tmp_path: Path) -> tuple[dict, Path]:
    config = {
        "voxtral": {"api_key": "k", "voice": "en_paul_confident"},
        "image_roast": {"probability": 0.3, "max_dim": 512},
        "llm_engine": "api",
        "anthropic": {"api_key": "sk-test"},
    }
    return config, tmp_path / "config.json"


def test_set_voxtral_voice_accepts_known_voice(cfg):
    config, path = cfg
    out = set_voxtral_voice(config, path, "gb_jane_sarcasm")
    assert config["voxtral"]["voice"] == "gb_jane_sarcasm"
    assert "voice" in out and out["voice"] == "gb_jane_sarcasm"
    # Persisted
    assert json.loads(path.read_text())["voxtral"]["voice"] == "gb_jane_sarcasm"


def test_set_voxtral_voice_rejects_unknown(cfg):
    config, path = cfg
    out = set_voxtral_voice(config, path, "nonexistent_voice")
    assert isinstance(out, str) and "Invalid" in out
    assert config["voxtral"]["voice"] == "en_paul_confident"  # unchanged
    assert not path.exists()


def test_voxtral_voices_list_is_complete():
    # The 10 documented Voxtral voices
    expected = {
        "en_paul_neutral", "en_paul_confident", "en_paul_happy",
        "en_paul_excited", "en_paul_cheerful", "en_paul_angry",
        "en_paul_frustrated", "en_paul_sad",
        "gb_oliver_neutral", "gb_jane_sarcasm",
    }
    assert set(VOXTRAL_VOICES) == expected


def test_set_image_roast_accepts_zero_to_one(cfg):
    config, path = cfg
    out = set_image_roast(config, path, 0.0)
    assert config["image_roast"]["probability"] == 0.0
    out = set_image_roast(config, path, 1.0)
    assert config["image_roast"]["probability"] == 1.0
    out = set_image_roast(config, path, 0.42)
    assert config["image_roast"]["probability"] == pytest.approx(0.42)
    assert isinstance(out, dict)


def test_set_image_roast_rejects_out_of_range(cfg):
    config, path = cfg
    assert isinstance(set_image_roast(config, path, -0.1), str)
    assert isinstance(set_image_roast(config, path, 1.1), str)
    assert config["image_roast"]["probability"] == 0.3


def test_set_llm_engine_accepts_cli_and_api(cfg):
    config, path = cfg
    out = set_llm_engine(config, path, "cli")
    assert config["llm_engine"] == "cli"
    assert isinstance(out, dict)
    out = set_llm_engine(config, path, "api")
    assert config["llm_engine"] == "api"


def test_set_llm_engine_rejects_unknown(cfg):
    config, path = cfg
    out = set_llm_engine(config, path, "openai")
    assert isinstance(out, str)
    assert config["llm_engine"] == "api"


def test_set_llm_engine_to_api_requires_key(cfg):
    config, path = cfg
    config["anthropic"]["api_key"] = ""
    out = set_llm_engine(config, path, "api")
    assert isinstance(out, str) and "key" in out.lower()
    assert config["llm_engine"] == "api"  # not changed (we asked for api but it was already api)
    # And cli works fine without an Anthropic key
    out = set_llm_engine(config, path, "cli")
    assert config["llm_engine"] == "cli"
