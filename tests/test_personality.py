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
