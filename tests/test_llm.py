import copy
import json

import pytest

from mops_voice.llm import (
    build_system_prompt,
    build_mcp_config,
    _format_history,
    MAX_HISTORY_TURNS,
    MopsLLM,
)
from mops_voice.config import DEFAULT_CONFIG


def test_build_system_prompt_includes_personality():
    config = copy.deepcopy(DEFAULT_CONFIG)
    prompt = build_system_prompt(config)
    assert "MOPS" in prompt
    assert "Fran" in prompt
    assert "humor=75%" in prompt
    assert "sarcasm=50%" in prompt
    assert "honesty=90%" in prompt


def test_build_system_prompt_updates_with_changed_dials():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["personality"]["humor"] = 10
    prompt = build_system_prompt(config)
    assert "humor=10%" in prompt


def test_max_history_turns():
    assert MAX_HISTORY_TURNS == 10


def test_build_mcp_config():
    config = build_mcp_config("/path/to/server.js", ["--headless"])
    assert "mcpServers" in config
    assert "mops" in config["mcpServers"]
    assert config["mcpServers"]["mops"]["command"] == "node"
    assert config["mcpServers"]["mops"]["args"] == ["/path/to/server.js", "--headless"]


def test_format_history_empty():
    assert _format_history([]) == ""


def test_format_history_with_entries():
    history = [
        {"user": "hello", "assistant": "Hi Fran"},
        {"user": "mill PCB", "assistant": "Loading program"},
    ]
    result = _format_history(history)
    assert "User: hello" in result
    assert "MOPS: Hi Fran" in result
    assert "User: mill PCB" in result


def test_check_cli():
    # claude should be available on this machine
    assert MopsLLM.check_cli() is True


def test_extract_personality_update(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = tmp_path / "config.json"
    llm = MopsLLM(config, config_path)
    text = "PERSONALITY_UPDATE:humor=90\nHumor 90, confirmed."
    result = llm._extract_personality_update(text)
    assert result == "Humor 90, confirmed."
    assert config["personality"]["humor"] == 90


def test_extract_personality_update_no_directive():
    config = copy.deepcopy(DEFAULT_CONFIG)
    llm = MopsLLM(config, None)
    text = "Just a normal response."
    result = llm._extract_personality_update(text)
    assert result == "Just a normal response."
    assert config["personality"]["humor"] == 75  # unchanged
