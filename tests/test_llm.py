import copy

import pytest

from mops_voice.llm import (
    build_system_prompt,
    MAX_TOOL_ITERATIONS,
    handle_personality_tool,
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


def test_max_tool_iterations():
    assert MAX_TOOL_ITERATIONS == 10


def test_handle_personality_tool_adjust(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
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
    config = copy.deepcopy(DEFAULT_CONFIG)
    result = handle_personality_tool(
        "get_personality",
        {},
        config,
        None,
    )
    assert "humor" in result
    assert "75" in result


def test_handle_personality_tool_invalid():
    config = copy.deepcopy(DEFAULT_CONFIG)
    result = handle_personality_tool(
        "adjust_personality",
        {"dial": "humor", "value": 200},
        config,
        None,
    )
    assert "range" in result.lower()
