import copy
import json

from mops_voice.llm import (
    build_system_prompt,
    _format_history,
    _parse_tool_calls,
    MAX_HISTORY_TURNS,
    MAX_TOOL_LOOPS,
    MopsLLM,
)
from mops_voice.config import DEFAULT_CONFIG


def test_build_system_prompt_includes_personality():
    config = copy.deepcopy(DEFAULT_CONFIG)
    prompt = build_system_prompt(config)
    assert "MOPS" in prompt
    assert "Fran" in prompt
    assert "Humor: 75%" in prompt
    assert "Sarcasm: 50%" in prompt
    assert "Honesty: 90%" in prompt


def test_build_system_prompt_updates_with_changed_dials():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["personality"]["humor"] = 10
    prompt = build_system_prompt(config)
    assert "Humor: 10%" in prompt
    assert "dry and professional" in prompt


def test_build_system_prompt_includes_tools():
    config = copy.deepcopy(DEFAULT_CONFIG)
    prompt = build_system_prompt(config, "- load_program: Load a program")
    assert "load_program" in prompt
    assert "TOOL_CALL" in prompt


def test_build_system_prompt_no_tools():
    config = copy.deepcopy(DEFAULT_CONFIG)
    prompt = build_system_prompt(config, "")
    assert "TOOL_CALL" not in prompt


def test_max_constants():
    assert MAX_HISTORY_TURNS == 10
    assert MAX_TOOL_LOOPS == 20


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


def test_parse_tool_calls_none():
    calls, text = _parse_tool_calls("Just a normal response.")
    assert calls == []
    assert text == "Just a normal response."


def test_parse_tool_calls_single():
    response = 'TOOL_CALL:{"name":"load_program","input":{"path":"pcb"}}\nLoading it now.'
    calls, text = _parse_tool_calls(response)
    assert len(calls) == 1
    assert calls[0]["name"] == "load_program"
    assert text == "Loading it now."


def test_parse_tool_calls_multiple():
    response = (
        'TOOL_CALL:{"name":"get_server_status","input":{}}\n'
        'TOOL_CALL:{"name":"load_program","input":{"path":"pcb"}}\n'
        "Done."
    )
    calls, text = _parse_tool_calls(response)
    assert len(calls) == 2
    assert text == "Done."


def test_check_cli():
    assert MopsLLM.check_cli() is True


def test_extract_personality_update(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = tmp_path / "config.json"
    llm = MopsLLM(config, config_path)
    text = "PERSONALITY_UPDATE:humor=90\nHumor 90, confirmed."
    result = llm._extract_personality_update(text)
    assert result == "Humor 90, confirmed."
    assert config["personality"]["humor"] == 90
