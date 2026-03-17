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
