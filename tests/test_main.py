import argparse

from mops_voice.main import parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.headless is False
    assert args.mods_url is None
    assert args.whisper_model is None


def test_parse_args_headless():
    args = parse_args(["--headless"])
    assert args.headless is True


def test_parse_args_mods_url():
    args = parse_args(["--mods-url", "http://localhost:8080"])
    assert args.mods_url == "http://localhost:8080"


def test_parse_args_whisper_model():
    args = parse_args(["--whisper-model", "small.en"])
    assert args.whisper_model == "small.en"
