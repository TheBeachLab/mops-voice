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
