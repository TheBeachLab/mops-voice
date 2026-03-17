"""LLM integration via Claude Code CLI with MCP support."""

import asyncio
import json
import shutil
from pathlib import Path

from mops_voice.personality import (
    adjust_personality,
    get_personality,
)

MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT_TEMPLATE = """\
You are {assistant_name}, a voice assistant for digital fabrication.
You address the user as {user_name}.
Personality settings: humor={humor}%, sarcasm={sarcasm}%, honesty={honesty}%.

CRITICAL RULES:
- Your responses will be spoken aloud through TTS. Write ONLY plain spoken text.
- NO markdown, NO bullet points, NO bold, NO headers, NO lists, NO emojis.
- Keep responses to 1-3 short sentences. Be brief like a real conversation.
- Never volunteer your personality settings unless specifically asked.
- Never format responses as documentation or instructions.
- Talk like a human, not a manual.

You control fabrication machines through MOPS tools.
When a personality setting is changed, respond with PERSONALITY_UPDATE:dial=value on its own line \
(e.g. PERSONALITY_UPDATE:humor=90) followed by a brief spoken confirmation.
Only report your settings when explicitly asked.\
"""


def build_system_prompt(config: dict) -> str:
    """Build system prompt with current personality settings."""
    p = config["personality"]
    return SYSTEM_PROMPT_TEMPLATE.format(
        assistant_name=config["assistant_name"],
        user_name=config["user_name"],
        humor=p["humor"],
        sarcasm=p["sarcasm"],
        honesty=p["honesty"],
    )


def build_mcp_config(server_path: str, extra_args: list[str] | None = None) -> dict:
    """Build MCP config JSON for the claude CLI."""
    args = [server_path] + (extra_args or [])
    return {
        "mcpServers": {
            "mops": {
                "command": "node",
                "args": args,
            }
        }
    }


def _format_history(history: list[dict]) -> str:
    """Format conversation history as context for the prompt."""
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for entry in history:
        lines.append(f"User: {entry['user']}")
        lines.append(f"MOPS: {entry['assistant']}")
    lines.append("")
    return "\n".join(lines)


class MopsLLM:
    """Manages Claude CLI calls with MCP tools."""

    def __init__(self, config: dict, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.history: list[dict] = []
        self.mcp_config_path: Path | None = None
        self._mcp_available = False

    @staticmethod
    def check_cli() -> bool:
        """Check if the claude CLI is available."""
        return shutil.which("claude") is not None

    def setup_mcp(self, server_path: str, extra_args: list[str] | None = None) -> None:
        """Write MCP config file for the claude CLI."""
        from mops_voice.config import CONFIG_DIR

        mcp_config = build_mcp_config(server_path, extra_args)
        self.mcp_config_path = CONFIG_DIR / "mcp.json"
        self.mcp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mcp_config_path, "w") as f:
            json.dump(mcp_config, f, indent=2)
        self._mcp_available = True

    def _prune_history(self):
        """Keep last N turns of conversation history."""
        if len(self.history) > MAX_HISTORY_TURNS:
            self.history = self.history[-MAX_HISTORY_TURNS:]

    def _extract_personality_update(self, text: str) -> str:
        """Extract and apply PERSONALITY_UPDATE directives from response text."""
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            if "PERSONALITY_UPDATE:" in line:
                try:
                    directive = line.split("PERSONALITY_UPDATE:")[1].strip()
                    dial, value = directive.split("=")
                    dial = dial.strip()
                    value = int(value.strip())
                    adjust_personality(
                        self.config, self.config_path, dial, value
                    )
                except (ValueError, IndexError):
                    pass
            else:
                clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    _ACKNOWLEDGMENTS = [
        "On it.",
        "Got it.",
        "Sure thing.",
        "Right away.",
        "Let me handle that.",
        "Working on it.",
        "Copy that.",
        "Roger.",
    ]

    def acknowledge(self) -> str:
        """Instant canned acknowledgment. No API call needed."""
        import random
        return random.choice(self._ACKNOWLEDGMENTS)

    async def chat(self, user_text: str, on_tool_call=None) -> str:
        """Send user text via claude CLI, return response text.

        on_tool_call: optional callback(tool_name, result_summary) for verbose output.
        Uses asyncio.create_subprocess_exec for safe argument passing (no shell).
        """
        self._prune_history()

        # Build the prompt with conversation history
        history_context = _format_history(self.history)
        if history_context:
            full_prompt = f"{history_context}\nUser: {user_text}"
        else:
            full_prompt = user_text

        system_prompt = build_system_prompt(self.config)

        # Build claude CLI command as a list (no shell interpolation)
        cmd = [
            "claude",
            "-p", full_prompt,
            "--system-prompt", system_prompt,
            "--output-format", "json",
            "--no-session-persistence",
            "--model", self.config.get("claude_model", "sonnet"),
        ]

        if self._mcp_available and self.mcp_config_path:
            cmd.extend(["--mcp-config", str(self.mcp_config_path)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()
            output = stdout.decode("utf-8").strip()

            if proc.returncode != 0 or not output:
                return "I'm having trouble thinking right now, Fran. Try again."

            # Parse JSON result
            try:
                result = json.loads(output)
                response_text = result.get("result", "")
            except json.JSONDecodeError:
                # If not JSON, treat as plain text
                response_text = output

            if not response_text:
                return "I'm having trouble thinking right now, Fran. Try again."

            # Handle personality updates in the response
            response_text = self._extract_personality_update(response_text)

            # Update conversation history
            self.history.append({
                "user": user_text,
                "assistant": response_text,
            })

            return response_text

        except FileNotFoundError:
            return "Claude CLI not found. Make sure 'claude' is in your PATH."
        except Exception:
            return "I'm having trouble thinking right now, Fran. Try again."
