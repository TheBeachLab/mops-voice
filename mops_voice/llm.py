"""LLM via Claude Code CLI + persistent MCP connection for tools."""

import asyncio
import json
import re
import shutil
from pathlib import Path

from mops_voice.personality import (
    adjust_personality,
    get_personality,
)

MAX_HISTORY_TURNS = 10
MAX_TOOL_LOOPS = 10

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
Only report your settings when explicitly asked.

{tool_instructions}\
"""

TOOL_INSTRUCTIONS_TEMPLATE = """
AVAILABLE TOOLS:
{tool_list}

To use a tool, respond with EXACTLY this format on its own line:
TOOL_CALL:{{"name":"tool_name","input":{{...}}}}

You can make multiple tool calls, one per line. After tool calls, add your spoken response.
Wait for tool results before making follow-up calls that depend on them.
"""


def build_system_prompt(config: dict, tool_descriptions: str = "") -> str:
    """Build system prompt with current personality settings and tools."""
    p = config["personality"]
    if tool_descriptions:
        tool_instructions = TOOL_INSTRUCTIONS_TEMPLATE.format(tool_list=tool_descriptions)
    else:
        tool_instructions = ""
    return SYSTEM_PROMPT_TEMPLATE.format(
        assistant_name=config["assistant_name"],
        user_name=config["user_name"],
        humor=p["humor"],
        sarcasm=p["sarcasm"],
        honesty=p["honesty"],
        tool_instructions=tool_instructions,
    )


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


# Regex to extract TOOL_CALL lines
_TOOL_CALL_RE = re.compile(r'^TOOL_CALL:(.+)$', re.MULTILINE)


def _parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """Extract TOOL_CALL directives and remaining spoken text."""
    tool_calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            tool_calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    clean_text = _TOOL_CALL_RE.sub("", text).strip()
    return tool_calls, clean_text


class MopsLLM:
    """Claude CLI for LLM + persistent MCP for tool execution."""

    def __init__(self, config: dict, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.history: list[dict] = []
        self._mcp_session = None
        self._mcp_tools: list[dict] = []
        self._mcp_cm = None
        self._mcp_session_cm = None
        self._tool_descriptions = ""

    @staticmethod
    def check_cli() -> bool:
        """Check if the claude CLI is available."""
        return shutil.which("claude") is not None

    async def connect_mcp(self, server_path: str, extra_args: list[str] | None = None) -> bool:
        """Start persistent MCP connection to MOPS server. Returns True on success."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            args = [server_path] + (extra_args or [])
            server_params = StdioServerParameters(command="node", args=args)

            self._mcp_cm = stdio_client(server_params)
            read, write = await self._mcp_cm.__aenter__()
            self._mcp_session_cm = ClientSession(read, write)
            self._mcp_session = await self._mcp_session_cm.__aenter__()
            await self._mcp_session.initialize()

            # Discover tools and build descriptions for system prompt
            tools_result = await self._mcp_session.list_tools()
            self._mcp_tools = tools_result.tools
            self._tool_descriptions = self._format_tool_descriptions()
            return True
        except Exception as e:
            self._mcp_session = None
            self._mcp_tools = []
            return False

    def _format_tool_descriptions(self) -> str:
        """Format MCP tools as text for the system prompt."""
        lines = []
        for tool in self._mcp_tools:
            desc = tool.description or "No description"
            # Summarize input schema
            schema = tool.inputSchema or {}
            props = schema.get("properties", {})
            params = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                params.append(f"  - {pname} ({ptype}): {pdesc}")
            param_str = "\n".join(params) if params else "  (no parameters)"
            lines.append(f"- {tool.name}: {desc}\n{param_str}")
        return "\n".join(lines)

    async def disconnect_mcp(self):
        """Shut down persistent MCP connection."""
        if self._mcp_session_cm:
            try:
                await self._mcp_session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._mcp_cm:
            try:
                await self._mcp_cm.__aexit__(None, None, None)
            except Exception:
                pass

    async def _execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool via MCP. Returns result string."""
        # Handle personality tools locally
        if name == "adjust_personality":
            result = adjust_personality(
                self.config, self.config_path,
                input_data.get("dial", ""), input_data.get("value", 0)
            )
            return json.dumps(result) if isinstance(result, dict) else result
        if name == "get_personality":
            return json.dumps(get_personality(self.config))

        # Execute via MCP
        if not self._mcp_session:
            return "Tool unavailable, MCP server not connected."
        try:
            result = await self._mcp_session.call_tool(name, input_data)
            return result.content[0].text if result.content else "OK"
        except Exception as e:
            return f"Tool error: {e}"

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
                    adjust_personality(
                        self.config, self.config_path, dial.strip(), int(value.strip())
                    )
                except (ValueError, IndexError):
                    pass
            else:
                clean_lines.append(line)
        return "\n".join(clean_lines).strip()

    async def _call_claude(self, prompt: str) -> str:
        """Call claude CLI with prompt, return raw response text."""
        system_prompt = build_system_prompt(self.config, self._tool_descriptions)

        cmd = [
            "claude",
            "-p", prompt,
            "--system-prompt", system_prompt,
            "--output-format", "json",
            "--no-session-persistence",
            "--model", self.config.get("claude_model", "sonnet"),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode("utf-8").strip()

        if proc.returncode != 0 or not output:
            return ""

        try:
            result = json.loads(output)
            return result.get("result", "")
        except json.JSONDecodeError:
            return output

    async def chat(self, user_text: str, on_tool_call=None) -> str:
        """Send user text, execute tool calls, return final spoken response."""
        self._prune_history()

        history_context = _format_history(self.history)
        if history_context:
            prompt = f"{history_context}\nUser: {user_text}"
        else:
            prompt = user_text

        for _iteration in range(MAX_TOOL_LOOPS):
            raw_response = await self._call_claude(prompt)
            if not raw_response:
                return "I'm having trouble thinking right now, Fran. Try again."

            tool_calls, spoken_text = _parse_tool_calls(raw_response)

            if not tool_calls:
                # No tools — just a spoken response
                spoken_text = self._extract_personality_update(spoken_text)
                self.history.append({"user": user_text, "assistant": spoken_text})
                return spoken_text

            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                input_data = tc.get("input", {})
                result = await self._execute_tool(name, input_data)

                if on_tool_call:
                    summary = result[:80] + "..." if len(result) > 80 else result
                    on_tool_call(name, summary)

                tool_results.append(f"Tool {name} result: {result}")

            # Feed results back to Claude for follow-up or final response
            results_text = "\n".join(tool_results)
            prompt = (
                f"{history_context}\nUser: {user_text}\n\n"
                f"You previously responded: {spoken_text}\n"
                f"Tool results:\n{results_text}\n\n"
                "Now give your final spoken response to the user based on these results."
            )

        return "I ran too many tools there, Fran. Let me try again."
