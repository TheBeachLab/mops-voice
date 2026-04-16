"""LLM via Claude Code CLI or direct Anthropic API + persistent MCP for tools."""

import asyncio
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path

from mops_voice.personality import (
    adjust_personality,
    get_personality,
)

MAX_HISTORY_TURNS = 10
MAX_TOOL_LOOPS = 10

# Map short aliases to full model IDs for the API
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250514",
    "opus": "claude-opus-4-0-20250514",
}

SYSTEM_PROMPT_TEMPLATE = """\
You are {assistant_name}, a voice assistant for digital fabrication.
You address the user as {user_name}.
The user's home directory is {home_dir}. Always use full absolute paths, never use ~ in file paths.

PERSONALITY (apply this to EVERY response):
- Humor: {humor}% — {humor_desc}
- Sarcasm: {sarcasm}% — {sarcasm_desc}
- Honesty: {honesty}% — {honesty_desc}
Your personality is NOT optional. Even when executing tools, your spoken responses must reflect these traits.

CRITICAL RULES:
- Your responses will be spoken aloud through TTS. Write ONLY plain spoken text.
- NO markdown, NO bullet points, NO bold, NO headers, NO lists, NO emojis.
- Keep responses to 1-3 short sentences. Be brief like a real conversation.
- Never volunteer your personality settings unless specifically asked.
- Never format responses as documentation or instructions.
- Talk like a human, not a manual.

You control fabrication machines through MOPS tools.
- Pay close attention to what the user says. If they mention a filename, use it immediately. Do not ask for info they already gave you.
- Input comes from speech recognition. Interpret spoken filenames naturally: "dot" means ".", "dash" means "-", "underscore" means "_". Example: "logo dot PNG" → "logo.png", "my underscore file dot SVG" → "my_file.svg". If a file is not found, try common variations (spaces, underscores, dashes, case changes) before giving up.
- Do NOT repeat tool calls that already succeeded in previous turns (check conversation history).
- If a tool call fails, say so honestly. NEVER claim success when a tool returned an error.
- If a tool call SUCCEEDS, acknowledge it. NEVER claim tools are missing or unavailable when they just worked.
- When the user says to just "cut", "send", or finish — do NOT repeat the full setup workflow. Only do the remaining steps (connect device + send).
- Before using trigger_action, ALWAYS call get_program_state first to find the exact module names and button names. Use those exact names.
- To resize output to a specific physical size, use set_physical_size (NOT manual DPI calculation).

CUTTING WORKFLOW — follow these steps IN ORDER, never skip any:
1. launch_browser (open mods if not already open)
2. find_machine (find the right machine and matching program)
3. load_program (load the cutting/milling program for that machine)
4. load_file (load the user's file — REQUIRED before setting size)
5. set_physical_size (set desired width, height, and unit — e.g. 10, 10, "cm". PNG only.)
6. Set any cut parameters if requested (speed, depth, tool, etc.) using set_parameter or set_parameters.
7. Tell the user the job is ready and ask for confirmation to cut/send.

IMPORTANT RULES:
- set_physical_size ONLY works with PNG files. For vector formats (SVG, DXF, HPGL), dimensions come from the file itself.
- Do NOT send to the machine until the user explicitly says "cut", "send", or "go".

SENDING TO MACHINE (only when user says "cut"/"send"/"go"):
1. get_program_state — find the on/off switch connected to WebUSB and the WebUSB module itself.
2. Ensure the on/off switch is ON: set_parameter(module_name="on/off:MODULE_ID", parameter="on/off", value="true").
3. trigger_action on the WebUSB module — click "Get Device". MOPS auto-selects via CDP. Do NOT tell the user to select anything.
4. Find which module has the "calculate" button (usually has "raster", "path", or "distance" in its name, NOT the WebUSB module).
5. trigger_action — click "calculate". This calculates the toolpath AND sends to the connected machine.

Once the device is connected (step 3), it stays connected for the session. For subsequent sends, skip to step 4.

CRITICAL — READ CAREFULLY:
- On/off switches are CHECKBOXES, not buttons. Use set_parameter, NEVER trigger_action on an on/off module.
- There is NO separate "send" button. "calculate" BOTH calculates AND sends when a device is connected and the on/off switch is ON.
- NEVER tell the user to manually select a device. MOPS handles it automatically via CDP.
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

    humor = p["humor"]
    sarcasm = p["sarcasm"]
    honesty = p["honesty"]

    humor_desc = (
        "Crack jokes, be playful, tease the user." if humor > 60
        else "Occasional light humor." if humor > 30
        else "Keep it dry and professional."
    )
    sarcasm_desc = (
        "Be very sarcastic. Give the user a hard time. Roast gently." if sarcasm > 70
        else "Moderate sarcasm, witty remarks." if sarcasm > 40
        else "Straight and sincere."
    )
    honesty_desc = (
        "Be blunt, even if it stings." if honesty > 80
        else "Honest but diplomatic." if honesty > 50
        else "Soften the truth, be encouraging."
    )

    return SYSTEM_PROMPT_TEMPLATE.format(
        assistant_name=config["assistant_name"],
        user_name=config["user_name"],
        humor=humor,
        sarcasm=sarcasm,
        honesty=honesty,
        humor_desc=humor_desc,
        sarcasm_desc=sarcasm_desc,
        honesty_desc=honesty_desc,
        home_dir=str(Path.home()),
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
            "--setting-sources", "",
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

    # --- Direct Anthropic API path ---

    def _resolve_model(self) -> str:
        """Resolve model alias to full model ID."""
        model = self.config.get("claude_model", "haiku")
        return MODEL_ALIASES.get(model, model)

    def _get_api_key(self) -> str:
        """Get Anthropic API key from config or environment."""
        api_cfg = self.config.get("anthropic", {})
        return api_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    def _build_api_tools(self) -> list[dict]:
        """Convert MCP tools to Anthropic API tool format."""
        tools = []
        for tool in self._mcp_tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
            })
        # Add personality tools
        tools.append({
            "name": "adjust_personality",
            "description": "Adjust a personality dial (humor, sarcasm, or honesty) to a value 0-100",
            "input_schema": {
                "type": "object",
                "properties": {
                    "dial": {"type": "string", "description": "The personality dial to adjust"},
                    "value": {"type": "integer", "description": "New value (0-100)"},
                },
                "required": ["dial", "value"],
            },
        })
        tools.append({
            "name": "get_personality",
            "description": "Get current personality settings",
            "input_schema": {"type": "object", "properties": {}},
        })
        return tools

    def _history_to_api_messages(self) -> list[dict]:
        """Convert conversation history to Anthropic API message format."""
        messages = []
        for entry in self.history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        return messages

    async def _call_api(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Call Anthropic Messages API directly. Returns the response dict."""
        api_key = self._get_api_key()
        system_prompt = build_system_prompt(self.config, self._tool_descriptions)

        body = {
            "model": self._resolve_model(),
            "max_tokens": 300,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools

        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

        loop = asyncio.get_running_loop()
        resp_body = await loop.run_in_executor(None, self._do_api_request, req)
        return resp_body

    @staticmethod
    def _do_api_request(req) -> dict:
        """Synchronous HTTP request (runs in executor)."""
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                err = json.loads(body)
                msg = err.get("error", {}).get("message", body)
            except json.JSONDecodeError:
                msg = body
            return {"_error": True, "message": msg, "status": e.code}

    async def _chat_api(self, user_text: str, on_tool_call=None) -> str:
        """Chat via direct Anthropic API with native tool_use."""
        self._prune_history()

        messages = self._history_to_api_messages()
        messages.append({"role": "user", "content": user_text})

        tools = self._build_api_tools() if self._mcp_tools else None
        tool_summaries = []

        for _iteration in range(MAX_TOOL_LOOPS):
            response = await self._call_api(messages, tools)

            if not response:
                return "I'm having trouble thinking right now, Fran. Try again."

            if response.get("_error"):
                # API failed — fall back to CLI with context about the error
                fallback_prompt = (
                    f"(System note: the Anthropic API call failed with: "
                    f"{response.get('message', 'unknown error')}. "
                    f"Briefly tell Fran about this issue in a conversational way, "
                    f"then try to answer their original question.)\n\n"
                    f"Fran said: {user_text}"
                )
                return await self._chat_cli(fallback_prompt, on_tool_call)

            if "content" not in response:
                return "I'm having trouble thinking right now, Fran. Try again."

            # Collect text and tool_use blocks from response
            text_parts = []
            tool_uses = []
            for block in response["content"]:
                if block["type"] == "text":
                    text_parts.append(block["text"])
                elif block["type"] == "tool_use":
                    tool_uses.append(block)

            spoken_text = " ".join(text_parts).strip()

            if not tool_uses:
                spoken_text = self._extract_personality_update(spoken_text)
                # Save history with tool summary so future turns know what happened
                assistant_record = spoken_text
                if tool_summaries:
                    assistant_record = (
                        "[Tools used: " + "; ".join(tool_summaries) + "] "
                        + spoken_text
                    )
                self.history.append({"user": user_text, "assistant": assistant_record})
                return spoken_text

            # Append the full assistant response to messages
            messages.append({"role": "assistant", "content": response["content"]})

            # Execute tools and build tool_result message
            tool_results = []
            for tu in tool_uses:
                name = tu["name"]
                input_data = tu.get("input", {})
                result = await self._execute_tool(name, input_data)

                if on_tool_call:
                    summary = result[:80] + "..." if len(result) > 80 else result
                    on_tool_call(name, summary)

                # Track for history
                short = result[:60] + "..." if len(result) > 60 else result
                tool_summaries.append(f"{name} → {short}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        return "I ran too many tools there, Fran. Let me try again."

    async def chat(self, user_text: str, on_tool_call=None) -> str:
        """Send user text, execute tool calls, return final spoken response."""
        if self.config.get("llm_engine") == "api":
            return await self._chat_api(user_text, on_tool_call)
        return await self._chat_cli(user_text, on_tool_call)

    async def _chat_cli(self, user_text: str, on_tool_call=None) -> str:
        """Chat via Claude CLI (original path)."""
        self._prune_history()

        history_context = _format_history(self.history)
        if history_context:
            prompt = f"{history_context}\nUser: {user_text}"
        else:
            prompt = user_text

        tool_summaries = []

        for _iteration in range(MAX_TOOL_LOOPS):
            raw_response = await self._call_claude(prompt)
            if not raw_response:
                return "I'm having trouble thinking right now, Fran. Try again."

            tool_calls, spoken_text = _parse_tool_calls(raw_response)

            if not tool_calls:
                # No tools — just a spoken response
                spoken_text = self._extract_personality_update(spoken_text)
                assistant_record = spoken_text
                if tool_summaries:
                    assistant_record = (
                        "[Tools used: " + "; ".join(tool_summaries) + "] "
                        + spoken_text
                    )
                self.history.append({"user": user_text, "assistant": assistant_record})
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

                short = result[:60] + "..." if len(result) > 60 else result
                tool_summaries.append(f"{name} → {short}")
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
