"""LLM via Claude Code CLI or direct Anthropic API + persistent MCP for tools."""

import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

import anthropic

from mops_voice.personality import (
    adjust_personality,
    get_personality,
)
from mops_voice.runtime_settings import (
    get_voice_settings,
    runtime_settings_tool_schemas,
    set_image_roast,
    set_llm_engine,
    set_voxtral_voice,
)
from mops_voice.logging_setup import mcp_stderr_file, redact
from mops_voice.image_attach import maybe_build_attachment

log = logging.getLogger("mops_voice.llm")

MAX_HISTORY_TURNS = 10
# A full GX-24 setup-plus-send uses ~9 tool-loop iterations on the happy
# path (4 cutting phases + 5 sending steps). 20 gives room for parameter
# probes and error recovery without aborting mid-workflow.
MAX_TOOL_LOOPS = 20

# Map short aliases to bare model IDs. Bare aliases always resolve to a
# supported version; date-suffixed variants can silently break on retirement.
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-5",
    "opus": "claude-opus-4-5",
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
- Your text streams to speech sentence-by-sentence as you type, and speech is SLOWER than the tools. Every word costs real wall-clock time and the user hears narration lag the action. Be brutally terse.
- Across a FULL turn (which may span several tool-use rounds), emit AT MOST 2-3 short sentences TOTAL: one opening acknowledgment, one closing status, plus optionally one note IF something unusual happens (an error, a surprise).
- In intermediate responses (ones that end with tool_use), emit AT MOST ONE very short sentence (5-10 words), or no sentence at all. The runtime blocks tool execution until that sentence finishes playing, so extra words directly slow the machine.
- NEVER preview upcoming steps ("now let me...", "next I'll...", "I'll load the program and your file"). NEVER restate at the end what you already said or what the tools already did.
- Keep each sentence short — under 15 words when you can.
- NEVER tell the user to "select", "pick", or "choose" a device in the browser. MOPS auto-selects via CDP when a device is physically plugged in. If no device shows up, the fix is for the user to PLUG IT IN, not to click anything in the browser.
- Never volunteer your personality settings unless specifically asked.
- Never format responses as documentation or instructions.
- Talk like a human, not a manual.
- If a tool_result contains an attached image, you have just glimpsed the file the user is about to cut/mill/print. Make ONE short, in-character quip about what you see (one sentence, under 12 words) before continuing the workflow. Examples: "Oh great, another logo." / "Bold choice, Fran." / "Whoever drew this should be ashamed." Skip the quip if the image is empty/unreadable. Never describe the image clinically; this is a roast, not narration.
- The user can ask you mid-session to switch the Voxtral voice (set_voxtral_voice), change the image-roast probability (set_image_roast), or switch the LLM engine between cli/api (set_llm_engine). When they say things like "be sarcastic" or "use the angry voice", call set_voxtral_voice with the matching voice id. When they say "roast more" / "roast less" / "stop roasting", adjust set_image_roast (full off=0, occasional=0.3, every cut=1). API keys are NOT changeable via voice — if asked, tell them to edit ~/.mops-voice/config.json.

You control fabrication machines through MOPS tools.
- Pay close attention to what the user says. If they mention a filename, use it immediately. Do not ask for info they already gave you.
- Input comes from speech recognition. Interpret spoken filenames naturally: "dot" means ".", "dash" means "-", "underscore" means "_". Example: "logo dot PNG" → "logo.png", "my underscore file dot SVG" → "my_file.svg". If a file is not found, try common variations (spaces, underscores, dashes, case changes) before giving up.
- Do NOT repeat tool calls that already succeeded in previous turns (check conversation history).
- If a tool call fails, say so honestly. NEVER claim success when a tool returned an error.
- If a tool call SUCCEEDS, acknowledge it. NEVER claim tools are missing or unavailable when they just worked.
- When the user says to just "cut", "send", or finish — do NOT repeat the full setup workflow. Only do the remaining steps (connect device + send).
- Before calling trigger_action OR set_parameter with a module_name you haven't seen in the CURRENT turn's tool results, ALWAYS call get_program_state first to find exact module, parameter, and button names. Don't guess — module names and parameter labels differ per machine (speed lives on the "Roland GX/GS-24 Relative" module, not "cut raster").
- If a prior turn already did part of the workflow (device connected, program loaded, speed set) and is visible in conversation history, DO NOT redo it. Jump to the remaining step. "Skip to step 3" means skip to step 3, don't re-run Phase 1 "just to check".
- To resize output to a specific physical size, use set_physical_size (NOT manual DPI calculation).

CUTTING WORKFLOW — run these phases in order. Parallelize WITHIN a phase; never merge tool_use blocks across phases (later phases depend on side effects of earlier ones).

PHASE A — BOOT (parallel): launch_browser + find_machine
- find_machine returns a JSON array sorted by relevanceScore. Pick scored[0], then choose the matchingPrograms entry whose path most specifically matches the machine (e.g. for "Roland GX-24" prefer a path containing "GX-24" over a generic "roland" path). Pass that exact path to load_program in Phase B.

PHASE B — LOAD PROGRAM (alone): load_program with the selected path.

PHASE C — LOAD FILE (alone): load_file. Must fully return before any size setting — set_physical_size reads state that load_file writes.

PHASE D — SIZE & PARAMS (parallel): set_physical_size + set_parameter(s) for any requested cut settings (speed, depth, tool).
- set_physical_size ONLY works with PNG. Vector formats (SVG, DXF, HPGL) carry their own dimensions.

Then tell the user the job is ready and ask for confirmation to cut/send.

Do NOT send to the machine until the user explicitly says "cut", "send", or "go".

SENDING TO MACHINE (only when user says "cut"/"send"/"go"):
Each numbered step is its own response unless it says "parallel". Do NOT collapse two numbered steps into one response — each step's result is a precondition for the next.

STEP 1 — SETUP (parallel, in ONE response):
- get_program_state (to find the on/off switch, WebUSB module ID, and the toolpath module with "calculate")
- set_parameter(module_name="on/off:MODULE_ID", parameter="", value="true") — on/off is a checkbox; leave `parameter` BLANK, not "on/off".
- trigger_action(module_name="WebUSB", action="Get Device")

STEP 2 — VERIFY DEVICE (alone, MANDATORY, no exceptions):
- list_devices
If the result is empty or does not show the target machine, STOP. Tell the user the cutter is not plugged in and ask them to plug it into USB. Do NOT speak before list_devices returns. Do NOT skip this step. Do NOT move on to step 3 without a matching device in the result.

STEP 3 — CALCULATE (alone):
- trigger_action on the toolpath module — click "calculate". Do not include any other tool_use block in this response.

STEP 4 — CHECK SEND BUTTON (alone):
- get_program_state — confirm the WebUSB module's send button is now labeled "send file" (it was "waiting for file" before calculate).

STEP 5 — SEND (alone):
- trigger_action(module_name="WebUSB", action="send file")
If `clicked` is anything other than "send file" (e.g. "waiting for file"), the send FAILED. Tell the user honestly — do NOT claim the machine is running.

Once a device is connected (step 2), it stays connected for the session. For subsequent sends, skip to step 3.

CRITICAL — READ CAREFULLY:
- calculate MUST run AFTER the on/off is ON and "Get Device" has been clicked AND list_devices has confirmed a device. Never parallelize calculate with setup tools.
- trigger_action returning {{success: true}} only confirms a button was clicked. ALWAYS compare the `clicked` field to the action you requested — if they differ, the operation failed.
- On/off switches are CHECKBOXES, not buttons. Use set_parameter, NEVER trigger_action on an on/off module.
- The WebUSB send button label toggles between "waiting for file" and "send file". After calculate it should flip to "send file"; if it doesn't, something is wrong.
- NEVER claim the machine is cutting/running on the basis of {{success: true}} alone. Verify via list_devices AND via the post-click button label.

MODS FILE OUTPUT — read before the user says "save the file", "export", etc:
- `save_program` dumps the program's JSON CONFIG, not the cut file. Do NOT call save_program when the user asks to save/export the cut file.
- The cut file is produced by the `save file` module, which auto-downloads to the user's Downloads folder when a toolpath FLOWS THROUGH IT. The trigger is calculate on the toolpath module — NOT a button on the save module itself (`save file` has no button to click).
- If the user asks to save/export the cut file: (1) if no toolpath has been calculated yet this turn, run STEP 3 (calculate) first; (2) mention that the file should land in ~/Downloads. Don't promise a specific filename unless a tool response gave you one; "postMessage.png.camm" style names are mods internals, not user-friendly.
- The `calculate` button lives on the TOOLPATH module (usually named "cut raster", "mill raster", etc.) — NOT on upstream modules like "distance transform", "edge detect", "offset", "image threshold" (those only have a "view" button). If you don't have the exact toolpath module name in recent context, call get_program_state first.
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


def _block_to_input_dict(block) -> dict:
    """Convert an SDK content block back to the minimal dict the Messages API
    accepts on input. `block.model_dump()` also includes response-only fields
    (`parsed_output`, `citations`, `caller`) that trigger a 400 on echo."""
    t = block.type
    if t == "text":
        return {"type": "text", "text": block.text}
    if t == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if t == "thinking":
        out = {"type": "thinking", "thinking": block.thinking}
        sig = getattr(block, "signature", None)
        if sig is not None:
            out["signature"] = sig
        return out
    # Unknown block type — strip known response-only fields as a fallback.
    d = block.model_dump(exclude_none=True)
    for key in ("parsed_output", "citations", "caller"):
        d.pop(key, None)
    return d


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
        self._mcp_stderr_fh = None
        self._tool_descriptions = ""
        self._anthropic: anthropic.AsyncAnthropic | None = None

    def _get_anthropic_client(self) -> anthropic.AsyncAnthropic:
        """Lazily construct the async Anthropic client."""
        if self._anthropic is None:
            self._anthropic = anthropic.AsyncAnthropic(api_key=self._get_api_key())
        return self._anthropic

    @staticmethod
    def check_cli() -> bool:
        """Check if the claude CLI is available."""
        return shutil.which("claude") is not None

    async def connect_mcp(self, command: str, args: list[str] | None = None) -> bool:
        """Start persistent MCP connection to MOPS server. Returns True on success."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            args = list(args or [])
            log.info("connecting MCP: %s %s", command, " ".join(args))
            server_params = StdioServerParameters(command=command, args=args)

            # Route the MCP subprocess's stderr into our log file so every
            # [mops] line from server.js/browser.js is captured alongside
            # our own Python logs.
            self._mcp_stderr_fh = mcp_stderr_file()
            self._mcp_cm = stdio_client(server_params, errlog=self._mcp_stderr_fh)
            read, write = await self._mcp_cm.__aenter__()
            self._mcp_session_cm = ClientSession(read, write)
            self._mcp_session = await self._mcp_session_cm.__aenter__()
            await self._mcp_session.initialize()

            tools_result = await self._mcp_session.list_tools()
            self._mcp_tools = tools_result.tools
            self._tool_descriptions = self._format_tool_descriptions()
            log.info(
                "MCP connected: %d tools (%s)",
                len(self._mcp_tools),
                ", ".join(t.name for t in self._mcp_tools),
            )
            return True
        except Exception:
            log.exception("MCP connection failed")
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
                log.exception("MCP session close failed")
        if self._mcp_cm:
            try:
                await self._mcp_cm.__aexit__(None, None, None)
            except Exception:
                log.exception("MCP transport close failed")
        if self._mcp_stderr_fh:
            try:
                self._mcp_stderr_fh.close()
            except Exception:
                pass
            self._mcp_stderr_fh = None

    def _maybe_attach_image(self, tool_use, result_text: str):
        """Wrap a tool_result in multimodal content if a PNG was loaded.

        Looks at file_path on load_file/setup_cut tool_use blocks. If the
        file is a PNG and the configured probability gate passes, returns
        a list of [text, image] content blocks suitable for tool_result.
        Otherwise returns the original text string.
        """
        if tool_use.name not in ("load_file", "setup_cut"):
            return result_text
        file_path = (tool_use.input or {}).get("file_path")
        if not file_path:
            return result_text
        cfg = self.config.get("image_roast") or {}
        probability = float(cfg.get("probability", 0))
        max_dim = int(cfg.get("max_dim", 512))
        block = maybe_build_attachment(file_path, probability, max_dim=max_dim)
        if block is None:
            return result_text
        log.info("attaching image preview from %s", file_path)
        return [{"type": "text", "text": result_text}, block]

    async def _execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool via MCP. Returns result string."""
        t0 = time.monotonic()
        log.info("tool → %s(%s)", name, json.dumps(input_data, default=str)[:400])

        # Handle personality tools locally
        if name == "adjust_personality":
            result = adjust_personality(
                self.config, self.config_path,
                input_data.get("dial", ""), input_data.get("value", 0)
            )
            out = json.dumps(result) if isinstance(result, dict) else result
            log.info("tool ← %s %.2fs: %s", name, time.monotonic() - t0, out[:300])
            return out
        if name == "get_personality":
            out = json.dumps(get_personality(self.config))
            log.info("tool ← %s %.2fs: %s", name, time.monotonic() - t0, out[:300])
            return out
        if name == "set_voxtral_voice":
            result = set_voxtral_voice(self.config, self.config_path, input_data.get("voice", ""))
            out = json.dumps(result) if isinstance(result, dict) else result
            log.info("tool ← %s %.2fs: %s", name, time.monotonic() - t0, out[:300])
            return out
        if name == "set_image_roast":
            result = set_image_roast(self.config, self.config_path, input_data.get("probability", 0))
            out = json.dumps(result) if isinstance(result, dict) else result
            log.info("tool ← %s %.2fs: %s", name, time.monotonic() - t0, out[:300])
            return out
        if name == "set_llm_engine":
            result = set_llm_engine(self.config, self.config_path, input_data.get("engine", ""))
            out = json.dumps(result) if isinstance(result, dict) else result
            log.info("tool ← %s %.2fs: %s", name, time.monotonic() - t0, out[:300])
            return out
        if name == "get_voice_settings":
            out = json.dumps(get_voice_settings(self.config))
            log.info("tool ← %s %.2fs: %s", name, time.monotonic() - t0, out[:300])
            return out

        # Execute via MCP
        if not self._mcp_session:
            log.warning("tool ✖ %s: MCP not connected", name)
            return "Tool unavailable, MCP server not connected."
        try:
            result = await self._mcp_session.call_tool(name, input_data)
            out = result.content[0].text if result.content else "OK"
            is_error = getattr(result, "isError", False)
            tag = "ERR" if is_error else "OK"
            log.info(
                "tool ← %s [%s] %.2fs: %s",
                name, tag, time.monotonic() - t0, out[:400].replace("\n", " "),
            )
            if len(out) > 400:
                log.debug("tool %s full result: %s", name, out)
            return out
        except Exception as e:
            log.exception("tool ✖ %s %.2fs", name, time.monotonic() - t0)
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

        log.debug("claude CLI prompt: %s", prompt[:600])
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        elapsed = time.monotonic() - t0
        output = stdout.decode("utf-8").strip()

        if proc.returncode != 0:
            log.error(
                "claude CLI exited %s in %.2fs: stderr=%s",
                proc.returncode, elapsed,
                stderr.decode("utf-8", errors="replace")[:600],
            )
            return ""
        if not output:
            log.warning("claude CLI returned empty output in %.2fs", elapsed)
            return ""

        try:
            result = json.loads(output)
            text = result.get("result", "")
            log.debug("claude CLI response (%.2fs): %s", elapsed, text[:600])
            return text
        except json.JSONDecodeError:
            log.debug("claude CLI raw response (%.2fs): %s", elapsed, output[:600])
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
        tools.extend(runtime_settings_tool_schemas())
        return tools

    def _history_to_api_messages(self) -> list[dict]:
        """Convert conversation history to Anthropic API message format."""
        messages = []
        for entry in self.history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        return messages

    async def _stream_api(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text_chunk=None,
    ):
        """Stream one Anthropic API call. Returns the final `Message` object.

        While streaming, forwards text deltas to `on_text_chunk` as they
        arrive (only for `text`-type content blocks — `tool_use` arg tokens
        are intentionally not streamed to TTS). Tool-use blocks are collected
        in the final message so the caller can dispatch them.
        """
        client = self._get_anthropic_client()
        system_prompt = build_system_prompt(self.config, self._tool_descriptions)

        # Prompt caching: marker on the last system block caches tools + system.
        kwargs = {
            "model": self._resolve_model(),
            "max_tokens": 300,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        log.debug(
            "anthropic stream → model=%s msgs=%d tools=%d key=%s",
            kwargs["model"], len(messages), len(tools or []),
            redact(self._get_api_key()),
        )
        log.debug("anthropic stream messages: %s", json.dumps(messages, default=str)[:1200])

        t0 = time.monotonic()
        current_block_type: str | None = None

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    current_block_type = getattr(event.content_block, "type", None)
                elif etype == "content_block_stop":
                    current_block_type = None
                elif etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if (
                        on_text_chunk
                        and current_block_type == "text"
                        and getattr(delta, "type", None) == "text_delta"
                    ):
                        try:
                            on_text_chunk(delta.text)
                        except Exception:
                            log.exception("on_text_chunk raised")
            final = await stream.get_final_message()

        elapsed = time.monotonic() - t0
        usage = final.usage
        # Surface cache metrics at INFO so they're easy to grep and trend.
        # `cache_read` > 0 = cache hit (cheap+fast);
        # `cache_write` > 0 = first-seen prefix being stored (~1.25×);
        # `input` is the uncached remainder only.
        log.info(
            "anthropic API ← %.2fs stop=%s input=%d cache_read=%d cache_write=%d output=%d",
            elapsed, final.stop_reason,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )
        return final

    async def _chat_api(
        self,
        user_text: str,
        on_tool_call=None,
        on_text_chunk=None,
        wait_for_speech=None,
    ) -> str:
        """Chat via direct Anthropic API with native tool_use + streaming."""
        self._prune_history()

        messages = self._history_to_api_messages()
        messages.append({"role": "user", "content": user_text})

        tools = self._build_api_tools() if self._mcp_tools else None
        tool_summaries = []

        for _iteration in range(MAX_TOOL_LOOPS):
            try:
                final = await self._stream_api(messages, tools, on_text_chunk)
            except anthropic.APIError as e:
                log.exception("anthropic API error, falling back to CLI")
                fallback_prompt = (
                    f"(System note: the Anthropic API call failed with: {e}. "
                    f"Briefly tell Fran about this issue in a conversational way, "
                    f"then try to answer their original question.)\n\n"
                    f"Fran said: {user_text}"
                )
                return await self._chat_cli(fallback_prompt, on_tool_call)

            text_parts: list[str] = []
            tool_uses = []
            for block in final.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            spoken_text = " ".join(text_parts).strip()

            if not tool_uses:
                spoken_text = self._extract_personality_update(spoken_text)
                assistant_record = spoken_text
                if tool_summaries:
                    assistant_record = (
                        "[Tools used: " + "; ".join(tool_summaries) + "] "
                        + spoken_text
                    )
                self.history.append({"user": user_text, "assistant": assistant_record})
                return spoken_text

            # Append the assistant response (serialize SDK blocks to minimal
            # input-accepted dicts — see `_block_to_input_dict`).
            messages.append({
                "role": "assistant",
                "content": [_block_to_input_dict(b) for b in final.content],
            })

            # PACING GATE — wait for the narration of this iteration to
            # finish playing before we fire the tools it described. Keeps
            # voice ahead of action. No-op on the first response if the
            # speech queue is already empty.
            if wait_for_speech is not None:
                gate_t0 = time.monotonic()
                drained = await wait_for_speech()
                gate_wait = time.monotonic() - gate_t0
                if gate_wait > 0.05:
                    log.info("pacing gate: waited %.2fs for speech (drained=%s)", gate_wait, drained)

            # Execute tool_use blocks concurrently. Claude decides what to
            # parallelize — the cutting workflow prompt keeps write ops
            # serial; status/info queries often arrive as a parallel batch.
            # `gather` preserves argument order, so we can still report
            # callbacks and build tool_results in emission order.
            if len(tool_uses) > 1:
                log.info(
                    "executing %d tools in parallel: %s",
                    len(tool_uses),
                    [tu.name for tu in tool_uses],
                )
            results = await asyncio.gather(*(
                self._execute_tool(tu.name, tu.input or {}) for tu in tool_uses
            ))

            tool_results = []
            for tu, result in zip(tool_uses, results):
                if on_tool_call:
                    summary = result[:80] + "..." if len(result) > 80 else result
                    on_tool_call(tu.name, summary)

                short = result[:60] + "..." if len(result) > 60 else result
                tool_summaries.append(f"{tu.name} → {short}")

                content = self._maybe_attach_image(tu, result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": content,
                })

            messages.append({"role": "user", "content": tool_results})

        return "I ran too many tools there, Fran. Let me try again."

    async def chat(
        self,
        user_text: str,
        on_tool_call=None,
        on_text_chunk=None,
        wait_for_speech=None,
    ) -> str:
        """Send user text, execute tool calls, return final spoken response.

        `on_text_chunk(delta: str)` — optional, fires with text deltas as the
        assistant response streams in (API engine only). For the CLI engine,
        non-streaming, the full response is forwarded as one chunk at the end.

        `wait_for_speech() -> awaitable` — optional pacing gate. If provided,
        it's awaited in the tool loop right after a streaming response with
        tool_use blocks and before those tools execute. This keeps voice
        ahead of action: the user hears the narration that describes a tool
        call before the call actually fires.
        """
        engine = self.config.get("llm_engine")
        log.info("chat start [engine=%s]: %r", engine, user_text)
        t0 = time.monotonic()
        try:
            if engine == "api":
                resp = await self._chat_api(
                    user_text, on_tool_call, on_text_chunk, wait_for_speech,
                )
            else:
                resp = await self._chat_cli(user_text, on_tool_call)
                if on_text_chunk and resp:
                    try:
                        on_text_chunk(resp)
                    except Exception:
                        log.exception("on_text_chunk raised (CLI path)")
            log.info("chat done %.2fs: %r", time.monotonic() - t0, resp)
            return resp
        except Exception:
            log.exception("chat failed after %.2fs", time.monotonic() - t0)
            raise

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
