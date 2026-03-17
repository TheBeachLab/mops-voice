"""Claude API async client with MCP integration and tool loop."""

import json
from pathlib import Path

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mops_voice.personality import (
    adjust_personality,
    get_personality,
    personality_tool_schemas,
)

MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT_TEMPLATE = """\
You are {assistant_name}, a voice assistant for digital fabrication.
You address the user as {user_name}.
Personality settings: humor={humor}%, sarcasm={sarcasm}%, honesty={honesty}%.
Adjust your tone accordingly. Keep responses concise — they'll be spoken aloud.
You control fabrication machines through MOPS tools.
When a personality setting is changed, confirm it briefly in-character.
When asked about your settings, report them.\
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


def handle_personality_tool(
    tool_name: str,
    tool_input: dict,
    config: dict,
    config_path: Path | None,
) -> str:
    """Handle a personality tool call. Returns result string for Claude."""
    if tool_name == "adjust_personality":
        result = adjust_personality(
            config, config_path, tool_input["dial"], tool_input["value"]
        )
        if isinstance(result, str):
            return result  # error message
        return json.dumps(result)
    elif tool_name == "get_personality":
        return json.dumps(get_personality(config))
    return f"Unknown personality tool: {tool_name}"


class MopsLLM:
    """Manages Claude API calls and MCP tool execution."""

    def __init__(self, config: dict, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
        self.messages: list[dict] = []
        self.max_messages = 100  # 50 pairs
        self.mcp_session: ClientSession | None = None
        self.mcp_tools: list[dict] = []
        self._mcp_cm = None
        self._mcp_session_cm = None

    @staticmethod
    def check_api_key() -> bool:
        """Check if ANTHROPIC_API_KEY is set. Call at startup."""
        import os

        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    async def connect_mcp(
        self, server_path: str, extra_args: list[str] | None = None
    ) -> bool:
        """Connect to the MOPS MCP server. Returns True on success."""
        try:
            args = [server_path] + (extra_args or [])
            server_params = StdioServerParameters(
                command="node",
                args=args,
            )
            # Keep context managers alive for session lifetime
            self._mcp_cm = stdio_client(server_params)
            read, write = await self._mcp_cm.__aenter__()
            self._mcp_session_cm = ClientSession(read, write)
            self.mcp_session = await self._mcp_session_cm.__aenter__()
            await self.mcp_session.initialize()

            # Discover tools
            tools_result = await self.mcp_session.list_tools()
            self.mcp_tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_result.tools
            ]
            return True
        except Exception:
            self.mcp_session = None
            self.mcp_tools = []
            return False

    async def disconnect_mcp(self):
        """Shut down MCP session and subprocess."""
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

    def _get_tools(self) -> list[dict]:
        """Get all tools: MCP tools + personality tools."""
        return self.mcp_tools + personality_tool_schemas()

    def _prune_messages(self):
        """Prune old messages, keeping last 20 (10 pairs)."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-20:]

    async def chat(self, user_text: str, on_tool_call=None) -> str:
        """Send user text, handle tool loop, return final text response.

        on_tool_call: optional callback(tool_name, result_summary) for verbose output.
        """
        self.messages.append({"role": "user", "content": user_text})
        self._prune_messages()

        tools = self._get_tools()

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = await self.client.messages.create(
                    model=self.config["claude_model"],
                    max_tokens=1024,
                    system=build_system_prompt(self.config),
                    messages=self.messages,
                    tools=tools if tools else None,
                )
            except Exception:
                # API error, rate limit, or network issue
                return "I'm having trouble reaching my brain, Fran. Try again."

            # Collect text and tool_use blocks
            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if not tool_uses:
                # No more tool calls — we have the final response
                final_text = " ".join(text_parts).strip()
                self.messages.append(
                    {"role": "assistant", "content": response.content}
                )
                return final_text

            # Process tool calls
            self.messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input

                # Check if it's a personality tool
                if tool_name in ("adjust_personality", "get_personality"):
                    result = handle_personality_tool(
                        tool_name, tool_input, self.config, self.config_path
                    )
                elif self.mcp_session:
                    # Call MCP tool
                    try:
                        mcp_result = await self.mcp_session.call_tool(
                            tool_name, tool_input
                        )
                        result = (
                            mcp_result.content[0].text
                            if mcp_result.content
                            else "OK"
                        )
                    except Exception as e:
                        result = f"Tool error: {e}"
                else:
                    result = "Tool unavailable — MCP server not connected."

                if on_tool_call:
                    # Summarize result for verbose output
                    summary = result[:80] + "..." if len(result) > 80 else result
                    on_tool_call(tool_name, summary)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})

        # Exceeded max iterations
        return "I got a bit carried away with the tools there, Fran. Let me try again."
