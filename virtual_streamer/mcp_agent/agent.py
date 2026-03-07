"""
MCP-based agentic loop for Twitch stream hosting.

Connects to the MCP server via stdio subprocess, uses an OpenAI-compatible
client, and runs a tool-call loop every LOOP_INTERVAL_SECONDS.
"""

import asyncio
import json
import logging
import os
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import AsyncOpenAI

from virtual_streamer.mcp_agent.config import AgentConfig
from virtual_streamer.mcp_agent.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP tool definition to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def _format_context(chat_messages: list[dict], queue_status: dict) -> str:
    """Build the context user message injected each iteration."""
    pending = queue_status.get("pending_count", "?")
    played = queue_status.get("played_count", "?")
    is_replaying = queue_status.get("is_replaying", False)
    error = queue_status.get("error")

    if error:
        queue_line = f"Queue: error - {error}"
    else:
        replay = " (replay mode)" if is_replaying else ""
        queue_line = f"Queue: {pending} pending, {played} played{replay}"

    lines = ["[STREAM STATUS]", queue_line, ""]

    if chat_messages:
        mentions = [m for m in chat_messages if m.get("is_mention")]
        lines.append(f"[RECENT CHAT - last {len(chat_messages)} messages]")
        for msg in chat_messages:
            ts = msg.get("timestamp", "")
            user = msg.get("username", "?")
            text = msg.get("message", "")
            mention = " [MENTION]" if msg.get("is_mention") else ""
            lines.append(f"{ts} <{user}>{mention}: {text}")
        lines.append("")
        lines.append(f"New mentions: {len(mentions)}")
    else:
        lines.append("[RECENT CHAT]")
        lines.append("No recent messages.")

    return "\n".join(lines)


class MCPAgentLoop:
    """Agentic loop that drives the Virtual Streamer via MCP tools."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self._conversation_history: list[dict] = []

    async def run(self) -> None:
        """Start the MCP subprocess and run the agentic loop forever."""
        cmd = self.config.mcp_server_command
        server_params = StdioServerParameters(
            command=cmd[0],
            args=cmd[1:],
            env={**os.environ, **self.config.to_mcp_env()},
        )

        logger.info(f"Starting MCP server subprocess: {' '.join(cmd)}")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("MCP session initialized")

                tools_result = await session.list_tools()
                openai_tools = [_mcp_tool_to_openai(t) for t in tools_result.tools]
                logger.info(
                    f"Discovered {len(openai_tools)} MCP tools: "
                    + ", ".join(t["function"]["name"] for t in openai_tools)
                )

                await self._loop(session, openai_tools)

    async def _loop(self, session: ClientSession, tools: list[dict]) -> None:
        """Main loop: run one iteration, sleep, repeat."""
        while True:
            try:
                await self._run_iteration(session, tools)
            except Exception as e:
                logger.error(f"Iteration error: {e}", exc_info=True)
            await asyncio.sleep(self.config.loop_interval)

    async def _run_iteration(
        self, session: ClientSession, tools: list[dict]
    ) -> None:
        """Run one agentic iteration."""
        # Fetch context via MCP tools
        chat_messages: list[dict] = []
        queue_status: dict = {}

        try:
            result = await session.call_tool("get_chat_messages", {"limit": 20})
            chat_messages = _parse_tool_result(result)
            if not isinstance(chat_messages, list):
                chat_messages = []
        except Exception as e:
            logger.warning(f"get_chat_messages failed: {e}")

        try:
            result = await session.call_tool("get_queue_status", {})
            queue_status = _parse_tool_result(result)
            if not isinstance(queue_status, dict):
                queue_status = {}
        except Exception as e:
            logger.warning(f"get_queue_status failed: {e}")

        # Skip iteration if nothing to do
        pending = queue_status.get("pending_count")
        has_mentions = any(m.get("is_mention") for m in chat_messages)
        queue_low = pending is not None and pending < 3

        if not has_mentions and not queue_low:
            logger.debug("No mentions and queue not low — skipping iteration")
            return

        context_msg = _format_context(chat_messages, queue_status)
        logger.debug(f"Context:\n{context_msg}")

        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + self._conversation_history
            + [{"role": "user", "content": context_msg}]
        )

        # Tool-call loop
        while True:
            response = await self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for tc in choice.message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info(f"Calling tool: {tool_name}({tool_args})")
                    try:
                        tool_result = await session.call_tool(tool_name, tool_args)
                        result_content = json.dumps(
                            _parse_tool_result(tool_result), ensure_ascii=False
                        )
                    except Exception as e:
                        logger.error(f"Tool {tool_name} failed: {e}")
                        result_content = json.dumps({"error": str(e)})

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_content,
                        }
                    )
            else:
                # Final response
                final_text = choice.message.content or ""
                if final_text:
                    logger.info(f"Agent response: {final_text}")
                messages.append({"role": "assistant", "content": final_text})
                break

        # Update conversation history (drop system message and context injection)
        new_history = messages[1:]  # strip system prompt
        # Drop the leading context user message we just added
        if new_history and new_history[0].get("content") == context_msg:
            new_history = new_history[1:]

        self._conversation_history = new_history[-(self.config.max_history):]


def _parse_tool_result(result: Any) -> Any:
    """Extract a plain Python object from an MCP CallToolResult.

    FastMCP serialises list[T] returns as one TextContent *per element*.
    A dict/scalar return is a single TextContent containing the JSON repr.
    """
    if not hasattr(result, "content"):
        return result

    parts = result.content
    if not parts:
        return result

    # Single content item → try JSON, fall back to raw text
    if len(parts) == 1:
        item = parts[0]
        if hasattr(item, "text"):
            try:
                return json.loads(item.text)
            except (json.JSONDecodeError, TypeError):
                return item.text
        return item

    # Multiple content items → each item is one element of the original list
    collected = []
    for item in parts:
        if hasattr(item, "text"):
            try:
                collected.append(json.loads(item.text))
            except (json.JSONDecodeError, TypeError):
                collected.append(item.text)
        else:
            collected.append(item)
    return collected
