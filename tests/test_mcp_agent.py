"""
Tests for virtual_streamer.mcp_agent.

Unit tests:
  - Helper functions: _parse_tool_result, _format_context, _mcp_tool_to_openai
  - MCPAgentLoop._run_iteration logic (mocked session + mocked LLM client)

Integration tests (marked with @pytest.mark.integration):
  - Launch the mock_server subprocess via stdio_client
  - Verify tool discovery and tool responses
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from virtual_streamer.mcp_agent.agent import (
    MCPAgentLoop,
    _format_context,
    _mcp_tool_to_openai,
    _parse_tool_result,
)
from virtual_streamer.mcp_agent.config import AgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_result(data) -> MagicMock:
    """Build a mock MCP CallToolResult containing JSON-encoded data."""
    result = MagicMock()
    item = MagicMock()
    item.text = json.dumps(data)
    result.content = [item]
    return result


def make_openai_response(
    finish_reason: str = "stop",
    content: str | None = None,
    tool_calls: list | None = None,
) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    response.choices = [choice]
    return response


def make_tool_call(call_id: str, name: str, arguments: dict) -> MagicMock:
    """Build a mock OpenAI tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _default_config(**overrides) -> AgentConfig:
    """Create an AgentConfig with sensible test defaults."""
    defaults = dict(
        llm_base_url="http://localhost:9999/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        loop_interval=0.0,
        max_history=10,
    )
    defaults.update(overrides)
    # Build via direct attribute injection to bypass env reads
    cfg = AgentConfig.__new__(AgentConfig)
    for k, v in defaults.items():
        object.__setattr__(cfg, k, v)
    # Set remaining fields with defaults
    for field_name, default in [
        ("api_url", "http://localhost:8000"),
        ("stream_id", "default"),
        ("programmation_id", None),
        ("twitch_client_id", None),
        ("twitch_client_secret", None),
        ("twitch_refresh_token", None),
        ("twitch_channel", None),
        ("twitch_bot_username", "testbot"),
    ]:
        if field_name not in defaults:
            object.__setattr__(cfg, field_name, default)
    return cfg


def _make_agent(config: AgentConfig | None = None) -> MCPAgentLoop:
    cfg = config or _default_config()
    agent = MCPAgentLoop.__new__(MCPAgentLoop)
    agent.config = cfg
    agent.client = MagicMock()
    agent._conversation_history = []
    return agent


# ---------------------------------------------------------------------------
# Unit: _parse_tool_result
# ---------------------------------------------------------------------------


class TestParseToolResult:
    def test_json_content(self):
        data = {"pending_count": 3, "played_count": 10}
        result = make_tool_result(data)
        assert _parse_tool_result(result) == data

    def test_list_content(self):
        data = [{"username": "viewer1", "message": "hello"}]
        result = make_tool_result(data)
        assert _parse_tool_result(result) == data

    def test_plain_text_content(self):
        result = MagicMock()
        item = MagicMock()
        item.text = "not json at all"
        result.content = [item]
        assert _parse_tool_result(result) == "not json at all"

    def test_raw_result_no_content_attr(self):
        raw = {"some": "dict"}
        assert _parse_tool_result(raw) == raw

    def test_empty_content_list(self):
        result = MagicMock()
        result.content = []
        assert _parse_tool_result(result) == result


# ---------------------------------------------------------------------------
# Unit: _format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_no_messages_queue_ok(self):
        out = _format_context([], {"pending_count": 5, "played_count": 10})
        assert "[STREAM STATUS]" in out
        assert "5 pending" in out
        assert "No recent messages" in out

    def test_with_mentions(self):
        msgs = [
            {"username": "u1", "message": "hi", "timestamp": "T1", "is_mention": False},
            {"username": "u2", "message": "@bot salut", "timestamp": "T2", "is_mention": True},
        ]
        out = _format_context(msgs, {"pending_count": 2, "played_count": 0})
        assert "[MENTION]" in out
        assert "New mentions: 1" in out
        assert "<u2>" in out

    def test_queue_error(self):
        out = _format_context([], {"error": "No active programmation"})
        assert "error" in out.lower()
        assert "No active programmation" in out

    def test_replay_mode(self):
        out = _format_context([], {"pending_count": 0, "played_count": 20, "is_replaying": True})
        assert "replay mode" in out

    def test_message_count_in_header(self):
        msgs = [
            {"username": f"u{i}", "message": "x", "timestamp": "T", "is_mention": False}
            for i in range(5)
        ]
        out = _format_context(msgs, {"pending_count": 5})
        assert "last 5 messages" in out


# ---------------------------------------------------------------------------
# Unit: _mcp_tool_to_openai
# ---------------------------------------------------------------------------


class TestMcpToolToOpenai:
    def test_basic_conversion(self):
        tool = SimpleNamespace(
            name="send_twitch_message",
            description="Send a message",
            inputSchema={"type": "object", "properties": {"message": {"type": "string"}}},
        )
        result = _mcp_tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "send_twitch_message"
        assert result["function"]["description"] == "Send a message"
        assert "message" in result["function"]["parameters"]["properties"]

    def test_empty_description_falls_back_to_empty_string(self):
        tool = SimpleNamespace(name="health_check", description=None, inputSchema={})
        result = _mcp_tool_to_openai(tool)
        assert result["function"]["description"] == ""


# ---------------------------------------------------------------------------
# Unit: MCPAgentLoop._run_iteration
# ---------------------------------------------------------------------------


MOCK_CHAT_WITH_MENTION = [
    {"username": "viewer1", "message": "hello", "timestamp": "T1", "is_mention": False},
    {
        "username": "viewer2",
        "message": "@bot aide-moi",
        "timestamp": "T2",
        "is_mention": True,
    },
]
MOCK_QUEUE_OK = {"pending_count": 5, "played_count": 10, "is_replaying": False}
MOCK_QUEUE_LOW = {"pending_count": 1, "played_count": 10, "is_replaying": False}


def _make_session(chat=None, queue=None, tool_result_factory=None) -> MagicMock:
    """Build a mock ClientSession with configurable call_tool responses."""
    chat = chat if chat is not None else []
    queue = queue if queue is not None else MOCK_QUEUE_OK

    async def call_tool(tool_name, args=None):
        if tool_name == "get_chat_messages":
            return make_tool_result(chat)
        if tool_name == "get_queue_status":
            return make_tool_result(queue)
        if tool_result_factory:
            return tool_result_factory(tool_name, args)
        return make_tool_result({"success": True})

    session = MagicMock()
    session.call_tool = call_tool
    return session


class TestRunIteration:
    @pytest.mark.asyncio
    async def test_skip_when_no_mentions_and_queue_ok(self):
        """No LLM call when chat has no mentions and pending >= 3."""
        agent = _make_agent()
        mock_create = AsyncMock()
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        session = _make_session(chat=[], queue=MOCK_QUEUE_OK)
        await agent._run_iteration(session, [])

        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_when_mention_present(self):
        """LLM is called when a chat message mentions the bot."""
        agent = _make_agent()
        mock_create = AsyncMock(
            return_value=make_openai_response(finish_reason="stop", content="Bonjour!")
        )
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        session = _make_session(chat=MOCK_CHAT_WITH_MENTION, queue=MOCK_QUEUE_OK)
        await agent._run_iteration(session, [])

        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_runs_when_queue_low(self):
        """LLM is called when pending_count < 3, even with no mentions."""
        agent = _make_agent()
        mock_create = AsyncMock(
            return_value=make_openai_response(finish_reason="stop", content="Je vais créer une vidéo!")
        )
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        session = _make_session(chat=[], queue=MOCK_QUEUE_LOW)
        await agent._run_iteration(session, [])

        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_call_loop(self):
        """Agent calls a tool when LLM requests it, then finishes on second call."""
        agent = _make_agent()

        tc = make_tool_call("call_1", "send_twitch_message", {"message": "Salut!"})
        first_response = make_openai_response(finish_reason="tool_calls", tool_calls=[tc])
        second_response = make_openai_response(finish_reason="stop", content="Done.")

        mock_create = AsyncMock(side_effect=[first_response, second_response])
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        called_tools: list[tuple] = []

        async def call_tool(tool_name, args=None):
            if tool_name == "get_chat_messages":
                return make_tool_result(MOCK_CHAT_WITH_MENTION)
            if tool_name == "get_queue_status":
                return make_tool_result(MOCK_QUEUE_OK)
            called_tools.append((tool_name, args))
            return make_tool_result({"success": True, "message": args.get("message", "")})

        session = MagicMock()
        session.call_tool = call_tool

        await agent._run_iteration(session, [])

        assert mock_create.call_count == 2
        assert ("send_twitch_message", {"message": "Salut!"}) in called_tools

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_response(self):
        """All tool calls in a single response are executed before the next LLM call."""
        agent = _make_agent()

        tc1 = make_tool_call("c1", "send_twitch_message", {"message": "Msg 1"})
        tc2 = make_tool_call("c2", "send_twitch_message", {"message": "Msg 2"})
        first_response = make_openai_response(finish_reason="tool_calls", tool_calls=[tc1, tc2])
        second_response = make_openai_response(finish_reason="stop", content="Done.")

        mock_create = AsyncMock(side_effect=[first_response, second_response])
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        called_tools: list[str] = []

        async def call_tool(tool_name, args=None):
            if tool_name == "get_chat_messages":
                return make_tool_result(MOCK_CHAT_WITH_MENTION)
            if tool_name == "get_queue_status":
                return make_tool_result(MOCK_QUEUE_OK)
            called_tools.append(tool_name)
            return make_tool_result({"success": True})

        session = MagicMock()
        session.call_tool = call_tool

        await agent._run_iteration(session, [])

        assert called_tools.count("send_twitch_message") == 2
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_history_updated_after_iteration(self):
        """Conversation history grows after a completed iteration."""
        agent = _make_agent()
        mock_create = AsyncMock(
            return_value=make_openai_response(finish_reason="stop", content="Réponse.")
        )
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        session = _make_session(chat=MOCK_CHAT_WITH_MENTION, queue=MOCK_QUEUE_OK)
        assert agent._conversation_history == []
        await agent._run_iteration(session, [])
        # History should contain at least the final assistant message
        assert any(m["role"] == "assistant" for m in agent._conversation_history)

    @pytest.mark.asyncio
    async def test_history_trimmed_to_max_history(self):
        """History is capped at max_history entries."""
        config = _default_config(max_history=4)
        agent = _make_agent(config)

        # Each iteration adds 1 assistant message (stop, no tool calls)
        mock_create = AsyncMock(
            return_value=make_openai_response(finish_reason="stop", content="ok")
        )
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        session = _make_session(chat=MOCK_CHAT_WITH_MENTION, queue=MOCK_QUEUE_OK)
        for _ in range(10):
            await agent._run_iteration(session, [])

        assert len(agent._conversation_history) <= 4

    @pytest.mark.asyncio
    async def test_tool_failure_is_caught(self):
        """A tool call failure is captured and returned as an error dict; loop continues."""
        agent = _make_agent()

        tc = make_tool_call("c1", "create_video", {"title": "Test"})
        first_response = make_openai_response(finish_reason="tool_calls", tool_calls=[tc])
        second_response = make_openai_response(finish_reason="stop", content="Désolé.")

        mock_create = AsyncMock(side_effect=[first_response, second_response])
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        async def call_tool(tool_name, args=None):
            if tool_name == "get_chat_messages":
                return make_tool_result(MOCK_CHAT_WITH_MENTION)
            if tool_name == "get_queue_status":
                return make_tool_result(MOCK_QUEUE_OK)
            raise RuntimeError("API unavailable")

        session = MagicMock()
        session.call_tool = call_tool

        # Should not raise
        await agent._run_iteration(session, [])

        assert mock_create.call_count == 2
        # The tool result message should contain an error
        calls = mock_create.call_args_list
        second_call_messages = calls[1][1].get("messages") or calls[1][0][0]
        tool_msg = next(
            (m for m in second_call_messages if m.get("role") == "tool"), None
        )
        assert tool_msg is not None
        assert "error" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_context_message_injected_into_llm_call(self):
        """The user context message is present in the messages sent to the LLM."""
        agent = _make_agent()
        mock_create = AsyncMock(
            return_value=make_openai_response(finish_reason="stop", content="ok")
        )
        agent.client.chat = MagicMock()
        agent.client.chat.completions = MagicMock()
        agent.client.chat.completions.create = mock_create

        session = _make_session(chat=MOCK_CHAT_WITH_MENTION, queue=MOCK_QUEUE_OK)
        await agent._run_iteration(session, [])

        call_kwargs = mock_create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert "[STREAM STATUS]" in user_msgs[-1]["content"]


# ---------------------------------------------------------------------------
# Integration: mock_server subprocess via stdio_client
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMockServerIntegration:
    """
    Launch the mock_server as a subprocess and talk to it via stdio_client.

    These tests require no external services — only the local Python interpreter.
    Skip with: pytest -m "not integration"
    """

    @pytest.mark.asyncio
    async def test_tools_discovered(self):
        """Mock server exposes the expected tool names."""
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "virtual_streamer.mcp_agent.mock_server"],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                result = await asyncio.wait_for(session.list_tools(), timeout=10)
                tool_names = {t.name for t in result.tools}

        expected = {
            "get_chat_messages",
            "send_twitch_message",
            "get_queue_status",
            "create_video",
            "create_video_from_broadcast",
            "get_system_status",
            "health_check",
            "fetch_news",
        }
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"

    @pytest.mark.asyncio
    async def test_get_chat_messages_returns_default(self):
        """get_chat_messages returns the default messages including one mention."""
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "virtual_streamer.mcp_agent.mock_server"],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                raw = await asyncio.wait_for(
                    session.call_tool("get_chat_messages", {"limit": 50}), timeout=10
                )

        messages = _parse_tool_result(raw)
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert any(m.get("is_mention") for m in messages)

    @pytest.mark.asyncio
    async def test_get_queue_status_returns_default(self):
        """get_queue_status returns the default queue with 2 pending."""
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "virtual_streamer.mcp_agent.mock_server"],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                raw = await asyncio.wait_for(
                    session.call_tool("get_queue_status", {}), timeout=10
                )

        status = _parse_tool_result(raw)
        assert isinstance(status, dict)
        assert status.get("pending_count") == 2

    @pytest.mark.asyncio
    async def test_send_twitch_message_returns_success(self):
        """send_twitch_message returns success=True."""
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "virtual_streamer.mcp_agent.mock_server"],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                raw = await asyncio.wait_for(
                    session.call_tool(
                        "send_twitch_message", {"message": "Bonjour les amis!"}
                    ),
                    timeout=10,
                )

        result = _parse_tool_result(raw)
        assert result.get("success") is True
        assert result.get("message") == "Bonjour les amis!"

    @pytest.mark.asyncio
    async def test_custom_queue_via_env(self):
        """MOCK_QUEUE_JSON env var overrides the default queue state."""
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        custom_queue = {"pending_count": 0, "played_count": 99, "is_replaying": True}
        env = {**os.environ, "MOCK_QUEUE_JSON": json.dumps(custom_queue)}

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "virtual_streamer.mcp_agent.mock_server"],
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                raw = await asyncio.wait_for(
                    session.call_tool("get_queue_status", {}), timeout=10
                )

        status = _parse_tool_result(raw)
        assert status.get("pending_count") == 0
        assert status.get("is_replaying") is True

    @pytest.mark.asyncio
    async def test_run_iteration_with_mock_server(self):
        """Full iteration: agent connects to mock server, LLM is mocked, no errors raised."""
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "virtual_streamer.mcp_agent.mock_server"],
            env=os.environ.copy(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)
                tools_result = await asyncio.wait_for(session.list_tools(), timeout=10)

        openai_tools = [_mcp_tool_to_openai(t) for t in tools_result.tools]

        # Re-open connection for the iteration (context managers can't be reused)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10)

                agent = _make_agent()
                # LLM: requests send_twitch_message then stops
                tc = make_tool_call(
                    "c1", "send_twitch_message", {"message": "Salut depuis l'intégration!"}
                )
                first_response = make_openai_response(
                    finish_reason="tool_calls", tool_calls=[tc]
                )
                second_response = make_openai_response(
                    finish_reason="stop", content="Message envoyé."
                )
                agent.client.chat = MagicMock()
                agent.client.chat.completions = MagicMock()
                agent.client.chat.completions.create = AsyncMock(
                    side_effect=[first_response, second_response]
                )

                await asyncio.wait_for(
                    agent._run_iteration(session, openai_tools), timeout=15
                )

        assert agent.client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line(
        "markers", "integration: tests that launch subprocesses (slower)"
    )
