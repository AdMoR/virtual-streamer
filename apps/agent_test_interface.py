"""
Virtual Streamer Agent Test Interface.

A Streamlit-based interface for testing the Virtual Streamer Agent
with mock tools and configurable context.

Run with:
    streamlit run apps/agent_test_interface.py
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from virtual_streamer.agents.virtual_streamer_agent.schema import WorkloadStatus
from virtual_streamer.agents.virtual_streamer_agent.test_runner import (
    VirtualStreamerTestRunner,
    TestRunnerConfig,
)
from virtual_streamer.agents.virtual_streamer_agent.context.mock_providers import (
    MockChatConfig,
    MockChatMessage,
    MockQueueConfig,
    MockWorkloadConfig,
)
from virtual_streamer.agents.virtual_streamer_agent.tools.mock import (
    MockToolFactoryConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Virtual Streamer Agent Test",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #2d3748;
        border-left: 3px solid #4299e1;
    }
    .agent-message {
        background-color: #1a365d;
        border-left: 3px solid #48bb78;
    }
    .mention-badge {
        background-color: #ed8936;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    .tool-call {
        background-color: #2d3748;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-family: monospace;
    }
    .timestamp {
        color: #718096;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "runner" not in st.session_state:
        st.session_state.runner = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = []
    
    if "agent_responses" not in st.session_state:
        st.session_state.agent_responses = []
    
    if "iteration_count" not in st.session_state:
        st.session_state.iteration_count = 0
    
    # Default context configuration
    if "queue_pending" not in st.session_state:
        st.session_state.queue_pending = 5
    
    if "queue_played" not in st.session_state:
        st.session_state.queue_played = 10
    
    if "workload" not in st.session_state:
        st.session_state.workload = "low"
    
    if "active_jobs" not in st.session_state:
        st.session_state.active_jobs = 0
    
    if "is_replaying" not in st.session_state:
        st.session_state.is_replaying = False
    
    if "time_offset" not in st.session_state:
        st.session_state.time_offset = 0.0
    
    if "next_videos" not in st.session_state:
        st.session_state.next_videos = "Fred se lance dans l'IA\nPourquoi les chats retombent sur leurs pattes"


def get_or_create_runner() -> VirtualStreamerTestRunner:
    """Get or create the test runner."""
    if st.session_state.runner is None:
        config = TestRunnerConfig(
            queue_config=MockQueueConfig(
                pending_count=st.session_state.queue_pending,
                played_count=st.session_state.queue_played,
                is_replaying=st.session_state.is_replaying,
                active_jobs=st.session_state.active_jobs,
                next_videos=st.session_state.next_videos.split("\n") if st.session_state.next_videos else [],
            ),
            workload_config=MockWorkloadConfig(
                workload=WorkloadStatus(st.session_state.workload),
                active_jobs=st.session_state.active_jobs,
                queue_pending=st.session_state.queue_pending,
            ),
            chat_config=MockChatConfig(messages=[]),
        )
        st.session_state.runner = VirtualStreamerTestRunner(config)
        # Run setup in async context
        asyncio.run(st.session_state.runner.setup())
    
    return st.session_state.runner


def update_runner_context():
    """Update runner context from session state."""
    runner = get_or_create_runner()
    
    runner.set_queue_pending(st.session_state.queue_pending)
    runner.set_queue_played(st.session_state.queue_played)
    runner.set_workload(WorkloadStatus(st.session_state.workload))
    runner.set_active_jobs(st.session_state.active_jobs)
    runner.set_replay_mode(st.session_state.is_replaying)
    runner.set_chat_time_offset(st.session_state.time_offset)
    
    videos = st.session_state.next_videos.split("\n") if st.session_state.next_videos else []
    runner.set_next_videos([v.strip() for v in videos if v.strip()])


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with context configuration."""
    st.sidebar.title("🎛️ Context Configuration")
    
    st.sidebar.markdown("---")
    
    # Queue Configuration
    st.sidebar.subheader("📼 Queue Status")
    
    st.session_state.queue_pending = st.sidebar.number_input(
        "Pending Videos",
        min_value=0,
        max_value=100,
        value=st.session_state.queue_pending,
        help="Number of fresh videos waiting to be played",
    )
    
    st.session_state.queue_played = st.sidebar.number_input(
        "Played Videos",
        min_value=0,
        max_value=1000,
        value=st.session_state.queue_played,
        help="Number of videos available for replay",
    )
    
    st.session_state.is_replaying = st.sidebar.checkbox(
        "Replay Mode",
        value=st.session_state.is_replaying,
        help="Whether the stream is currently playing replays",
    )
    
    st.session_state.next_videos = st.sidebar.text_area(
        "Next Videos (one per line)",
        value=st.session_state.next_videos,
        height=100,
        help="Titles of upcoming videos",
    )
    
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.subheader("⚙️ System Status")
    
    workload_options = ["low", "normal", "high", "critical", "unknown"]
    st.session_state.workload = st.sidebar.selectbox(
        "Workload Level",
        options=workload_options,
        index=workload_options.index(st.session_state.workload),
        help="Current system workload level",
    )
    
    st.session_state.active_jobs = st.sidebar.number_input(
        "Active Jobs",
        min_value=0,
        max_value=20,
        value=st.session_state.active_jobs,
        help="Number of video generation jobs running",
    )
    
    st.sidebar.markdown("---")
    
    # Time Configuration
    st.sidebar.subheader("⏰ Time Settings")
    
    st.session_state.time_offset = st.sidebar.slider(
        "Chat Time Offset (minutes)",
        min_value=0.0,
        max_value=60.0,
        value=st.session_state.time_offset,
        step=1.0,
        help="Simulate old conversation (0 = messages are current)",
    )
    
    st.sidebar.markdown("---")
    
    # Apply Context Button
    if st.sidebar.button("🔄 Apply Context", use_container_width=True):
        update_runner_context()
        st.sidebar.success("Context updated!")
    
    st.sidebar.markdown("---")
    
    # Preset Scenarios
    st.sidebar.subheader("📋 Quick Scenarios")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Empty Queue", use_container_width=True):
            st.session_state.queue_pending = 0
            st.session_state.is_replaying = True
            st.session_state.workload = "low"
            update_runner_context()
            st.rerun()
    
    with col2:
        if st.button("Busy System", use_container_width=True):
            st.session_state.workload = "critical"
            st.session_state.active_jobs = 5
            update_runner_context()
            st.rerun()
    
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        if st.button("Stale Chat", use_container_width=True):
            st.session_state.time_offset = 10.0
            update_runner_context()
            st.rerun()
    
    with col4:
        if st.button("Reset All", use_container_width=True):
            st.session_state.queue_pending = 5
            st.session_state.queue_played = 10
            st.session_state.workload = "low"
            st.session_state.active_jobs = 0
            st.session_state.is_replaying = False
            st.session_state.time_offset = 0.0
            st.session_state.chat_history = []
            st.session_state.tool_calls = []
            st.session_state.agent_responses = []
            st.session_state.iteration_count = 0
            if st.session_state.runner:
                st.session_state.runner.clear_history()
                st.session_state.runner.clear_chat_history()
            st.rerun()


def render_chat_panel():
    """Render the main chat panel."""
    st.subheader("💬 Chat Simulation")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        runner = get_or_create_runner()
        all_messages = runner.get_chat_messages()
        
        if not all_messages and not st.session_state.agent_responses:
            st.info("No messages yet. Add a chat message below to start testing.")
        else:
            # Display user messages and agent responses interleaved
            for msg in all_messages:
                mention_badge = ' <span class="mention-badge">MENTION</span>' if msg.is_mention else ''
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<span class="timestamp">{msg.timestamp}</span> '
                    f'<strong>@{msg.username}</strong>{mention_badge}: {msg.message}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            
            # Display agent responses
            for response in st.session_state.agent_responses:
                if response.get("response"):
                    st.markdown(
                        f'<div class="chat-message agent-message">'
                        f'<span class="timestamp">{response.get("timestamp", "")}</span> '
                        f'<strong>🤖 Agent</strong>: {response["response"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    
    st.markdown("---")
    
    # New message input
    st.markdown("**Add Chat Message**")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        username = st.text_input("Username", value="viewer123", key="new_username")
    
    with col2:
        message = st.text_input("Message", key="new_message", placeholder="Type a message...")
    
    col3, col4, col5 = st.columns([1, 1, 2])
    
    with col3:
        is_mention = st.checkbox("Is Mention", key="is_mention")
    
    with col4:
        if st.button("➕ Add Message", use_container_width=True):
            if message:
                runner = get_or_create_runner()
                runner.add_chat_message(username, message, is_mention)
                st.session_state.chat_history.append({
                    "username": username,
                    "message": message,
                    "is_mention": is_mention,
                    "timestamp": datetime.now().isoformat(),
                })
                st.rerun()
    
    with col5:
        if st.button("🚀 Run Agent Iteration", type="primary", use_container_width=True):
            run_agent_iteration()


def run_agent_iteration():
    """Run an agent iteration and display results."""
    runner = get_or_create_runner()
    update_runner_context()
    
    with st.spinner("Running agent iteration..."):
        try:
            result = asyncio.run(runner.run_iteration())
            
            st.session_state.iteration_count += 1
            
            # Store the result
            result["timestamp"] = datetime.now().isoformat()
            result["iteration"] = st.session_state.iteration_count
            
            if result.get("response"):
                st.session_state.agent_responses.append(result)
            
            # Store tool calls
            for tc in result.get("tool_calls", []):
                st.session_state.tool_calls.append(tc)
            
            st.success(f"Iteration {st.session_state.iteration_count} completed!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error running iteration: {e}")
            logger.exception("Error running agent iteration")


def render_logs_panel():
    """Render the logs and inspection panel."""
    st.subheader("📊 Inspection Panel")
    
    tab1, tab2, tab3 = st.tabs(["Tool Calls", "Last Prompt", "Raw Events"])
    
    runner = get_or_create_runner()
    
    with tab1:
        st.markdown("**Recent Tool Calls**")
        
        tool_calls = runner.get_tool_calls()
        
        if not tool_calls:
            st.info("No tool calls yet.")
        else:
            for tc in reversed(tool_calls[-10:]):
                with st.expander(f"🔧 {tc.tool_name} - {tc.timestamp}", expanded=False):
                    st.markdown("**Arguments:**")
                    st.json(tc.arguments)
                    st.markdown("**Result:**")
                    st.json(tc.result)
        
        # Show sent messages
        st.markdown("---")
        st.markdown("**Messages Sent by Agent**")
        
        sent_messages = runner.get_sent_messages()
        
        if not sent_messages:
            st.info("Agent hasn't sent any messages yet.")
        else:
            for msg in reversed(sent_messages[-5:]):
                st.markdown(
                    f'<div class="tool-call">'
                    f'<span class="timestamp">{msg.get("timestamp", "")}</span><br>'
                    f'{msg.get("message", "")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    
    with tab2:
        st.markdown("**Last Prompt Sent to Agent**")
        
        last_prompt = runner.get_last_prompt()
        
        if last_prompt:
            st.code(last_prompt, language=None)
        else:
            st.info("No prompt sent yet.")
        
        st.markdown("---")
        st.markdown("**Last Context State**")
        
        last_context = runner.get_last_context()
        
        if last_context:
            # Format context for display
            context_display = {}
            for key, value in last_context.items():
                if hasattr(value, 'model_dump'):
                    context_display[key] = value.model_dump()
                elif isinstance(value, list):
                    context_display[key] = [
                        v.model_dump() if hasattr(v, 'model_dump') else v
                        for v in value
                    ]
                else:
                    context_display[key] = value
            
            st.json(context_display)
        else:
            st.info("No context available yet.")
    
    with tab3:
        st.markdown("**Recent Agent Events**")
        
        events = runner.get_recent_events(20)
        
        if not events:
            st.info("No events yet.")
        else:
            for event in reversed(events):
                with st.expander(f"{event.event_type} - {event.timestamp}", expanded=False):
                    if event.content:
                        st.markdown("**Content:**")
                        st.text(event.content)


def render_status_bar():
    """Render the status bar at the top."""
    runner = get_or_create_runner()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Iterations", st.session_state.iteration_count)
    
    with col2:
        st.metric("Tool Calls", len(runner.get_tool_calls()))
    
    with col3:
        st.metric("Chat Messages", len(runner.get_chat_messages()))
    
    with col4:
        st.metric("Queue Pending", st.session_state.queue_pending)
    
    with col5:
        workload_emoji = {
            "low": "🟢",
            "normal": "🟡",
            "high": "🟠",
            "critical": "🔴",
            "unknown": "⚪",
        }
        st.metric("Workload", f"{workload_emoji.get(st.session_state.workload, '')} {st.session_state.workload.upper()}")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    st.title("🤖 Virtual Streamer Agent Test Interface")
    st.markdown("Test the Virtual Streamer Agent with mock tools and configurable context.")
    
    # Render sidebar
    render_sidebar()
    
    # Render status bar
    render_status_bar()
    
    st.markdown("---")
    
    # Main content in two columns
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        render_chat_panel()
    
    with col_right:
        render_logs_panel()


if __name__ == "__main__":
    main()
