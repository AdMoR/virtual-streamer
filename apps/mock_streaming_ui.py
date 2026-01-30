"""
Mock Streaming Test Environment.

A Streamlit-based interface for testing Twitch + Video Server interactions
without real Twitch credentials. Simulates:
- User greetings (new user joins)
- User questions (!allo command)
- Video generation (/generate command)
- Feedback submission
- Video player polling

Run with:
    streamlit run apps/mock_streaming_ui.py
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Mock Streaming Test",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    .user-message {
        background-color: #1e3a5f;
        border-left: 3px solid #4299e1;
    }
    .bot-message {
        background-color: #2d3748;
        border-left: 3px solid #48bb78;
    }
    .system-message {
        background-color: #3d2d4a;
        border-left: 3px solid #9f7aea;
        font-style: italic;
    }
    .video-card {
        background-color: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .job-card {
        background-color: #0f3460;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }
    .status-pending { color: #f6e05e; }
    .status-running { color: #4299e1; }
    .status-completed { color: #48bb78; }
    .status-failed { color: #fc8181; }
    .timestamp {
        color: #718096;
        font-size: 0.75em;
    }
    .command-hint {
        background-color: #2d3748;
        padding: 8px 12px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Client
# =============================================================================

class MockStreamingClient:
    """Async HTTP client for the Virtual Streamer API."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip("/")

    async def health_check(self) -> dict:
        """Check API health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.api_url}/health")
                return {"status": "ok", "data": resp.json()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def submit_greeting(self, user_name: str, character_id: str = "jesus") -> dict:
        """Submit a greeting video generation request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/v1/jesus-agents/greeting/submit",
                json={"user_name": user_name, "character_id": character_id}
            )
            return {"status_code": resp.status_code, "data": resp.json()}

    async def submit_question(
        self, user_name: str, question: str, character_id: str = "jesus"
    ) -> dict:
        """Submit a Q&A video generation request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/v1/jesus-agents/answering/submit",
                json={
                    "user_name": user_name,
                    "question": question,
                    "character_id": character_id,
                }
            )
            return {"status_code": resp.status_code, "data": resp.json()}

    async def submit_generate(self, stream_id: str, title: str, user: str) -> dict:
        """Submit a video generation from broadcast request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/v1/video-generation/generate-from-broadcast",
                json={"stream_id": stream_id, "title": title, "user": user}
            )
            return {"status_code": resp.status_code, "data": resp.json()}

    async def get_next_video(self, stream_id: str) -> dict:
        """Get the next video to play from a stream."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.api_url}/api/v1/streams/{stream_id}/next-video"
            )
            return {"status_code": resp.status_code, "data": resp.json()}

    async def mark_played(self, entry_id: str) -> dict:
        """Mark a playlist entry as played."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/v1/playlist/{entry_id}/played"
            )
            return {"status_code": resp.status_code, "data": resp.json()}

    async def submit_feedback(self, entry_id: str, user: str, feedback: str) -> dict:
        """Submit feedback for a played video."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.api_url}/api/v1/video-generation/feedback",
                json={"entry_id": entry_id, "user": user, "feedback": feedback}
            )
            return {"status_code": resp.status_code, "data": resp.json()}

    async def get_job_status(self, job_id: str) -> dict:
        """Get the status of a video generation job."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.api_url}/api/v1/video-generation/jobs/{job_id}"
            )
            return {"status_code": resp.status_code, "data": resp.json()}

    async def list_jobs(self, limit: int = 20) -> dict:
        """List recent video generation jobs."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.api_url}/api/v1/video-generation/jobs",
                params={"limit": limit}
            )
            return {"status_code": resp.status_code, "data": resp.json()}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChatMessage:
    """Represents a chat message."""
    username: str
    message: str
    timestamp: str
    message_type: str = "user"  # user, bot, system


@dataclass
class APILog:
    """Represents an API call log entry."""
    timestamp: str
    method: str
    endpoint: str
    request_data: Optional[dict] = None
    response_data: Optional[dict] = None
    status_code: Optional[int] = None
    error: Optional[str] = None


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        # Configuration
        "api_url": "http://localhost:8000",
        "stream_id": "default",
        "character_id": "jesus",
        
        # Chat state
        "chat_messages": [],
        "current_username": "viewer123",
        
        # Video player state
        "current_video": None,
        "pending_feedback_entry": None,
        "pending_feedback_user": None,
        
        # Job tracking
        "tracked_jobs": [],
        
        # API logs
        "api_logs": [],
        
        # API health
        "api_healthy": None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_client() -> MockStreamingClient:
    """Get the API client with current configuration."""
    return MockStreamingClient(st.session_state.api_url)


def add_chat_message(username: str, message: str, msg_type: str = "user"):
    """Add a message to the chat history."""
    st.session_state.chat_messages.append(ChatMessage(
        username=username,
        message=message,
        timestamp=datetime.now().strftime("%H:%M:%S"),
        message_type=msg_type,
    ))


def add_api_log(
    method: str,
    endpoint: str,
    request_data: dict = None,
    response_data: dict = None,
    status_code: int = None,
    error: str = None,
):
    """Add an entry to the API logs."""
    st.session_state.api_logs.insert(0, APILog(
        timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        method=method,
        endpoint=endpoint,
        request_data=request_data,
        response_data=response_data,
        status_code=status_code,
        error=error,
    ))
    # Keep only last 50 logs
    st.session_state.api_logs = st.session_state.api_logs[:50]


def track_job(job_id: str, job_type: str, metadata: dict = None):
    """Add a job to the tracking list."""
    st.session_state.tracked_jobs.insert(0, {
        "job_id": job_id,
        "job_type": job_type,
        "submitted_at": datetime.now().isoformat(),
        "metadata": metadata or {},
        "last_status": "pending",
    })
    # Keep only last 20 jobs
    st.session_state.tracked_jobs = st.session_state.tracked_jobs[:20]


# =============================================================================
# Chat Message Handler
# =============================================================================

async def handle_chat_message(username: str, message: str) -> dict:
    """
    Process a chat message like the Twitch chat reader would.
    
    Commands:
    - !allo <question> - Submit a question to the answering agent
    - /generate <title> - Submit a video generation request
    - + or - - Submit feedback for the pending video
    """
    client = get_client()
    message_stripped = message.strip()
    message_lower = message_stripped.lower()
    
    try:
        # Question command: !allo <question>
        if message_lower.startswith("!allo ") or message_lower.startswith("allo "):
            prefix_len = 6 if message_lower.startswith("!allo ") else 5
            question = message_stripped[prefix_len:].strip()
            
            if not question:
                return {"type": "error", "message": "Question is empty"}
            
            add_api_log("POST", "/api/v1/jesus-agents/answering/submit", 
                       {"user_name": username, "question": question})
            
            result = await client.submit_question(
                username, question, st.session_state.character_id
            )
            
            add_api_log("POST", "/api/v1/jesus-agents/answering/submit",
                       response_data=result["data"], status_code=result["status_code"])
            
            if result["status_code"] in [200, 202]:
                job_id = result["data"].get("job_id")
                track_job(job_id, "question", {"question": question, "user": username})
                add_chat_message("Bot", f"Question received! Job ID: {job_id[:8]}...", "bot")
                return {"type": "question", "job_id": job_id, "response": result}
            else:
                add_chat_message("Bot", f"Error: {result['data']}", "bot")
                return {"type": "error", "response": result}
        
        # Generate command: /generate <title>
        elif message_lower.startswith("/generate "):
            title = message_stripped[10:].strip()
            
            if not title:
                return {"type": "error", "message": "Title is empty"}
            
            add_api_log("POST", "/api/v1/video-generation/generate-from-broadcast",
                       {"stream_id": st.session_state.stream_id, "title": title, "user": username})
            
            result = await client.submit_generate(
                st.session_state.stream_id, title, username
            )
            
            add_api_log("POST", "/api/v1/video-generation/generate-from-broadcast",
                       response_data=result["data"], status_code=result["status_code"])
            
            if result["status_code"] in [200, 202]:
                job_id = result["data"].get("job_id")
                track_job(job_id, "generate", {"title": title, "user": username})
                add_chat_message("Bot", f"Video generation started! Job ID: {job_id[:8]}...", "bot")
                return {"type": "generate", "job_id": job_id, "response": result}
            elif result["status_code"] == 429:
                add_chat_message("Bot", "Queue full! Try again later.", "bot")
                return {"type": "error", "response": result}
            elif result["status_code"] == 404:
                add_chat_message("Bot", "No active programmation found.", "bot")
                return {"type": "error", "response": result}
            else:
                add_chat_message("Bot", f"Error: {result['data']}", "bot")
                return {"type": "error", "response": result}
        
        # Feedback: + or -
        elif message_stripped in ["+", "-", "like", "dislike", "good", "bad"]:
            if st.session_state.pending_feedback_entry:
                entry_id = st.session_state.pending_feedback_entry
                feedback_user = st.session_state.pending_feedback_user or username
                
                add_api_log("POST", "/api/v1/video-generation/feedback",
                           {"entry_id": entry_id, "user": feedback_user, "feedback": message_stripped})
                
                result = await client.submit_feedback(entry_id, feedback_user, message_stripped)
                
                add_api_log("POST", "/api/v1/video-generation/feedback",
                           response_data=result["data"], status_code=result["status_code"])
                
                if result["status_code"] == 200:
                    add_chat_message("Bot", "Thanks for your feedback!", "bot")
                    st.session_state.pending_feedback_entry = None
                    st.session_state.pending_feedback_user = None
                    return {"type": "feedback", "response": result}
                else:
                    add_chat_message("Bot", f"Feedback error: {result['data']}", "bot")
                    return {"type": "error", "response": result}
            else:
                add_chat_message("Bot", "No video waiting for feedback.", "bot")
                return {"type": "info", "message": "No pending feedback"}
        
        # Regular chat message
        else:
            return {"type": "chat", "message": message_stripped}
    
    except Exception as e:
        add_api_log("ERROR", "unknown", error=str(e))
        add_chat_message("System", f"Error: {e}", "system")
        return {"type": "error", "error": str(e)}


async def handle_user_join(username: str) -> dict:
    """Handle a user joining the chat (greeting)."""
    client = get_client()
    
    try:
        add_api_log("POST", "/api/v1/jesus-agents/greeting/submit",
                   {"user_name": username, "character_id": st.session_state.character_id})
        
        result = await client.submit_greeting(username, st.session_state.character_id)
        
        add_api_log("POST", "/api/v1/jesus-agents/greeting/submit",
                   response_data=result["data"], status_code=result["status_code"])
        
        if result["status_code"] in [200, 202]:
            job_id = result["data"].get("job_id")
            track_job(job_id, "greeting", {"user": username})
            add_chat_message("System", f"🎉 {username} joined! Greeting video queued.", "system")
            return {"type": "greeting", "job_id": job_id, "response": result}
        else:
            add_chat_message("System", f"Greeting error: {result['data']}", "system")
            return {"type": "error", "response": result}
    
    except Exception as e:
        add_api_log("ERROR", "/api/v1/jesus-agents/greeting/submit", error=str(e))
        return {"type": "error", "error": str(e)}


# =============================================================================
# Video Player Functions
# =============================================================================

async def poll_next_video() -> dict:
    """Poll for the next video to play."""
    client = get_client()
    
    try:
        add_api_log("GET", f"/api/v1/streams/{st.session_state.stream_id}/next-video")
        
        result = await client.get_next_video(st.session_state.stream_id)
        
        add_api_log("GET", f"/api/v1/streams/{st.session_state.stream_id}/next-video",
                   response_data=result["data"], status_code=result["status_code"])
        
        if result["status_code"] == 200 and result["data"].get("entry_id"):
            st.session_state.current_video = result["data"]
            return {"type": "video", "data": result["data"]}
        elif result["status_code"] == 404:
            st.session_state.current_video = None
            return {"type": "empty", "message": "No videos in queue"}
        else:
            return {"type": "error", "response": result}
    
    except Exception as e:
        add_api_log("ERROR", f"/api/v1/streams/{st.session_state.stream_id}/next-video", error=str(e))
        return {"type": "error", "error": str(e)}


async def mark_video_played(entry_id: str, metadata: dict = None) -> dict:
    """Mark the current video as played and set up feedback."""
    client = get_client()
    
    try:
        add_api_log("POST", f"/api/v1/playlist/{entry_id}/played")
        
        result = await client.mark_played(entry_id)
        
        add_api_log("POST", f"/api/v1/playlist/{entry_id}/played",
                   response_data=result["data"], status_code=result["status_code"])
        
        if result["status_code"] == 200:
            # Set up pending feedback
            user = metadata.get("user") if metadata else None
            if user:
                st.session_state.pending_feedback_entry = entry_id
                st.session_state.pending_feedback_user = user
                add_chat_message("Bot", f"@{user}, your video just played! + or - ?", "bot")
            
            st.session_state.current_video = None
            return {"type": "success", "response": result}
        else:
            return {"type": "error", "response": result}
    
    except Exception as e:
        add_api_log("ERROR", f"/api/v1/playlist/{entry_id}/played", error=str(e))
        return {"type": "error", "error": str(e)}


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the configuration sidebar."""
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.markdown("---")
    
    # API Configuration
    st.sidebar.subheader("🌐 API Settings")
    
    st.session_state.api_url = st.sidebar.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="Base URL of the Virtual Streamer API",
    )
    
    # Health check button
    if st.sidebar.button("🏥 Check Health", use_container_width=True):
        with st.spinner("Checking..."):
            result = asyncio.run(get_client().health_check())
            if result["status"] == "ok":
                st.session_state.api_healthy = True
                st.sidebar.success("API is healthy!")
            else:
                st.session_state.api_healthy = False
                st.sidebar.error(f"API error: {result.get('error')}")
    
    # Show health status
    if st.session_state.api_healthy is True:
        st.sidebar.markdown("🟢 API Connected")
    elif st.session_state.api_healthy is False:
        st.sidebar.markdown("🔴 API Disconnected")
    
    st.sidebar.markdown("---")
    
    # Stream Configuration
    st.sidebar.subheader("📺 Stream Settings")
    
    st.session_state.stream_id = st.sidebar.text_input(
        "Stream ID",
        value=st.session_state.stream_id,
        help="ID of the stream to use for video operations",
    )
    
    st.session_state.character_id = st.sidebar.text_input(
        "Character ID",
        value=st.session_state.character_id,
        help="Character for greeting/answering agents",
    )
    
    st.sidebar.markdown("---")
    
    # Command Reference
    st.sidebar.subheader("📖 Commands")
    st.sidebar.markdown("""
    <div class="command-hint">
    <strong>!allo &lt;question&gt;</strong><br>
    Ask a question<br><br>
    <strong>/generate &lt;title&gt;</strong><br>
    Generate a video<br><br>
    <strong>+ / -</strong><br>
    Submit feedback
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Clear buttons
    st.sidebar.subheader("🧹 Clear Data")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    with col2:
        if st.button("Clear Logs", use_container_width=True):
            st.session_state.api_logs = []
            st.rerun()
    
    if st.sidebar.button("Clear All", use_container_width=True):
        for key in ["chat_messages", "api_logs", "tracked_jobs", "current_video", 
                    "pending_feedback_entry", "pending_feedback_user"]:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.rerun()


def render_chat_panel():
    """Render the chat simulator panel."""
    st.subheader("💬 Chat Simulator")
    
    # Chat history container
    chat_container = st.container(height=300)
    
    with chat_container:
        if not st.session_state.chat_messages:
            st.info("No messages yet. Simulate a user action below!")
        else:
            for msg in st.session_state.chat_messages:
                css_class = {
                    "user": "user-message",
                    "bot": "bot-message",
                    "system": "system-message",
                }.get(msg.message_type, "user-message")
                
                st.markdown(
                    f'<div class="chat-message {css_class}">'
                    f'<span class="timestamp">[{msg.timestamp}]</span> '
                    f'<strong>{msg.username}</strong>: {msg.message}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    
    st.markdown("---")
    
    # User actions
    st.markdown("**Simulate User Action**")
    
    # Username input
    col1, col2 = st.columns([1, 3])
    with col1:
        username = st.text_input("Username", value=st.session_state.current_username, key="chat_username")
        st.session_state.current_username = username
    
    with col2:
        message = st.text_input("Message", key="chat_message", 
                               placeholder="Type a message, !allo question, or /generate title...")
    
    # Action buttons
    col_join, col_send, col_question, col_generate = st.columns(4)
    
    with col_join:
        if st.button("🎉 User Join", use_container_width=True, help="Simulate user joining"):
            add_chat_message("System", f"→ {username} joined the chat", "system")
            with st.spinner("Submitting greeting..."):
                asyncio.run(handle_user_join(username))
            st.rerun()
    
    with col_send:
        if st.button("📤 Send", use_container_width=True, type="primary"):
            if message:
                add_chat_message(username, message, "user")
                with st.spinner("Processing..."):
                    asyncio.run(handle_chat_message(username, message))
                st.rerun()
    
    with col_question:
        if st.button("❓ Quick Question", use_container_width=True, help="Ask 'How are you?'"):
            add_chat_message(username, "!allo How are you?", "user")
            with st.spinner("Submitting question..."):
                asyncio.run(handle_chat_message(username, "!allo How are you?"))
            st.rerun()
    
    with col_generate:
        if st.button("🎬 Quick Generate", use_container_width=True, help="Generate test video"):
            add_chat_message(username, "/generate Test video from UI", "user")
            with st.spinner("Submitting generation..."):
                asyncio.run(handle_chat_message(username, "/generate Test video from UI"))
            st.rerun()
    
    # Feedback buttons (shown when pending)
    if st.session_state.pending_feedback_entry:
        st.markdown("---")
        st.markdown(f"**Pending Feedback** for `{st.session_state.pending_feedback_entry[:8]}...`")
        col_plus, col_minus = st.columns(2)
        with col_plus:
            if st.button("👍 +", use_container_width=True, key="feedback_plus"):
                user = st.session_state.pending_feedback_user or username
                add_chat_message(user, "+", "user")
                with st.spinner("Submitting feedback..."):
                    asyncio.run(handle_chat_message(user, "+"))
                st.rerun()
        with col_minus:
            if st.button("👎 -", use_container_width=True, key="feedback_minus"):
                user = st.session_state.pending_feedback_user or username
                add_chat_message(user, "-", "user")
                with st.spinner("Submitting feedback..."):
                    asyncio.run(handle_chat_message(user, "-"))
                st.rerun()


def render_video_panel():
    """Render the video player mock panel."""
    st.subheader("📺 Video Player Mock")
    
    # Current video display
    video = st.session_state.current_video
    
    if video:
        entry_id = video.get("entry_id", "N/A")
        video_url = video.get("video_url", "N/A")
        metadata = video.get("metadata", {})
        
        st.markdown(f"""
        <div class="video-card">
            <h4>🎬 Now Playing</h4>
            <p><strong>Entry ID:</strong> <code>{entry_id}</code></p>
            <p><strong>User:</strong> {metadata.get('user', 'N/A')}</p>
            <p><strong>Title:</strong> {metadata.get('title', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Video URL (truncated)
        if video_url and video_url != "N/A":
            with st.expander("Video URL"):
                st.code(video_url, language=None)
        
        # Metadata
        if metadata:
            with st.expander("Full Metadata"):
                st.json(metadata)
        
        # Mark as played button
        if st.button("✅ Mark as Played", use_container_width=True, type="primary"):
            with st.spinner("Marking as played..."):
                asyncio.run(mark_video_played(entry_id, metadata))
            st.rerun()
    else:
        st.info("No video currently loaded. Click 'Poll Next' to fetch one.")
    
    st.markdown("---")
    
    # Poll button
    if st.button("🔄 Poll Next Video", use_container_width=True):
        with st.spinner("Polling..."):
            result = asyncio.run(poll_next_video())
            if result["type"] == "empty":
                st.warning("No videos in queue")
            elif result["type"] == "error":
                st.error(f"Error: {result.get('error') or result.get('response')}")
        st.rerun()


def render_job_tracker():
    """Render the job tracker panel."""
    st.subheader("📋 Job Tracker")
    
    # Refresh button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_jobs"):
            with st.spinner("Refreshing..."):
                client = get_client()
                for job in st.session_state.tracked_jobs:
                    try:
                        result = asyncio.run(client.get_job_status(job["job_id"]))
                        if result["status_code"] == 200:
                            job["last_status"] = result["data"].get("status", "unknown")
                            job["result"] = result["data"].get("result")
                    except Exception:
                        pass
            st.rerun()
    
    # Job list
    if not st.session_state.tracked_jobs:
        st.info("No jobs tracked yet. Submit a greeting, question, or generate request.")
    else:
        for job in st.session_state.tracked_jobs:
            status = job.get("last_status", "unknown")
            status_class = f"status-{status}"
            job_type = job.get("job_type", "unknown")
            job_id = job.get("job_id", "N/A")
            
            type_emoji = {"greeting": "👋", "question": "❓", "generate": "🎬"}.get(job_type, "📦")
            
            with st.expander(f"{type_emoji} {job_type.upper()} - {job_id[:8]}... ({status})"):
                st.markdown(f"""
                <p><strong>Status:</strong> <span class="{status_class}">{status.upper()}</span></p>
                <p><strong>Submitted:</strong> {job.get('submitted_at', 'N/A')}</p>
                """, unsafe_allow_html=True)
                
                if job.get("metadata"):
                    st.json(job["metadata"])
                
                if job.get("result"):
                    st.markdown("**Result:**")
                    st.json(job["result"])


def render_api_logs():
    """Render the API logs panel."""
    st.subheader("📜 API Logs")
    
    if not st.session_state.api_logs:
        st.info("No API calls yet.")
    else:
        for log in st.session_state.api_logs[:20]:
            status_color = "🟢" if log.status_code and log.status_code < 400 else "🔴" if log.status_code else "⚪"
            
            with st.expander(f"{status_color} [{log.timestamp}] {log.method} {log.endpoint}"):
                if log.request_data:
                    st.markdown("**Request:**")
                    st.json(log.request_data)
                
                if log.response_data:
                    st.markdown("**Response:**")
                    st.json(log.response_data)
                
                if log.error:
                    st.error(f"Error: {log.error}")
                
                if log.status_code:
                    st.caption(f"Status Code: {log.status_code}")


def render_status_bar():
    """Render the status bar at the top."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Chat Messages", len(st.session_state.chat_messages))
    
    with col2:
        st.metric("Tracked Jobs", len(st.session_state.tracked_jobs))
    
    with col3:
        st.metric("API Calls", len(st.session_state.api_logs))
    
    with col4:
        pending = st.session_state.pending_feedback_entry
        st.metric("Pending Feedback", "Yes" if pending else "No")
    
    with col5:
        video = st.session_state.current_video
        st.metric("Video Loaded", "Yes" if video else "No")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    st.title("📺 Mock Streaming Test Environment")
    st.markdown("Test Twitch + Video Server interactions without real credentials.")
    
    # Render sidebar
    render_sidebar()
    
    # Render status bar
    render_status_bar()
    
    st.markdown("---")
    
    # Main content layout
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        render_chat_panel()
    
    with col_right:
        render_video_panel()
    
    st.markdown("---")
    
    # Bottom panels in tabs
    tab_jobs, tab_logs = st.tabs(["📋 Job Tracker", "📜 API Logs"])
    
    with tab_jobs:
        render_job_tracker()
    
    with tab_logs:
        render_api_logs()


if __name__ == "__main__":
    main()
