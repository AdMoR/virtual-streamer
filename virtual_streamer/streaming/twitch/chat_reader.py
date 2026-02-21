"""
Twitch Chat Reader - Refactored for HTTP API

This module reads Twitch chat and submits questions to the Virtual Streamer API
for video generation. It replaces the old RabbitMQ-based approach with direct
HTTP API calls.

Environment variables:
- TWITCH_CLIENT_ID: Twitch API client ID
- TWITCH_CLIENT_SECRET: Twitch API client secret
- TWITCH_REFRESH_TOKEN: Twitch refresh token for authentication
- TWITCH_CHANNEL: Twitch channel to connect to
- API_URL: Virtual Streamer API URL (default: http://localhost:8000)
- STREAM_ID: Stream ID to add videos to (default: default)
- STORY_TEMPLATE_ID: Story template to use for generation (default: None)
"""

import json
import os
import time
import asyncio
import threading
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import websockets
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TwitchChatReader")

# Configuration from environment
# Default API_URL uses Docker Compose service name for inter-container communication
API_URL = os.environ.get("API_URL", "http://virtual_streamer_api:8000")
STREAM_ID = os.environ.get("STREAM_ID", "default")
STORY_TEMPLATE_ID = os.environ.get("STORY_TEMPLATE_ID")

# Feedback configuration
FEEDBACK_TIMEOUT = 120  # seconds to wait for user feedback


@dataclass
class PendingFeedback:
    """Tracks a pending feedback request for a user."""
    entry_id: str
    job_id: str
    expires_at: float


async def submit_question_to_api(user: str, message: str):
    """
    Submit a question to the Virtual Streamer API for video generation.
    
    Args:
        user: The username of the person asking the question
        message: The content of their message/question
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Submit video generation request
            response = await client.post(
                f"{API_URL}/api/v1/video-generation",
                json={
                    "title": f"Question from {user}: {message}",
                    "stream_id": STREAM_ID,
                    "story_template_id": STORY_TEMPLATE_ID,
                    "metadata": {
                        "source": "twitch",
                        "user": user,
                        "question": message,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                }
            )
            
            if response.status_code == 200 or response.status_code == 202:
                data = response.json()
                logger.info(f"Submitted question from {user}, job_id: {data.get('job_id')}")
                return data
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
    except httpx.TimeoutException:
        logger.error("API request timed out")
        return None
    except httpx.RequestError as e:
        logger.error(f"API request failed: {e}")
        return None


async def submit_generate_to_api(user: str, title: str) -> dict:
    """
    Submit a video generation request using the active broadcast's story template.
    
    This calls the generate-from-broadcast endpoint which:
    1. Gets the active programmation for STREAM_ID
    2. Uses that programmation's story_template_id
    3. Enforces queue limits (max 5 pending jobs)
    
    Args:
        user: The username of the person requesting generation
        title: The video title/topic from the user's message
        
    Returns:
        dict with job_id and status on success, or error info on failure
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_URL}/api/v1/video-generation/generate-from-broadcast",
                json={
                    "stream_id": STREAM_ID,
                    "title": title,
                    "user": user,
                }
            )
            
            data = response.json()
            
            if response.status_code == 200 or response.status_code == 202:
                logger.info(
                    f"Submitted generate request from {user}, "
                    f"job_id: {data.get('job_id')}, "
                    f"template: {data.get('story_template_id')}"
                )
                return {"success": True, **data}
            elif response.status_code == 429:
                logger.warning(f"Queue full for generate request from {user}: {data.get('detail')}")
                return {"success": False, "error": "queue_full", "detail": data.get("detail")}
            elif response.status_code == 404:
                logger.warning(f"No active programmation for generate request: {data.get('detail')}")
                return {"success": False, "error": "no_programmation", "detail": data.get("detail")}
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"success": False, "error": "api_error", "detail": data.get("detail", response.text)}
                
    except httpx.TimeoutException:
        logger.error("API request timed out")
        return {"success": False, "error": "timeout", "detail": "Request timed out"}
    except httpx.RequestError as e:
        logger.error(f"API request failed: {e}")
        return {"success": False, "error": "request_error", "detail": str(e)}


async def submit_feedback_to_api(entry_id: str, user: str, feedback: str) -> bool:
    """Submit raw feedback to API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{API_URL}/api/v1/video-generation/feedback",
                json={"entry_id": entry_id, "user": user, "feedback": feedback}
            )
            return response.status_code == 200
    except Exception as e:
        logger.error(f"Feedback submit error: {e}")
        return False


class TwitchClient:
    """
    Client for connecting to Twitch chat and handling messages.
    
    This class handles authentication with Twitch API, connecting to the
    chat websocket, and processing incoming messages with automatic token refresh.
    """

    def __init__(
        self, 
        client_id: str, 
        client_secret: str, 
        refresh_token: str,
        channel_name: str = None,
        bot_username: str = None
    ):
        """
        Initialize the Twitch client.
        
        Args:
            client_id: Twitch API client ID
            client_secret: Twitch API client secret
            refresh_token: Twitch refresh token for authentication
            channel_name: Twitch channel to connect to
            bot_username: Username to use for the bot
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.channel_name = channel_name or os.environ.get("TWITCH_CHANNEL", "")
        self.bot_username = bot_username or os.environ.get("TWITCH_BOT_USERNAME", "virtualstreamerbot")
        
        self.access_token = None
        self.token_expiry = time.time()
        self.base_url = "https://api.twitch.tv/helix"
        self.oauth_url = "https://id.twitch.tv/oauth2"
        self.chat_url = "wss://irc-ws.chat.twitch.tv:443"

        # Token refresh lock
        self.token_lock = threading.Lock()
        self.token_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "refresh_token.json"
        )

        # Event callbacks
        self._on_user_join_callback: Optional[Callable[[str], Any]] = None
        self._on_new_message_callback: Optional[Callable[[str, str], Any]] = None

        # Active websocket reference (set during connect_to_chat)
        self._active_websocket = None

        # Feedback monitoring state
        self._pending_feedback: Dict[str, PendingFeedback] = {}
        self._feedback_asked_users: set = set()  # Users already asked (ask at most once)
        self._last_played_check: datetime = datetime.utcnow()

        # Start token refresh thread
        self.token_refresh_thread = threading.Thread(
            target=self._token_refresh_monitor, 
            daemon=True
        )
        self.token_refresh_thread.start()

        # Initial token refresh
        self.refresh_access_token()
        logger.info("Twitch client initialized")

    def _token_refresh_monitor(self):
        """Background thread that monitors token expiration."""
        while True:
            try:
                current_time = time.time()
                if self.token_expiry - current_time < 600:  # 10 minutes
                    self.refresh_access_token()
                    logger.info("Token refreshed by background monitor")
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in token refresh monitor: {e}")
                time.sleep(60)

    def refresh_access_token(self) -> bool:
        """Refresh the Twitch access token."""
        try:
            with self.token_lock:
                response = httpx.post(
                    f"{self.oauth_url}/token",
                    params={
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                
                data = response.json()
                self.access_token = data["access_token"]
                self.refresh_token = data["refresh_token"]
                self.token_expiry = time.time() + data["expires_in"]
                
                self._save_refresh_token()
                logger.info(f"Access token refreshed, expires in {data['expires_in']} seconds")
                return True
                
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            return False

    def _save_refresh_token(self):
        """Save the current refresh token to file and .env.local."""
        try:
            with open(self.token_file, "w") as f:
                json.dump({
                    "refresh_token": self.refresh_token,
                    "updated_at": datetime.now().isoformat(),
                }, f)
        except Exception as e:
            logger.error(f"Failed to save refresh token to file: {e}")

        # Also update .env.local so credentials stay in sync across restarts
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            for name in (".env.local", ".env"):
                env_path = os.path.join(project_root, name)
                if os.path.exists(env_path):
                    with open(env_path, "r") as f:
                        lines = f.read().splitlines()
                    updated = False
                    for i, line in enumerate(lines):
                        if line.strip().startswith("refresh_token"):
                            lines[i] = f'refresh_token="{self.refresh_token}"'
                            updated = True
                            break
                    if not updated:
                        lines.append(f'refresh_token="{self.refresh_token}"')
                    with open(env_path, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    logger.info(f"Updated refresh token in {name}")
                    break
        except Exception as e:
            logger.error(f"Failed to update .env.local with refresh token: {e}")

    def ensure_token_valid(self) -> bool:
        """Check if the current token is valid and refresh if needed."""
        with self.token_lock:
            if not self.access_token or time.time() >= self.token_expiry:
                return self.refresh_access_token()
            return True

    async def _monitor_played_videos(self, websocket):
        """
        Background task: polls for played videos and prompts users for feedback.
        Runs alongside chat reading, shares self._pending_feedback dict.
        """
        while True:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"{API_URL}/api/v1/streams/{STREAM_ID}/played-since",
                        params={"since": self._last_played_check.isoformat()}
                    )
                    
                    if response.status_code == 200:
                        entries = response.json()
                        self._last_played_check = datetime.utcnow()
                        
                        for entry in entries:
                            metadata = entry.get("metadata") or {}
                            user = metadata.get("user")
                            
                            # Skip if no user, already pending, or already asked once
                            if not user or user in self._pending_feedback or user in self._feedback_asked_users:
                                continue
                            
                            self._pending_feedback[user] = PendingFeedback(
                                entry_id=entry["entry_id"],
                                job_id=metadata.get("job_id", ""),
                                expires_at=time.time() + FEEDBACK_TIMEOUT,
                            )
                            self._feedback_asked_users.add(user)
                            
                            msg = f"{user}, ta video vient de passer! + ou - ?"
                            await websocket.send(f"PRIVMSG #{self.channel_name} :{msg}")
                            logger.info(f"Feedback requested from {user}")
                            
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            await asyncio.sleep(10)  # Poll every 10 seconds

    def set_on_user_join_callback(self, callback: Callable[[str], Any]) -> None:
        """
        Register a callback for user join events.
        
        Args:
            callback: Function to call with username when a user joins
        """
        self._on_user_join_callback = callback

    async def on_user_join(self, username: str) -> None:
        """
        Called when a new user joins the channel.
        
        Args:
            username: The username of the user who joined
        """
        # TODO: Call remote API to handle user join event
        if self._on_user_join_callback:
            result = self._on_user_join_callback(username)
            if asyncio.iscoroutine(result):
                await result

    async def handle_join(self, message: str) -> None:
        """
        Handle a user joining the channel.
        
        Args:
            message: Raw JOIN message from Twitch IRC
        """
        try:
            # JOIN format: :username!username@username.tmi.twitch.tv JOIN #channel
            if "!" in message:
                username = message.split("!")[0].lstrip(":")
                # Exclude bot's own join
                if username.lower() != self.bot_username.lower():
                    logger.info(f"User {username} joined #{self.channel_name}")
                    await self.on_user_join(username)
        except Exception as e:
            logger.error(f"Error handling JOIN message: {e}")

    def set_on_new_message_callback(self, callback: Callable[[str, str], Any]) -> None:
        """
        Register a callback for new chat messages.
        
        Args:
            callback: Function to call with (username, message) for each chat message
        """
        self._on_new_message_callback = callback

    async def on_new_message(self, username: str, message: str) -> None:
        """
        Called when a new chat message is received.
        
        Args:
            username: The username of the message sender
            message: The content of the message
        """
        if self._on_new_message_callback:
            result = self._on_new_message_callback(username, message)
            if asyncio.iscoroutine(result):
                await result

    async def send_chat_message(self, message: str) -> None:
        """
        Send a message to the connected Twitch channel.

        Can be called externally while the WebSocket is active (e.g., from MCP tools).

        Args:
            message: The message to send to the chat

        Raises:
            RuntimeError: If not connected to Twitch chat
        """
        if self._active_websocket is None:
            raise RuntimeError("Not connected to Twitch chat")
        await self._active_websocket.send(
            f"PRIVMSG #{self.channel_name} :{message}"
        )

    @property
    def is_connected(self) -> bool:
        """Whether the client has an active WebSocket connection."""
        return self._active_websocket is not None

    async def connect_to_chat(self):
        """Connect to Twitch chat via WebSocket."""
        if not self.ensure_token_valid():
            logger.error("Failed to obtain valid access token")
            await asyncio.sleep(30)
            return

        try:
            async with websockets.connect(self.chat_url) as websocket:
                self._active_websocket = websocket

                # Authenticate
                await websocket.send(f"PASS oauth:{self.access_token}")
                await websocket.send(f"NICK {self.bot_username}")
                await websocket.send(f"JOIN #{self.channel_name}")
                await websocket.send(
                    "CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership"
                )

                logger.info(f"Connected to #{self.channel_name} chat")

                # Start played video monitor as background task
                monitor_task = asyncio.create_task(self._monitor_played_videos(websocket))

                last_ping = time.time()
                ping_interval = 250

                try:
                    while True:
                        try:
                            if time.time() - last_ping > ping_interval:
                                await websocket.send("PING :tmi.twitch.tv")
                                last_ping = time.time()

                            response = await asyncio.wait_for(
                                websocket.recv(), 
                                timeout=30
                            )
                            await self.handle_message(websocket, response)

                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed as e:
                            logger.warning(f"Connection closed: {e}")
                            break
                        except Exception as e:
                            logger.error(f"Error in chat connection: {e}")
                            break
                finally:
                    self._active_websocket = None
                    monitor_task.cancel()

        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 400:
                logger.warning("Invalid status code, refreshing token...")
                self.refresh_access_token()
            else:
                logger.error(f"WebSocket connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in connect_to_chat: {e}")

    async def handle_message(self, websocket, message: str):
        """Process a message from Twitch chat."""
        logger.debug(f"Received message: {message}")

        if message.startswith("PING"):
            pong_response = message.replace("PING", "PONG")
            await websocket.send(pong_response)
        elif "JOIN" in message and "PRIVMSG" not in message:
            await self.handle_join(message)
        elif "PRIVMSG" in message:
            await self.handle_privmsg(websocket, message)
        elif "NOTICE" in message:
            if "Login authentication failed" in message:
                logger.warning("Authentication failed, refreshing token...")
                self.refresh_access_token()

    async def handle_privmsg(self, websocket, message: str):
        """Handle a chat message."""
        try:
            parts = message.split(":", 2)
            if len(parts) > 2:
                user_info, chat_message = parts[1], parts[2]
                username = user_info.split("!")[0]
            else:
                logger.warning(f"Failed to parse message: {message}")
                return

            logger.info(f"[{self.channel_name}] {username}: {chat_message}")

            # Notify callback of new message
            await self.on_new_message(username, chat_message)

            # Parse command prefix early so !generate can take precedence over feedback
            chat_lower = chat_message.lstrip(" ").lower()

            # Check for pending feedback, but let !generate take precedence
            if username in self._pending_feedback:
                pending = self._pending_feedback[username]

                if chat_lower.startswith("!generate "):
                    # !generate takes precedence: clear feedback and fall through to command handling
                    del self._pending_feedback[username]
                elif time.time() < pending.expires_at:
                    # User has pending feedback - capture their message as raw feedback
                    success = await submit_feedback_to_api(
                        pending.entry_id,
                        username,
                        chat_message  # Raw message
                    )
                    del self._pending_feedback[username]

                    response = "Merci pour ton retour!" if success else "Erreur, désolé!"
                    await websocket.send(f"PRIVMSG #{self.channel_name} :{response}")
                    return  # Don't process as command
                else:
                    # Expired, clean up
                    del self._pending_feedback[username]

            # Check for command prefix
            if chat_lower.startswith("!allo") or chat_lower.startswith("allo"):
                # Extract the question
                if chat_lower.startswith("!allo"):
                    question_text = chat_message[5:].strip()
                else:
                    question_text = chat_message[4:].strip()
                
                if question_text:
                    # Submit to API
                    await submit_question_to_api(username, question_text)
                    
                    # Send acknowledgment
                    response = f"Merci {username}, ta question est en cours de traitement."
                    await websocket.send(
                        f"PRIVMSG #{self.channel_name} :{response}"
                    )

            elif chat_lower.startswith("!generate "):
                # Extract the title from the message
                title = chat_message[10:].strip()  # len("!generate ") = 10
                
                if title:
                    # Submit to API using active broadcast's story template
                    result = await submit_generate_to_api(username, title)
                    
                    if result.get("success"):
                        response = (
                            f"Merci {username}! Vidéo en cours de génération "
                            f"(template: {result.get('story_template_id')})"
                        )
                    elif result.get("error") == "queue_full":
                        response = (
                            f"Désolé {username}, la file d'attente est pleine. "
                            "Réessaie dans quelques minutes!"
                        )
                    elif result.get("error") == "no_programmation":
                        response = (
                            f"Désolé {username}, aucune programmation active en ce moment."
                        )
                    else:
                        response = f"Désolé {username}, une erreur s'est produite."
                    
                    await websocket.send(
                        f"PRIVMSG #{self.channel_name} :{response}"
                    )
                else:
                    # No title provided
                    response = (
                        f"{username}, utilise !generate <titre> pour générer une vidéo. "
                        "Ex: !generate Fred se lance dans la conquête spatiale"
                    )
                    await websocket.send(
                        f"PRIVMSG #{self.channel_name} :{response}"
                    )

        except Exception as e:
            logger.error(f"Error handling chat message: {e}")

    def run(self):
        """Start the chat reader with automatic reconnection."""
        async def read_chat():
            backoff_time = 1
            max_backoff = 300

            while True:
                try:
                    logger.info(f"Connecting to Twitch chat for channel: {self.channel_name}")
                    await self.connect_to_chat()
                    backoff_time = 1
                except Exception as e:
                    logger.error(f"Error in chat connection loop: {e}")

                logger.info(f"Reconnecting in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(read_chat())
        except KeyboardInterrupt:
            logger.info("Chat reader stopped by user")
        except Exception as e:
            logger.critical(f"Fatal error in chat reader: {e}")
        finally:
            loop.close()


def main():
    """Main entry point."""
    # Load credentials from environment
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
    refresh_token = os.environ.get("TWITCH_REFRESH_TOKEN")
    channel_name = os.environ.get("TWITCH_CHANNEL")

    if not all([client_id, client_secret, refresh_token, channel_name]):
        logger.error(
            "Missing required environment variables. "
            "Ensure TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, "
            "TWITCH_REFRESH_TOKEN, and TWITCH_CHANNEL are set."
        )
        return

    logger.info(f"Starting Twitch chat reader for channel: {channel_name}")
    logger.info(f"API URL: {API_URL}")
    logger.info(f"Stream ID: {STREAM_ID}")

    client = TwitchClient(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
        channel_name=channel_name,
    )
    client.run()


if __name__ == "__main__":
    main()
