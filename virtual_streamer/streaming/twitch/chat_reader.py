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
from datetime import datetime
from typing import Optional

import websockets
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TwitchChatReader")

# Configuration from environment
API_URL = os.environ.get("API_URL", "http://localhost:8000")
STREAM_ID = os.environ.get("STREAM_ID", "default")
STORY_TEMPLATE_ID = os.environ.get("STORY_TEMPLATE_ID")


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
        """Save the current refresh token to a file."""
        try:
            with open(self.token_file, "w") as f:
                json.dump({
                    "refresh_token": self.refresh_token,
                    "updated_at": datetime.now().isoformat(),
                }, f)
        except Exception as e:
            logger.error(f"Failed to save refresh token: {e}")

    def ensure_token_valid(self) -> bool:
        """Check if the current token is valid and refresh if needed."""
        with self.token_lock:
            if not self.access_token or time.time() >= self.token_expiry:
                return self.refresh_access_token()
            return True

    async def connect_to_chat(self):
        """Connect to Twitch chat via WebSocket."""
        if not self.ensure_token_valid():
            logger.error("Failed to obtain valid access token")
            await asyncio.sleep(30)
            return

        try:
            async with websockets.connect(self.chat_url) as websocket:
                # Authenticate
                await websocket.send(f"PASS oauth:{self.access_token}")
                await websocket.send(f"NICK {self.bot_username}")
                await websocket.send(f"JOIN #{self.channel_name}")
                await websocket.send(
                    "CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership"
                )

                logger.info(f"Connected to #{self.channel_name} chat")

                last_ping = time.time()
                ping_interval = 250

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

            # Check for command prefix
            chat_lower = chat_message.lstrip(" ").lower()
            
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
