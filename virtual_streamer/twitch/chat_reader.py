"""
DEPRECATED: This file is deprecated and will be removed in a future version.

The functionality has been migrated to:
- virtual_streamer/streaming/twitch/chat_reader.py

The new version:
- Uses HTTP API instead of RabbitMQ
- Configurable stream_id and story_template_id
- Better error handling and reconnection logic

To use the new Twitch reader:
    python -m virtual_streamer.streaming.twitch.chat_reader
"""
import warnings
warnings.warn(
    "virtual_streamer.twitch.chat_reader is deprecated. "
    "Use virtual_streamer.streaming.twitch.chat_reader instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import os
import re
import time
import asyncio
import threading
import websockets
import requests
import logging
from datetime import datetime, timedelta
from virtual_streamer.utils.utils import add_to_queue, ChatQuestion
from virtual_streamer.workflows.prompts import VERY_SARCASTIC_STANDUP_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TwitchChatReader")


def append_question(user, message):
    """
    Add a question from a user to the message queue.

    Args:
        user: The username of the person asking the question
        message: The content of their message/question
    """
    queue_name = "chat_log"
    # API : name: str,  question: str, routing_queue: str = "video_response_queue", prompt: str = None
    msg = ChatQuestion(
        name=user,
        question=message,
        routing_queue="obs",
        prompt=VERY_SARCASTIC_STANDUP_PROMPT,
        next_queue=None,
    )
    add_to_queue(queue_name, msg.serialize())
    logger.info(f"Added question from {user}: {message}")


class TwitchClient:
    """
    Client for connecting to Twitch chat and handling messages.

    This class handles authentication with Twitch API, connecting to the
    chat websocket, and processing incoming messages with automatic token refresh.
    """

    def __init__(self, client_id, client_secret, refresh_token):
        """
        Initialize the Twitch client.

        Args:
            client_id: Twitch API client ID
            client_secret: Twitch API client secret
            refresh_token: Twitch refresh token for authentication
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expiry = time.time()
        self.base_url = "https://api.twitch.tv/helix"
        self.oauth_url = "https://id.twitch.tv/oauth2"
        self.chat_url = "wss://irc-ws.chat.twitch.tv:443"

        # Token refresh lock to prevent multiple simultaneous refreshes
        self.token_lock = threading.Lock()
        # Store refresh tokens to a file
        self.token_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "refresh_token.json"
        )

        # Start token refresh thread
        self.token_refresh_thread = threading.Thread(
            target=self._token_refresh_monitor, daemon=True
        )
        self.token_refresh_thread.start()

        print("Lock : ", self.token_lock)

        # Ensure we have a valid token to start with
        self.refresh_access_token()

        print("1st refresh ok")

    def _token_refresh_monitor(self):
        """
        Background thread that monitors token expiration and refreshes when needed.
        Runs continuously to ensure token is always valid.
        """
        while True:
            try:
                # Check if token needs refresh (if it expires in less than 10 minutes)
                current_time = time.time()
                if self.token_expiry - current_time < 0:  # 10 minutes in seconds
                    self.refresh_access_token()
                    logger.info("Token refreshed by background monitor")
                    self.token_expiry = current_time + 10 * 60

                # Sleep for 5 minutes before checking again
                time.sleep(300)
            except Exception as e:
                logger.error(f"Error in token refresh monitor: {e}")
                time.sleep(60)  # Sleep for a minute if there's an error

    def refresh_access_token(self):
        """
        Refresh the Twitch access token using the refresh token.
        Updates the access_token and token_expiry properties.
        """
        try:
            with self.token_lock:
                url = f"{self.oauth_url}/token"
                params = {
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
                response = requests.post(url, params=params)
                response.raise_for_status()  # Raise exception for HTTP errors

                response_data = response.json()
                self.access_token = response_data["access_token"]
                self.refresh_token = response_data["refresh_token"]
                self.token_expiry = time.time() + response_data["expires_in"]

                # Save the new refresh token to file
                self._save_refresh_token()

                logger.info(
                    f"Access token refreshed, expires in {response_data['expires_in']} seconds"
                )
                return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh access token: {e}")
            return False
        except KeyError as e:
            logger.error(f"Missing key in token response: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error refreshing token: {e}")
            return False

    def _save_refresh_token(self):
        """Save the current refresh token to a file for persistence."""
        try:
            with open(self.token_file, "w") as f:
                json.dump(
                    {
                        "refresh_token": self.refresh_token,
                        "updated_at": datetime.now().isoformat(),
                    },
                    f,
                )
            logger.debug("Refresh token saved to file")
        except Exception as e:
            logger.error(f"Failed to save refresh token: {e}")

    def _load_refresh_token(self):
        """Load the refresh token from file if available."""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, "r") as f:
                    data = json.load(f)
                    self.refresh_token = data["refresh_token"]
                    logger.info(
                        f"Loaded refresh token from file (updated: {data['updated_at']})"
                    )
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to load refresh token: {e}")
            return False

    def ensure_token_valid(self):
        """
        Check if the current token is valid and refresh if needed.
        Thread-safe method that can be called before any API operation.
        """
        with self.token_lock:
            current_time = time.time()

            # If token is missing or expired, refresh it
            if not self.access_token or current_time >= self.token_expiry:
                return self.refresh_access_token()
            return True

    def get_headers(self):
        """Get the authorization headers for API requests."""
        self.ensure_token_valid()
        return {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
        }

    async def connect_to_chat(self, channel_name):
        """
        Connect to Twitch chat via WebSocket.

        Args:
            channel_name: The Twitch channel to connect to
        """
        # Ensure we have a valid token
        if not self.ensure_token_valid():
            logger.error("Failed to obtain valid access token")
            await asyncio.sleep(30)  # Wait before retrying
            return

        try:
            async with websockets.connect(self.chat_url) as websocket:
                # Send authentication messages
                await websocket.send(f"PASS oauth:{self.access_token}")
                await websocket.send(f"NICK allojesuschrist")
                await websocket.send(f"JOIN #{channel_name}")

                # Request capabilities
                await websocket.send(
                    "CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership"
                )

                logger.info(f"Connected to #{channel_name} chat")

                # Set up ping timer
                last_ping = time.time()
                ping_interval = 250  # seconds

                while True:
                    try:
                        # Check if we need to send a ping
                        if time.time() - last_ping > ping_interval:
                            await websocket.send("PING :tmi.twitch.tv")
                            last_ping = time.time()
                            logger.debug("Sent PING to Twitch")

                        # Set a timeout for receiving messages
                        response = await asyncio.wait_for(websocket.recv(), timeout=30)

                        # Process the message
                        self.handle_message(websocket, response, channel_name)

                    except asyncio.TimeoutError:
                        # No message received within timeout, continue the loop
                        continue
                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"Connection closed: {e}")
                        # Reconnect will happen automatically on next iteration
                        break
                    except Exception as e:
                        logger.error(f"Error in chat connection: {e}")
                        # Wait a bit before reconnecting
                        break

        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 400:  # Bad Request, likely due to invalid token
                logger.warning(
                    f"Invalid status code {e.status_code}, refreshing token..."
                )
                self.refresh_access_token()
            else:
                logger.error(f"WebSocket connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in connect_to_chat: {e}")

    def get_stream_chat(self, channel_name):
        """
        Start reading the chat for a given channel with automatic reconnection.

        Args:
            channel_name: The Twitch channel to connect to
        """

        async def read_chat():
            backoff_time = 1  # Start with 1 second backoff
            max_backoff = 300  # Maximum backoff of 5 minutes

            while True:
                try:
                    logger.info(
                        f"Connecting to Twitch chat for channel: {channel_name}"
                    )
                    await self.connect_to_chat(channel_name)

                    # Reset backoff time on successful connection
                    backoff_time = 1
                except Exception as e:
                    logger.error(f"Error in chat connection loop: {e}")

                # Implement exponential backoff for reconnection attempts
                logger.info(f"Reconnecting in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff)

        # Create and run the event loop
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

    def handle_message(self, websocket, message, channel_name):
        """
        Process a message from Twitch chat.

        Args:
            websocket: The active websocket connection
            message: The raw message from Twitch
            channel_name: The channel the message is from
        """
        logger.debug(f"Received message: {message}")

        # Handle different message types
        if message.startswith("PING"):
            pong_response = message.replace("PING", "PONG")
            asyncio.create_task(self.send_pong(pong_response, websocket))
        elif "PRIVMSG" in message:
            self.handle_privmsg(websocket, message, channel_name)
        elif "JOIN" in message:
            self.handle_join(message)
        elif "PART" in message:
            self.handle_part(message)
        elif "NOTICE" in message:
            self.handle_notice(message)
            # Check for authentication failure
            if "Login authentication failed" in message:
                logger.warning("Authentication failed, refreshing token...")
                self.refresh_access_token()
        else:
            logger.debug(f"Unhandled message: {message}")

    async def send_pong(self, pong_response, websocket):
        """
        Send a PONG response to Twitch's PING.

        Args:
            pong_response: The PONG message to send
            websocket: The active websocket connection
        """
        await websocket.send(pong_response)
        logger.debug(f"Sent PONG: {pong_response}")

    def handle_notice(self, msg):
        """
        Handle NOTICE messages from Twitch.

        Args:
            msg: The notice message
        """
        logger.info(f"Received notice: {msg}")

    def handle_privmsg(self, websocket, message, channel):
        """
        Handle a private message (chat message) from Twitch.

        Args:
            websocket: The active websocket connection
            message: The raw message from Twitch
            channel: The channel the message is from
        """
        try:
            # Parse the message
            parts = message.split(":", 2)
            if len(parts) > 2:
                user_info, chat_message = parts[1], parts[2]
                username = user_info.split("!")[0]
            else:
                logger.warning("Failed to parse message: " + message)
                return

            logger.info(f"[{channel}] {username}: {chat_message}")

            # Process commands
            if chat_message.lstrip(" ").lower().startswith(
                "!allo"
            ) or chat_message.lstrip(" ").lower().startswith("allo"):
                question_text = (
                    chat_message[5:].strip()
                    if chat_message.lstrip(" ").lower().startswith("!allo")
                    else chat_message[4:].strip()
                )
                append_question(username, question_text)
                response = f"Merci {username}, ta question est en cours de traitement."
            elif chat_message.lower().startswith(
                "jesus"
            ) or chat_message.lower().startswith("jésus"):
                response = f"{username}, si tu souhaites poser une question à Jésus. Utilises !allo avant ta question. Ex: !allo Salut, la forme ?."
            else:
                response = ""

            # Send response if needed
            if response:
                asyncio.create_task(self.send_message(websocket, channel, response))

        except Exception as e:
            logger.error(f"Error handling chat message: {e}")

    async def send_message(self, websocket, channel_name, msg):
        """
        Send a message to the Twitch chat.

        Args:
            websocket: The active websocket connection
            channel_name: The channel to send the message to
            msg: The message content to send
        """
        if not msg:
            return

        try:
            await websocket.send(f"PRIVMSG #{channel_name} :{msg}")
            logger.info(f"Sent message to #{channel_name}: {msg}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def handle_join(self, message):
        """
        Handle JOIN messages (when someone joins the channel).

        Args:
            message: The raw JOIN message
        """
        try:
            parts = message.split("!")
            if len(parts) > 1:
                username = parts[0][1:]
                logger.debug(f"{username} joined the chat")
        except Exception as e:
            logger.error(f"Error handling join message: {e}")

    def handle_part(self, message):
        """
        Handle PART messages (when someone leaves the channel).

        Args:
            message: The raw PART message
        """
        try:
            parts = message.split("!")
            if len(parts) > 1:
                username = parts[0][1:]
                logger.debug(f"{username} left the chat")
        except Exception as e:
            logger.error(f"Error handling part message: {e}")


# Main execution
if __name__ == "__main__":
    # Load credentials from file or environment variables
    try:
        creds_file = "/home/amor/Documents/code_dw/virtual-streamer/creds.json"
        if os.path.exists(creds_file):
            with open(creds_file) as f:
                creds = json.load(f)
                logger.info("Loaded credentials from creds.json")
        else:
            creds = os.environ
            logger.info("Using credentials from environment variables")
    except Exception as e:
        logger.error(f"Error loading credentials: {e}")
        creds = os.environ
        logger.info("Falling back to environment variables for credentials")

    # Initialize the Twitch client
    try:
        client_id = creds.get("client_id")
        client_secret = creds.get("client_secret")
        refresh_token = creds.get("refresh_token")
        print("Oui")

        if not all([client_id, client_secret, refresh_token]):
            logger.error(
                "Missing required credentials. Ensure client_id, client_secret, and refresh_token are provided."
            )
            exit(1)

        twitch_client = TwitchClient(client_id, client_secret, refresh_token)
        logger.info("Twitch client initialized")

        # Start reading chat
        channel_name = creds.get("channel_name", "allojesuschrist")
        logger.info(f"Starting chat reader for channel: {channel_name}")
        twitch_client.get_stream_chat(channel_name)
    except KeyboardInterrupt:
        logger.info("Chat reader stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error starting chat reader: {e}")
