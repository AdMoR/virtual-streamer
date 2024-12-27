import json
import os
from virtual_streamer.utils.utils import add_to_queue, ChatQuestion
from prompts import VERY_SARCASTIC_STANDUP_PROMPT
import requests
import time
import json
import asyncio
import websockets
import logging


logger = logging.getLogger()


def append_question(user, message):
    # !/usr/bin/python
    queue_name = "chat_log"
    # API : name: str,  question: str, routing_queue: str = "video_response_queue", prompt: str = None
    msg = ChatQuestion(name=user, question=message, routing_queue="obs",
                       prompt=VERY_SARCASTIC_STANDUP_PROMPT, next_queue=None)
    add_to_queue(queue_name, msg.serialize())


class TwitchClient:
    def __init__(self, client_id, client_secret, refresh_token):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expiry = 0
        self.base_url = "https://api.twitch.tv/helix"
        self.oauth_url = "https://id.twitch.tv/oauth2"
        self.chat_url = "wss://irc-ws.chat.twitch.tv:443"
        self.refresh_access_token()

    def refresh_access_token(self):
        url = f"{self.oauth_url}/token"
        params = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        response = requests.post(url, params=params)
        response_data = response.json()
        self.access_token = response_data['access_token']
        self.refresh_token = response_data['refresh_token']
        self.token_expiry = time.time() + response_data['expires_in']
        logger.info("Access token refreshed")

    def ensure_token_valid(self):
        if time.time() > self.token_expiry:
            self.refresh_access_token()

    def get_headers(self):
        self.ensure_token_valid()
        return {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.access_token}"
        }

    async def connect_to_chat(self, channel_name):
        uri = self.chat_url
        async with websockets.connect(uri) as websocket:
            await websocket.send(f"PASS oauth:{self.access_token}")
            await websocket.send(f"NICK allojesuschrist")
            await websocket.send(f"JOIN #{channel_name}")
            print("connection done")
            while True:
                try:
                    response = await websocket.recv()
                    print(response)
                    self.handle_message(websocket, response, channel_name)
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed: {e}")
                    break

    def get_stream_chat(self, channel_name):
        async def read_chat():
            while True:
                try:
                    print("pipou")
                    await self.connect_to_chat(channel_name)
                except websockets.exceptions.InvalidStatusCode as e:
                    if e.status_code == 400:  # Bad Request, likely due to invalid token
                        print("Invalid token, refreshing...")
                        self.refresh_access_token()
                    else:
                        print(f"WebSocket error: {e}")
                        break
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    break
                await asyncio.sleep(5)  # Wait before attempting to reconnect

        asyncio.get_event_loop().run_until_complete(read_chat())

    def handle_message(self, websocket, message, channel_name):
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
        else:
            logger.info(f"Unhandled message: {message}")

    async def send_pong(self, pong_response, websocket):
        await websocket.send(pong_response)

    def handle_notice(self, msg):
        logger.info(msg)

    def handle_privmsg(self, websocket, message, channel):
        parts = message.split(":", 2)
        if len(parts) > 2:
            user_info, chat_message = parts[1], parts[2]
            username = user_info.split("!")[0]
        else:
            logger.info("Failure to parse message")
            return

        if chat_message.lstrip(" ").lower().startswith("!allo") or chat_message.lstrip(" ").lower().startswith("allo"):
            append_question(username, chat_message[5:])
            response = f"Merci {username}, ta question est en cours de traitement."
        elif chat_message.lower().startswith("jesus") or chat_message.lower().startswith("jésus"):
            response = f"{username}, si tu souhaites poser une question à Jésus. Utilises !allo avant ta question. Ex: !allo Salut, la forme ?."
        else:
            response = ""
        asyncio.create_task(self.send_message(websocket, channel, response))

    async def send_message(self, websocket, channel_name, msg):
        await websocket.send(f"PRIVMSG #{channel_name} :{msg}")

    def handle_join(self, message):
        parts = message.split("!")
        if len(parts) > 1:
            username = parts[0][1:]
            logger.info(f"{username} joined the chat")

    def handle_part(self, message):
        parts = message.split("!")
        if len(parts) > 1:
            username = parts[0][1:]
            logger.info(f"{username}")


# Usage
try:
    creds = json.load(open("./creds.json"))
except Exception as e:
    creds = os.environ

client_id = creds["client_id"]
client_secret = creds["client_secret"]
refresh_token = creds["refresh_token"]
twitch_client = TwitchClient(client_id, client_secret, refresh_token)
logger.info("Client initialized")
twitch_client.get_stream_chat("allojesuschrist")
