# Twitch Chat

## Read chat messages

To read recent messages from the Twitch chat, use `get_chat_messages`.

```
get_chat_messages(limit=50)
```

Returns a list of message objects with the sender username, text, and timestamp.

To only see messages that mention the bot:

```
get_chat_messages(mentions_only=True)
```

You can also read the last 50 messages as a resource without calling a tool:

```
virtual-streamer://chat/recent
```

---

## Send a message to chat

To post a message to the Twitch channel, use `send_twitch_message`.

```
send_twitch_message(message="Hello everyone!")
```

Messages are automatically truncated to 500 characters. Returns `{"success": true}` on success or `{"success": false, "error": "..."}` if Twitch is not connected.
