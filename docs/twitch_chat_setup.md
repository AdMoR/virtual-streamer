# Twitch Chat Video Generation Setup

This guide explains how to set up and run the Twitch chat reader that allows users to request video generation through chat commands.

## Overview

The Twitch chat reader connects to your Twitch channel and listens for commands from viewers. When users send commands, it communicates with the Virtual Streamer API to generate and queue videos.

## Features

- **!allo command**: Users can ask questions that trigger video generation
- **!generate command**: Users can request videos on specific topics using the active broadcast's story template
- **Queue management**: Automatically limits pending jobs (max 5) to prevent overload
- **Feedback system**: After videos play, users are prompted to provide feedback (+/-)
- **Automatic reconnection**: Handles connection issues and token refresh automatically

## Prerequisites

1. A Twitch account and channel
2. Twitch application credentials (Client ID, Client Secret)
3. A Twitch refresh token for authentication
4. Virtual Streamer API running and accessible

## Setup

### 1. Get Twitch Credentials

Create a Twitch application at https://dev.twitch.tv/console/apps:

1. Click "Register Your Application"
2. Set name (e.g., "Virtual Streamer Bot")
3. Set OAuth Redirect URL (e.g., `http://localhost:3000`)
4. Select category "Chat Bot"
5. Note your **Client ID** and **Client Secret**

### 2. Get Refresh Token

Use a tool like https://twitchtokengenerator.com/ to generate a refresh token with these scopes:
- `chat:read`
- `chat:edit`
- `channel:read:subscriptions` (optional)

### 3. Configure Credentials

Add your credentials to `.env.local` in the project root:

```bash
# Twitch Credentials
client_id="your_client_id_here"
client_secret="your_client_secret_here"
refresh_token="your_refresh_token_here"

# Optional Configuration
API_URL=http://localhost:8000
STREAM_ID=default
STORY_TEMPLATE_ID=your_template_id
```

**Note**: The refresh token will be automatically updated when it expires, and saved to `virtual_streamer/streaming/twitch/refresh_token.json`.

## Usage

### Option 1: Using the Shell Script (Easiest)

```bash
./scripts/start_twitch_chat.sh yourchannel
```

### Option 2: Using the Python Script

```bash
# Basic usage
python scripts/run_twitch_chat.py --channel yourchannel

# With custom API URL and stream ID
python scripts/run_twitch_chat.py \
  --channel yourchannel \
  --api-url http://localhost:8000 \
  --stream-id live

# With story template
python scripts/run_twitch_chat.py \
  --channel yourchannel \
  --story-template-id your_template_id
```

### Option 3: Using Docker

```bash
# Build and run in Docker
python scripts/run_twitch_chat.py --channel yourchannel --docker
```

### Option 4: Direct Python Module

```bash
# Export environment variables first
eval "$(python scripts/bootstrap_secrets.py --export-shell)"
export TWITCH_CHANNEL=yourchannel

# Run the chat reader
python -m virtual_streamer.streaming.twitch.chat_reader
```

## Bootstrap Secrets Script

The `bootstrap_secrets.py` script helps manage credentials:

```bash
# Validate credentials (shows masked values)
python scripts/bootstrap_secrets.py --validate

# Export as shell variables
eval "$(python scripts/bootstrap_secrets.py --export-shell)"

# Save to JSON file
python scripts/bootstrap_secrets.py --output secrets.json

# Override channel
python scripts/bootstrap_secrets.py --channel yourchannel --validate
```

## Chat Commands

Once running, viewers can use these commands in your Twitch chat:

### !allo <question>
Submit a question for video generation.

**Example:**
```
!allo Comment fonctionne la photosynthèse?
```

The bot will acknowledge the question and submit it to the API for processing.

### !generate <title>
Generate a video on a specific topic using the active broadcast's story template.

**Example:**
```
!generate Fred découvre les mystères de l'océan
```

**Features:**
- Uses the story template from the active programmation
- Limits queue to max 5 pending jobs
- Responds with error if queue is full or no programmation is active

### Feedback (+/-)
After your video plays, the bot will prompt you for feedback. Simply reply with your feedback in the next message.

**Example:**
```
Bot: @username, ta video vient de passer! + ou - ?
User: Super vidéo! Très instructif
```

The bot captures the raw message as feedback.

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TWITCH_CLIENT_ID` | Yes | - | Twitch application client ID |
| `TWITCH_CLIENT_SECRET` | Yes | - | Twitch application client secret |
| `TWITCH_REFRESH_TOKEN` | Yes | - | Twitch refresh token |
| `TWITCH_CHANNEL` | Yes | - | Channel name to connect to |
| `TWITCH_BOT_USERNAME` | No | virtualstreamerbot | Bot username |
| `API_URL` | No | http://virtual_streamer_api:8000 | API endpoint |
| `STREAM_ID` | No | default | Stream ID for videos |
| `STORY_TEMPLATE_ID` | No | - | Default story template |

### Token Management

- The refresh token is automatically refreshed when it expires
- New refresh tokens are saved to `virtual_streamer/streaming/twitch/refresh_token.json`
- A background thread monitors token expiration and refreshes proactively

## Troubleshooting

### "Missing required environment variables"
Make sure your `.env.local` file contains `client_id`, `client_secret`, and `refresh_token`.

### "Authentication failed"
Your refresh token may have expired or been revoked. Generate a new one from the Twitch token generator.

### "Connection closed"
This is normal - the bot will automatically reconnect. If it persists, check your network connection and Twitch API status.

### "API error"
Ensure the Virtual Streamer API is running and accessible at the configured `API_URL`.

### Queue full errors
The system limits pending video generation jobs to 5. Wait for some to complete before submitting new requests.

## Architecture

```
Twitch Chat
    ↓
Chat Reader (WebSocket)
    ↓
Command Parser
    ↓
Virtual Streamer API
    ↓
Video Generation Queue
    ↓
Stream Playlist
```

The chat reader:
1. Connects to Twitch IRC via WebSocket
2. Authenticates using OAuth tokens
3. Listens for PRIVMSG events (chat messages)
4. Parses commands and extracts parameters
5. Makes HTTP requests to the Virtual Streamer API
6. Monitors played videos and prompts for feedback
7. Handles token refresh automatically

## Development

### Running in Development

```bash
# Install dependencies
pip install -r requirements_cpu.txt

# Run with debug logging
export LOG_LEVEL=DEBUG
python scripts/run_twitch_chat.py --channel yourchannel
```

### Testing

```bash
# Test bootstrap script
python scripts/bootstrap_secrets.py --validate

# Test with mock environment
export TWITCH_CLIENT_ID=test
export TWITCH_CLIENT_SECRET=test
export TWITCH_REFRESH_TOKEN=test
export TWITCH_CHANNEL=testchannel
python -m virtual_streamer.streaming.twitch.chat_reader
```

## Security Notes

- **Never commit** `.env.local` or `secrets.json` to version control
- The `.env.local` file contains sensitive credentials
- Refresh tokens are automatically rotated and saved securely
- Use read-only API tokens when possible
- Consider using environment variable management tools in production

## Production Deployment

For production, consider:

1. **Docker Deployment**: Use the provided Dockerfile
2. **Environment Variables**: Use Docker secrets or a secrets manager
3. **Monitoring**: Log aggregation and health checks
4. **High Availability**: Multiple instances with load balancing
5. **Rate Limiting**: Implement per-user rate limits

Example Docker Compose service:

```yaml
services:
  twitch-chat:
    build:
      context: .
      dockerfile: docker/streaming/twitch/Dockerfile
    environment:
      - TWITCH_CLIENT_ID=${TWITCH_CLIENT_ID}
      - TWITCH_CLIENT_SECRET=${TWITCH_CLIENT_SECRET}
      - TWITCH_REFRESH_TOKEN=${TWITCH_REFRESH_TOKEN}
      - TWITCH_CHANNEL=${TWITCH_CHANNEL}
      - API_URL=http://virtual_streamer_api:8000
      - STREAM_ID=default
    restart: unless-stopped
    depends_on:
      - virtual_streamer_api
```

## License

See the main project LICENSE file.
