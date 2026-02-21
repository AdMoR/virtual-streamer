#!/usr/bin/env python3
"""
Run the Twitch Chat Reader process.

This script bootstraps the Twitch chat reader with proper credentials and
configuration. It loads secrets from .env.local and starts the chat reader
to listen for commands in Twitch chat.

The chat reader supports the following commands:
- !allo <question>: Submit a question for video generation
- !generate <title>: Generate a video using the active broadcast's story template
- Feedback: After a video plays, users can respond with + or - for feedback

Usage:
    # Run with default settings
    python scripts/run_twitch_chat.py

    # Specify a channel
    python scripts/run_twitch_chat.py --channel yourchannel

    # Specify API URL and stream ID
    python scripts/run_twitch_chat.py --api-url http://localhost:8000 --stream-id mystream

    # Run in Docker
    python scripts/run_twitch_chat.py --docker

Environment variables (loaded from .env.local by default):
    TWITCH_CLIENT_ID: Twitch application client ID
    TWITCH_CLIENT_SECRET: Twitch application client secret
    TWITCH_REFRESH_TOKEN: Twitch refresh token for authentication
    TWITCH_CHANNEL: Channel name to connect to
    API_URL: Virtual Streamer API URL (default: http://localhost:8000)
    STREAM_ID: Stream ID for video generation (default: default)
    STORY_TEMPLATE_ID: Optional story template ID
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.bootstrap_secrets import load_secrets


def run_chat_reader():
    """Import and run the chat reader."""
    from virtual_streamer.streaming.twitch.chat_reader import main
    main()


def run_docker(args):
    """Run the chat reader in Docker."""
    import subprocess

    # Build docker image
    print("Building Docker image...")
    subprocess.run([
        'docker', 'build',
        '-f', 'docker/streaming/twitch/Dockerfile',
        '-t', 'virtual-streamer-twitch',
        '.'
    ], check=True)

    # Prepare environment variables
    env_vars = []
    credentials = load_secrets()

    if args.channel:
        credentials['TWITCH_CHANNEL'] = args.channel
    if args.api_url:
        credentials['API_URL'] = args.api_url
    if args.stream_id:
        credentials['STREAM_ID'] = args.stream_id
    if args.story_template_id:
        credentials['STORY_TEMPLATE_ID'] = args.story_template_id

    for key, value in credentials.items():
        env_vars.extend(['-e', f'{key}={value}'])

    # Run container
    print("Starting Twitch chat reader container...")
    subprocess.run([
        'docker', 'run',
        '--rm',
        '--name', 'virtual-streamer-twitch',
        *env_vars,
        'virtual-streamer-twitch'
    ], check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Run the Twitch Chat Reader for Virtual Streamer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run locally with default settings
  python scripts/run_twitch_chat.py --channel mychannel

  # Run with custom API and stream
  python scripts/run_twitch_chat.py --channel mychannel --api-url http://api:8000 --stream-id live

  # Run in Docker
  python scripts/run_twitch_chat.py --channel mychannel --docker

Commands available in Twitch chat:
  !allo <question>     - Submit a question for video generation
  !generate <title>    - Generate a video with the specified title
  +/-                  - Provide feedback after your video plays
        """
    )

    parser.add_argument(
        '--channel',
        type=str,
        help='Twitch channel name to connect to'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        help='Virtual Streamer API URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--stream-id',
        type=str,
        help='Stream ID for video generation (default: default)'
    )
    parser.add_argument(
        '--story-template-id',
        type=str,
        help='Story template ID to use for video generation'
    )
    parser.add_argument(
        '--docker',
        action='store_true',
        help='Run in Docker container'
    )
    parser.add_argument(
        '--bot-username',
        type=str,
        help='Bot username (default: virtualstreamerbot)'
    )

    args = parser.parse_args()

    # Run in Docker if requested
    if args.docker:
        run_docker(args)
        return

    # Load secrets from .env.local
    print("Loading credentials from .env.local...")
    credentials = load_secrets()

    # Override with command line arguments
    if args.channel:
        credentials['TWITCH_CHANNEL'] = args.channel
    if args.api_url:
        credentials['API_URL'] = args.api_url
    if args.stream_id:
        credentials['STREAM_ID'] = args.stream_id
    if args.story_template_id:
        credentials['STORY_TEMPLATE_ID'] = args.story_template_id
    if args.bot_username:
        credentials['TWITCH_BOT_USERNAME'] = args.bot_username

    # Validate required fields
    if not credentials.get('TWITCH_CHANNEL'):
        print("Error: TWITCH_CHANNEL is required. Use --channel or set it in .env.local")
        sys.exit(1)

    # Set environment variables
    for key, value in credentials.items():
        os.environ[key] = value
        if 'secret' not in key.lower() and 'token' not in key.lower():
            print(f"  {key}: {value}")

    print(f"\nStarting Twitch chat reader for channel: {credentials['TWITCH_CHANNEL']}")
    print(f"API URL: {credentials['API_URL']}")
    print(f"Stream ID: {credentials['STREAM_ID']}")
    print("\nAvailable commands:")
    print("  !allo <question>     - Submit a question for video generation")
    print("  !generate <title>    - Generate a video with the specified title")
    print("  +/-                  - Provide feedback after your video plays")
    print("\nPress Ctrl+C to stop\n")

    # Run the chat reader
    try:
        run_chat_reader()
    except KeyboardInterrupt:
        print("\n\nStopping chat reader...")
        sys.exit(0)


if __name__ == '__main__':
    main()
