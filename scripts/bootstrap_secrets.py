#!/usr/bin/env python3
"""
Bootstrap secrets for the Virtual Streamer Twitch integration.

This script reads credentials from .env.local and sets them up for use by
the Twitch chat reader. It can either:
1. Export them as environment variables (for shell use)
2. Return them as a dictionary (for programmatic use)
3. Write them to a JSON file

Usage:
    # Export as shell commands (eval in your shell)
    python scripts/bootstrap_secrets.py --export-shell

    # Save to a JSON file
    python scripts/bootstrap_secrets.py --output secrets.json

    # Just validate and print
    python scripts/bootstrap_secrets.py --validate
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional


def find_env_file() -> Optional[Path]:
    """Find the .env.local file in the project root."""
    current = Path(__file__).resolve().parent.parent
    env_file = current / ".env.local"

    if env_file.exists():
        return env_file

    # Try .env as fallback
    env_file = current / ".env"
    if env_file.exists():
        return env_file

    return None


def parse_env_file(env_file: Path) -> Dict[str, str]:
    """Parse the .env file and extract Twitch credentials."""
    credentials = {}

    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Handle different formats
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'").strip(",")

                # Map credentials
                if key == 'client_id':
                    credentials['TWITCH_CLIENT_ID'] = value
                elif key == 'client_secret':
                    credentials['TWITCH_CLIENT_SECRET'] = value
                elif key == 'refresh_token':
                    credentials['TWITCH_REFRESH_TOKEN'] = value
                elif key.startswith('TWITCH_'):
                    credentials[key] = value
                elif key in ['API_URL', 'STREAM_ID', 'STORY_TEMPLATE_ID']:
                    credentials[key] = value

    return credentials


def load_secrets() -> Dict[str, str]:
    """Load secrets from environment file."""
    env_file = find_env_file()

    if not env_file:
        print("Error: Could not find .env.local or .env file", file=sys.stderr)
        sys.exit(1)

    print(f"Loading secrets from: {env_file}", file=sys.stderr)
    credentials = parse_env_file(env_file)

    # Validate required fields
    required = ['TWITCH_CLIENT_ID', 'TWITCH_CLIENT_SECRET', 'TWITCH_REFRESH_TOKEN']
    missing = [field for field in required if field not in credentials]

    if missing:
        print(f"Error: Missing required fields: {', '.join(missing)}", file=sys.stderr)
        print("Please ensure your .env.local file contains:", file=sys.stderr)
        print("  client_id=<your_client_id>", file=sys.stderr)
        print("  client_secret=<your_client_secret>", file=sys.stderr)
        print("  refresh_token=<your_refresh_token>", file=sys.stderr)
        sys.exit(1)

    # Add defaults for optional fields
    credentials.setdefault('API_URL', 'http://localhost:8000')
    credentials.setdefault('STREAM_ID', 'default')
    credentials.setdefault('TWITCH_CHANNEL', '')

    return credentials


def export_shell(credentials: Dict[str, str]) -> None:
    """Print export statements for shell evaluation."""
    for key, value in credentials.items():
        # Escape single quotes in value
        safe_value = value.replace("'", "'\\''")
        print(f"export {key}='{safe_value}'")


def save_json(credentials: Dict[str, str], output_path: str) -> None:
    """Save credentials to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(credentials, f, indent=2)
    print(f"Secrets saved to: {output_path}", file=sys.stderr)


def validate(credentials: Dict[str, str]) -> None:
    """Validate and print credentials (masking sensitive parts)."""
    print("Loaded credentials:", file=sys.stderr)
    for key, value in credentials.items():
        if any(secret in key.lower() for secret in ['secret', 'token', 'key']):
            # Mask sensitive values
            if len(value) > 8:
                masked = value[:4] + '*' * (len(value) - 8) + value[-4:]
            else:
                masked = '*' * len(value)
            print(f"  {key}: {masked}", file=sys.stderr)
        else:
            print(f"  {key}: {value}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap secrets for Virtual Streamer Twitch integration'
    )
    parser.add_argument(
        '--export-shell',
        action='store_true',
        help='Print export statements for shell evaluation (eval "$(python bootstrap_secrets.py --export-shell)")'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save credentials to a JSON file'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate and print credentials (with masking)'
    )
    parser.add_argument(
        '--channel',
        type=str,
        help='Override Twitch channel name'
    )

    args = parser.parse_args()

    # Load credentials
    credentials = load_secrets()

    # Override channel if provided
    if args.channel:
        credentials['TWITCH_CHANNEL'] = args.channel

    # Perform requested action
    if args.export_shell:
        export_shell(credentials)
    elif args.output:
        save_json(credentials, args.output)
    elif args.validate or not any([args.export_shell, args.output]):
        validate(credentials)


if __name__ == '__main__':
    main()
