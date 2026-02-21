"""
Twitch OAuth Token Generator

Runs the OAuth Authorization Code flow to obtain a refresh token for the
Twitch chat bot. Spins up a temporary local HTTP server to capture the
callback automatically — no manual code pasting needed.

Usage:
    python virtual_streamer/utils/get_refresh_token.py

Prerequisites:
    - Twitch app registered at https://dev.twitch.tv/console/apps
    - OAuth Redirect URL set to http://localhost:3000 in the Twitch app settings
    - client_id and client_secret configured in .env.local
"""

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.bootstrap_secrets import find_env_file, parse_env_file

REDIRECT_URI = "http://localhost:3000"
REDIRECT_PORT = 3000
SCOPES = "chat:read chat:edit user:bot user:read:chat user:write:chat"


def load_client_credentials():
    """Load client_id and client_secret from .env.local."""
    env_file = find_env_file()
    if not env_file:
        print("Error: Could not find .env.local or .env file in project root.")
        print(f"  Looked in: {PROJECT_ROOT}")
        sys.exit(1)

    creds = parse_env_file(env_file)
    client_id = creds.get("TWITCH_CLIENT_ID")
    client_secret = creds.get("TWITCH_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("Error: client_id and client_secret must be set in .env.local")
        print("  Expected format:")
        print('  client_id="your_client_id"')
        print('  client_secret="your_client_secret"')
        sys.exit(1)

    return client_id, client_secret


def update_env_local(refresh_token: str):
    """Update the refresh_token line in .env.local."""
    env_file = find_env_file()
    if not env_file:
        return

    lines = env_file.read_text().splitlines()
    updated = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("refresh_token"):
            lines[i] = f'refresh_token="{refresh_token}"'
            updated = True
            break

    if not updated:
        lines.append(f'refresh_token="{refresh_token}"')

    env_file.write_text("\n".join(lines) + "\n")
    print(f"  Updated {env_file}")


def update_creds_json(refresh_token: str):
    """Update creds.json with the new refresh token."""
    creds_path = PROJECT_ROOT / "creds.json"
    creds = {}
    if creds_path.exists():
        with open(creds_path) as f:
            creds = json.load(f)

    creds["refresh_token"] = refresh_token
    with open(creds_path, "w") as f:
        json.dump(creds, f, indent=2)
    print(f"  Updated {creds_path}")


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback."""

    authorization_code = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            CallbackHandler.authorization_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization successful!</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )
        elif "error" in params:
            error = params.get("error", ["unknown"])[0]
            desc = params.get("error_description", [""])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h2>Authorization failed</h2>"
                f"<p>Error: {error}</p><p>{desc}</p></body></html>".encode()
            )
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def wait_for_callback():
    """Start a local server and wait for the OAuth callback."""
    server = HTTPServer(("localhost", REDIRECT_PORT), CallbackHandler)
    server.timeout = 120  # 2 minute timeout

    print(f"\nWaiting for authorization callback on {REDIRECT_URI} ...")
    print("(will timeout after 2 minutes)\n")

    while CallbackHandler.authorization_code is None:
        server.handle_request()
        if CallbackHandler.authorization_code is None:
            # Timeout or irrelevant request
            break

    server.server_close()
    return CallbackHandler.authorization_code


def exchange_code_for_tokens(client_id, client_secret, code):
    """Exchange the authorization code for access and refresh tokens."""
    response = requests.post(
        "https://id.twitch.tv/oauth2/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
        },
        timeout=30,
    )

    if response.status_code != 200:
        print(f"Error: Token exchange failed ({response.status_code})")
        print(f"  Response: {response.text}")
        sys.exit(1)

    return response.json()


def main():
    print("=== Twitch OAuth Token Generator ===\n")

    # Load credentials
    client_id, client_secret = load_client_credentials()
    print(f"Client ID: {client_id[:8]}...{client_id[-4:]}")

    # Build authorization URL
    auth_url = (
        f"https://id.twitch.tv/oauth2/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES}"
    )

    print(f"\nOpening browser for Twitch authorization...")
    print(f"  URL: {auth_url}\n")
    webbrowser.open(auth_url)

    # Wait for callback
    code = wait_for_callback()
    if not code:
        print("Error: Did not receive authorization code (timeout or denied).")
        sys.exit(1)

    print(f"Authorization code received.")

    # Exchange for tokens
    print("Exchanging code for tokens...")
    token_data = exchange_code_for_tokens(client_id, client_secret, code)

    access_token = token_data["access_token"]
    refresh_token = token_data["refresh_token"]
    expires_in = token_data.get("expires_in", "unknown")

    print(f"\nTokens obtained successfully!")
    print(f"  Access token: {access_token[:8]}...{access_token[-4:]}")
    print(f"  Refresh token: {refresh_token[:8]}...{refresh_token[-4:]}")
    print(f"  Expires in: {expires_in} seconds")

    # Save tokens
    print(f"\nSaving refresh token...")
    update_env_local(refresh_token)
    update_creds_json(refresh_token)

    print(f"\nDone! You can now run the Twitch chat bot:")
    print(f"  python scripts/run_twitch_chat.py --channel <your_channel>")


if __name__ == "__main__":
    main()
