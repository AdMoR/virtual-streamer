#!/bin/bash
# Simple wrapper to start the Twitch chat reader
#
# Usage:
#   ./scripts/start_twitch_chat.sh mychannel
#   ./scripts/start_twitch_chat.sh mychannel --api-url http://localhost:8000

CHANNEL=${1:-""}

if [ -z "$CHANNEL" ]; then
    echo "Usage: $0 <channel_name> [additional options]"
    echo ""
    echo "Examples:"
    echo "  $0 mychannel"
    echo "  $0 mychannel --api-url http://localhost:8000"
    echo "  $0 mychannel --stream-id live"
    exit 1
fi

# Shift to get remaining arguments
shift

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project directory
cd "$PROJECT_DIR"

# Run the Python script
python3 scripts/run_twitch_chat.py --channel "$CHANNEL" "$@"
