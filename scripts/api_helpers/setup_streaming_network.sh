#!/bin/bash
#
# Setup Streaming Network
#
# This script creates a shared Docker network and connects the main
# Virtual Streamer services to it, allowing the streaming stack to
# communicate with the main API, MySQL, and MinIO.
#
# Usage:
#   ./scripts/setup_streaming_network.sh
#
# Prerequisites:
#   - Docker must be running
#   - Main services should be running (docker compose up)

set -e

NETWORK_NAME="vs_streaming"

echo "Setting up streaming network..."

# Create the shared network if it doesn't exist
if docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
    echo "Network '${NETWORK_NAME}' already exists."
else
    echo "Creating network '${NETWORK_NAME}'..."
    docker network create "${NETWORK_NAME}"
    echo "Network created."
fi

# Get the project name (usually the directory name)
# Docker Compose uses this as prefix for container names
PROJECT_NAME=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')

# Alternative: try to detect from running containers
if ! docker ps --format '{{.Names}}' | grep -q "${PROJECT_NAME}"; then
    # Try common patterns
    for pattern in "virtual-streamer" "virtual_streamer" "virtualstreamer"; do
        if docker ps --format '{{.Names}}' | grep -q "${pattern}"; then
            PROJECT_NAME="${pattern}"
            break
        fi
    done
fi

echo "Detected project name: ${PROJECT_NAME}"

# List of main services that need to be connected
SERVICES=(
    "virtual_streamer_api"
    "mysql"
    "minio"
)

# Connect each service to the streaming network
for service in "${SERVICES[@]}"; do
    # Try different container name patterns
    for container in "${PROJECT_NAME}-${service}-1" "${PROJECT_NAME}_${service}_1" "${service}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            echo "Connecting ${container} to ${NETWORK_NAME}..."
            docker network connect "${NETWORK_NAME}" "${container}" 2>/dev/null && \
                echo "  Connected!" || \
                echo "  Already connected or failed."
            break
        fi
    done
done

echo ""
echo "Network setup complete!"
echo ""
echo "Next steps:"
echo "1. Start the streaming stack:"
echo "   docker compose -f compose_streaming.yml up -d"
echo ""
echo "2. Access OBS via VNC:"
echo "   - VNC: vnc://localhost:5901"
echo "   - Web: http://localhost:6901"
echo ""
echo "3. The video player is available at:"
echo "   http://localhost:5000"
