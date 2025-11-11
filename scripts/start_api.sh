#!/bin/bash
# Start the Virtual Streamer API server

# Default configuration
export DATA_DIR=${DATA_DIR:-/data}
export TEMP_DIR=${TEMP_DIR:-./temp}
export ENTITY_SERVICE_HOST=${ENTITY_SERVICE_HOST:-localhost}
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}

# Ensure temp directory exists
mkdir -p "$TEMP_DIR"

echo "============================================"
echo "Virtual Streamer API Server"
echo "============================================"
echo "Data directory: $DATA_DIR"
echo "Temp directory: $TEMP_DIR"
echo "API endpoint: http://$API_HOST:$API_PORT"
echo "Docs: http://$API_HOST:$API_PORT/docs"
echo "============================================"
echo

# Start the server
uvicorn virtual_streamer.api.main:app \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --reload \
    --log-level info

