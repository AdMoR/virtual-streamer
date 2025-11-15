#!/bin/bash
# Start the Streamlit UI for Video Generation

# Default configuration
export API_BASE_URL=${API_BASE_URL:-http://localhost:8000/api/v1}

echo "============================================"
echo "Virtual Streamer - Video Generation UI"
echo "============================================"
echo "API Base URL: $API_BASE_URL"
echo "============================================"
echo

# Start Streamlit
streamlit run apps/video_generation_ui.py




