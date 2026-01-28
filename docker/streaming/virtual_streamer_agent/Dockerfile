# Virtual Streamer Agent Dockerfile
#
# Runs the ADK agent loop that controls Twitch streaming.
# The agent monitors chat, creates videos, and responds to viewers.
#
# Build context should be the repository root:
#   docker build -f docker/streaming/virtual_streamer_agent/Dockerfile .

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_agents.txt .
RUN pip install --no-cache-dir -r requirements_agents.txt

# Copy the application code
COPY virtual_streamer/ /app/virtual_streamer/
COPY configs/ /app/configs/
COPY setup.py .
COPY pyproject.toml .

# Install the package in editable mode
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app

# Default environment variables
ENV API_URL=http://virtual_streamer_api:8000
ENV STREAM_ID=default
ENV TOOLS_CONFIG=/app/configs/virtual_streamer_tools.yaml
ENV LOOP_INTERVAL=5
ENV MAX_CHAT_MESSAGES=100

# Run the agent loop
ENTRYPOINT ["python", "-m", "virtual_streamer.streaming.agent_loop.main"]
