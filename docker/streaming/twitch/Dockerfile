# Twitch Chat Reader Dockerfile
# Reads chat and submits questions to the API

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_cpu.txt .
RUN pip install --no-cache-dir -r requirements_cpu.txt

# Copy the streaming module
COPY virtual_streamer/streaming/ /app/streaming/
COPY setup.py .

# Install package
RUN pip install -e .

# Run the chat reader
ENTRYPOINT ["python", "-m", "streaming.twitch.chat_reader"]
