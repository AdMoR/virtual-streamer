# Video Server Dockerfile
# API proxy + Atari game agent

FROM python:3.11-slim

WORKDIR /app

# System libs for opencv and ALE
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Expose port
EXPOSE 5000

# Copy project files
COPY pyproject.toml ./
COPY requirements.txt ./


# Install the full package (includes ale-py and shimmy[atari])
RUN uv pip install --system --no-cache .

COPY configs/ ./configs/
COPY virtual_streamer/ ./virtual_streamer/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:5000/health')" || exit 1

# Run the server
CMD ["uvicorn", "virtual_streamer.streaming.video_server.app:app", "--host", "0.0.0.0", "--port", "5000"]
