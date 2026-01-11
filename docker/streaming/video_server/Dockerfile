# Video Server Dockerfile
# Pure API proxy for OBS video streaming

FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn httpx

# Copy the video server module
COPY virtual_streamer/streaming/video_server/ /app/

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:5000/health')" || exit 1

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
