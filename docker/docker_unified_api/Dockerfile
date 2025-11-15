# ============================================================================
# Virtual Streamer Unified API - Dockerfile with UV
# ============================================================================
# Multi-stage build for optimal image size and build speed using UV package manager
# UV is 10-100x faster than pip for dependency resolution and installation

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    python3.9 \
    python3.9-dev \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install uv - the fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Make python3.9 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Set UV environment variables for optimal performance
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies using uv (much faster than pip)
# Using pyproject.toml directly if uv.lock exists, otherwise requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-dev; \
    else \
        uv pip install --system -e .; \
    fi

# Install additional packages that aren't in pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system git+https://github.com/elliottzheng/batch-face.git@master


# ============================================================================
# Runtime Stage - Smaller final image
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.9 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages

# Set working directory
WORKDIR /src

# Create necessary directories with proper permissions
RUN mkdir -p /data/out_video_folder \
             /data/temp \
             /data/characters \
             /data/audios \
             /data/clips \
             /root/.aws && \
    chmod -R 755 /data

# Copy application code
COPY . .

# Install the package itself (already built in builder stage)
RUN python -m pip install --no-deps -e .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    TEMP_DIR=/tmp/virtual_streamer

# Healthcheck to monitor API health
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Expose API port
EXPOSE 8000

# Run unified API with single worker (ensures single ML model instance)
CMD ["python", "-m", "uvicorn", "virtual_streamer.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
