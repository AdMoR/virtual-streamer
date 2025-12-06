# ============================================================================
# Virtual Streamer Worker - Dockerfile with UV
# ============================================================================
# Optimized build using UV package manager for faster dependency installation

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv - the fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv



# Set UV environment variables
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency files for caching
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./

# Install dependencies using uv (much faster than pip)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-dev; \
    else \
        uv pip install --system -e .; \
    fi

# Install additional packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system git+https://github.com/elliottzheng/batch-face.git@master


# ============================================================================
# Runtime Stage
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates \
    libgirepository1.0-dev \
	&& rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

#RUN mkdir -p $WORKDIR && mkdir /root/.aws

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ./uv.lock .
COPY ./pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable



# Create necessary directories
RUN mkdir -p /data/out_video_folder \
             /data/temp \
             /data/characters \
             /data/audios \
             /data/clips \
             /root/.aws && \
    chmod -R 755 /data

# copy sources
COPY ./checkpoints/ ./checkpoints
COPY ./models ./models
COPY ./face_detection ./face_detection
COPY ./webservice.py .
COPY ./README.md .
COPY ./webservice_prod.py .
COPY ./virtual_streamer ./virtual_streamer

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    TEMP_DIR=/tmp/virtual_streamer

# Default command to run the unified API
# Note: Using single worker to ensure only one copy of ML models is loaded
CMD ["uv", "run", "python", "-m", "uvicorn", "virtual_streamer.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

# Default command to run the webservice with Gunicorn and the production config
#CMD ["uv", "run", "fastapi", "run", "/src/webservice.py"]
