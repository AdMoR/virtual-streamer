# ============================================================================
# Virtual Streamer Worker - Dockerfile with UV
# ============================================================================
# Optimized build using UV package manager for faster dependency installation

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9

# Install build dependencies
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

# Set UV environment variables
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency files for caching
COPY pyproject.toml ./
COPY uv.lock* ./

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

# Create necessary directories
RUN mkdir -p /data/out_video_folder \
             /data/temp \
             /data/characters \
             /data/audios \
             /data/clips \
             /root/.aws && \
    chmod -R 755 /data

# Copy application code
COPY . .

# Install the package
RUN python -m pip install --no-deps -e .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    TEMP_DIR=/tmp/virtual_streamer

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Expose port
EXPOSE 8000

# Default command to run the unified API
# Note: Using single worker to ensure only one copy of ML models is loaded
CMD ["python", "-m", "uvicorn", "virtual_streamer.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
