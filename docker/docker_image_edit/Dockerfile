# Flux Kontext Image Edit Server
#
# Base: PyTorch 2.5 + CUDA 12.4 runtime (includes cuDNN 9)
# The model weights are NOT baked in — mount a HuggingFace cache volume so
# they are downloaded once and reused across container restarts.
#
# Build:
#   docker build -t flux-kontext-server -f docker/docker_image_edit/Dockerfile .
#
# Run:
#   docker run --gpus all \
#     -p 8010:8010 \
#     -v huggingface-cache:/root/.cache/huggingface \
#     [-e GGUF_MODEL_PATH=/models/flux1-kontext-dev-Q4_K_M.gguf] \
#     flux-kontext-server

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (installed before copying app code for better layer caching)
COPY docker/docker_image_edit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code — only the image generation server
COPY virtual_streamer/image_generation/flux_kontext_server/ ./flux_kontext_server/

EXPOSE 8010

CMD ["uvicorn", "flux_kontext_server.app:app", \
     "--host", "0.0.0.0", "--port", "8010", "--log-level", "info"]
