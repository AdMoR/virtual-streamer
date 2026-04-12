"""
Flux Kontext Image Generation Server

FastAPI inference server for FLUX.1-Kontext-dev loaded from a bnb+hqq 4-bit
quantized checkpoint (transformer in BitsAndBytes NF4, text_encoder_2 in HQQ).
Supports:
  - POST /txt2image  — pure text-to-image generation
  - POST /edit       — image editing conditioned on one or several reference images

Images are exchanged as base64-encoded PNG strings in JSON bodies.

Environment variables
---------------------
KONTEXT_MODEL_ID : HuggingFace repo (or local path) of the quantized checkpoint.
                   Default: HighCWu/FLUX.1-Kontext-dev-bnb-hqq-4bit
HF_HOME          : HuggingFace cache directory (default system default).
SERVER_PORT      : Port to listen on (default: 8010).

Quantization requirements
-------------------------
    pip install git+https://github.com/huggingface/diffusers.git@main
    pip install "transformers>=4.53.1"
    pip install -U bitsandbytes hqq
"""

import base64
import io
import logging
import math
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from diffusers import FluxKontextPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KONTEXT_MODEL_ID = os.environ.get(
    "KONTEXT_MODEL_ID",
    "HighCWu/FLUX.1-Kontext-dev-bnb-hqq-4bit",
)

# ---------------------------------------------------------------------------
# Global pipeline holder
# ---------------------------------------------------------------------------

_pipeline: Optional[FluxKontextPipeline] = None


def _load_pipeline() -> FluxKontextPipeline:
    """Load the bnb+hqq 4-bit quantized FluxKontextPipeline."""

    logger.info("Loading quantized pipeline from %s...", KONTEXT_MODEL_ID)
    pipe = FluxKontextPipeline.from_pretrained(
        KONTEXT_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    logger.info("Pipeline ready.")
    return pipe


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    try:
        _pipeline = _load_pipeline()
    except Exception:
        logger.exception("Failed to load pipeline — server will start but /edit and /txt2image will 503")
    yield
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Flux Kontext Image Edit Server",
    description="bnb+hqq 4-bit quantized FLUX.1-Kontext image generation & editing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Txt2ImageRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the desired image")
    width: int = Field(default=680, ge=64, le=2048)
    height: int = Field(default=496, ge=64, le=2048)
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)
    seed: int = Field(default=-1, description="-1 for random")


class ImageEditRequest(BaseModel):
    prompt: str = Field(..., description="Editing instruction")
    images_b64: List[str] = Field(
        ...,
        min_length=1,
        description="One or more reference images as base64-encoded PNG/JPEG strings",
    )
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=2.5, ge=0.0, le=20.0)
    seed: int = Field(default=-1, description="-1 for random")


class ImageGenerationResponse(BaseModel):
    image_b64: str = Field(description="Generated image as base64-encoded PNG")
    width: int
    height: int
    seed: int


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _tile_images(images: List[Image.Image]) -> Image.Image:
    """Tile multiple images into a square(-ish) grid for multi-image conditioning."""
    n = len(images)
    if n == 1:
        return images[0]

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Resize all to the size of the first image
    w, h = images[0].size
    resized = [img.resize((w, h), Image.LANCZOS) for img in images]

    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for idx, img in enumerate(resized):
        row, col = divmod(idx, cols)
        grid.paste(img, (col * w, row * h))

    return grid


def _resolve_seed(seed: int) -> int:
    if seed == -1:
        return int(torch.randint(0, 2**31, (1,)).item())
    return seed


def _require_pipeline():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check server logs")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok" if _pipeline is not None else "model_not_loaded",
        "model_loaded": _pipeline is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_id": KONTEXT_MODEL_ID,
    }


@app.post("/txt2image", response_model=ImageGenerationResponse)
async def txt2image(req: Txt2ImageRequest):
    """Generate an image from a text prompt with no conditioning image."""
    _require_pipeline()

    seed = _resolve_seed(req.seed)
    generator = torch.Generator("cpu").manual_seed(seed)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            result = _pipeline(
                prompt=req.prompt,
                height=req.height,
                width=req.width,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
            )
    except Exception as exc:
        logger.exception("txt2image failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    img = result.images[0]
    return ImageGenerationResponse(
        image_b64=_pil_to_b64(img),
        width=img.width,
        height=img.height,
        seed=seed,
    )


@app.post("/edit", response_model=ImageGenerationResponse)
async def edit(req: ImageEditRequest):
    """
    Edit or transform one or several reference images according to a text prompt.

    When multiple images are supplied they are tiled into a grid and passed as
    a single conditioning image to the pipeline.
    """
    _require_pipeline()

    seed = _resolve_seed(req.seed)
    generator = torch.Generator("cpu").manual_seed(seed)

    try:
        pil_images = [_b64_to_pil(b64) for b64 in req.images_b64]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid base64 image: {exc}") from exc

    conditioning_image = _tile_images(pil_images)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            result = _pipeline(
                prompt=req.prompt,
                image=conditioning_image,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                generator=generator,
            )
    except Exception as exc:
        logger.exception("edit failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    img = result.images[0]
    return ImageGenerationResponse(
        image_b64=_pil_to_b64(img),
        width=img.width,
        height=img.height,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("SERVER_PORT", 8010)),
        log_level="info",
    )
