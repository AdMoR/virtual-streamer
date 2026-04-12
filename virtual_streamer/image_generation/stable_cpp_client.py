"""
Stable Diffusion WebUI-compatible Image Generation Client

Client for a stable-diffusion.cpp server using the AUTOMATIC1111-compatible
WebUI API surface.

Abstract interface
------------------
:class:`ImageGenerationClientInterface` defines two operations:

  - :meth:`txt2image` — generate an image from a text prompt only
  - :meth:`image_edit` — edit / transform a reference image guided by a text
                          prompt (img2img)

Both methods accept typed Pydantic parameter models and return an
:class:`ImageGenerationResult`.

Concrete implementation
-----------------------
:class:`StableDiffusionCppClient` implements the interface using the WebUI
compatibility endpoints:

  POST /sdapi/v1/txt2img  — text-to-image (synchronous)
  POST /sdapi/v1/img2img  — image-to-image (synchronous)

Usage
-----
    from virtual_streamer.image_generation.stable_cpp_client import (
        StableDiffusionCppClient, StableDiffusionCppConfig,
        Txt2ImageParams, ImageEditParams,
    )

    # Text-to-image
    async with StableDiffusionCppClient(StableDiffusionCppConfig()) as client:
        result = await client.txt2image(
            Txt2ImageParams(prompt="A fox in a snowy forest"),
            output_dir="./output",
        )
        print(result.image_path)

    # Image editing (img2img)
    async with StableDiffusionCppClient(StableDiffusionCppConfig()) as client:
        result = await client.image_edit(
            ImageEditParams(
                prompt="Make the sky purple and add stars",
                image_paths=["photo.jpg"],
                denoising_strength=0.75,
            ),
            output_dir="./output",
        )
        print(result.image_path)
"""

import asyncio
import base64
import io
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import httpx
from PIL import Image
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

class StableDiffusionCppConfig(BaseModel):
    """Connection settings for the stable-diffusion.cpp inference server."""

    server_url: str = Field(
        default="http://gx10-cbc5:1234",
        description="Base URL of the running stable-diffusion.cpp server",
    )
    timeout: float = Field(
        default=300.0,
        description="HTTP timeout in seconds",
    )


# =============================================================================
# Parameter models
# =============================================================================

class Txt2ImageParams(BaseModel):
    """Parameters for text-to-image generation (POST /sdapi/v1/txt2img)."""

    prompt: str = Field(description="Text prompt describing the desired image")
    negative_prompt: str = Field(default="")
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=200)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    seed: int = Field(default=-1, description="-1 for a random seed")
    sampler_name: str = Field(default="Euler a")
    scheduler: str = Field(default="Automatic")
    batch_size: int = Field(default=1, ge=1, le=16)


class ImageEditParams(BaseModel):
    """Parameters for img2img editing (POST /sdapi/v1/img2img)."""

    prompt: str = Field(description="Editing instruction")
    image_paths: List[str] = Field(
        description="Local paths to one or more reference images (tiled when multiple)",
        min_length=1,
    )
    negative_prompt: str = Field(default="")
    denoising_strength: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Denoising strength: 0 = no change, 1 = full redraw",
    )
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=200)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    seed: int = Field(default=-1, description="-1 for a random seed")
    sampler_name: str = Field(default="Euler a")
    scheduler: str = Field(default="Automatic")


# =============================================================================
# Result model
# =============================================================================

class ImageGenerationResult(BaseModel):
    """Result returned after a successful generation or edit."""

    image_path: str = Field(description="Local path to the saved output image")
    width: int
    height: int
    seed: int
    prompt_id: str = Field(description="Unique identifier for this generation")


# =============================================================================
# Abstract interface
# =============================================================================

class ImageGenerationClientInterface(ABC):
    """
    Abstract interface for image generation / editing clients.

    Implementations must provide :meth:`txt2image`, :meth:`image_edit`, and
    :meth:`close`.  The class also acts as an async context manager.
    """

    @abstractmethod
    async def txt2image(
        self,
        params: Txt2ImageParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Generate an image from *params.prompt* with no conditioning image.

        Args:
            params:     Generation parameters (prompt, resolution, steps…)
            output_dir: Local directory where the output PNG will be saved.

        Returns:
            :class:`ImageGenerationResult` with the local image path and metadata.
        """
        ...

    @abstractmethod
    async def image_edit(
        self,
        params: ImageEditParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Edit a reference image guided by a text prompt (img2img).

        When multiple images are supplied they are tiled side-by-side before
        being sent to the server.

        Args:
            params:     Edit parameters (prompt, image_paths, strength, steps…)
            output_dir: Local directory where the output PNG will be saved.

        Returns:
            :class:`ImageGenerationResult` with the local image path and metadata.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (HTTP connections, …)."""
        ...

    async def __aenter__(self) -> "ImageGenerationClientInterface":
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()


# =============================================================================
# Stable Diffusion WebUI-compatible implementation
# =============================================================================

class StableDiffusionCppClient(ImageGenerationClientInterface):
    """
    Image generation / editing client using the WebUI-compatible API of a
    stable-diffusion.cpp server.

    Both endpoints are synchronous — the server blocks until generation is
    complete and returns the image directly in the response body.

    The HTTP client is created lazily on first use and reused across calls.
    """

    def __init__(self, config: Optional[StableDiffusionCppConfig] = None) -> None:
        self.config = config or StableDiffusionCppConfig()
        self._http: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.config.server_url,
                timeout=self.config.timeout,
            )
        return self._http

    @staticmethod
    def _images_to_b64_grid(paths: List[str]) -> str:
        """
        Load one or more images and return a single base64-encoded PNG.

        Multiple images are tiled side-by-side into a horizontal strip.
        """
        imgs = [Image.open(p).convert("RGB") for p in paths]
        if len(imgs) == 1:
            combined = imgs[0]
        else:
            total_w = sum(i.width for i in imgs)
            max_h = max(i.height for i in imgs)
            combined = Image.new("RGB", (total_w, max_h))
            x = 0
            for img in imgs:
                combined.paste(img, (x, 0))
                x += img.width

        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _save_b64_image(b64: str, output_dir: str) -> tuple[str, int, int]:
        """Decode a base64 PNG, save to *output_dir*, return (path, w, h)."""
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = f"{uuid.uuid4().hex}.png"
        fpath = out / fname
        img.save(fpath)
        return str(fpath), img.width, img.height

    @staticmethod
    def _parse_seed(body: dict, fallback: int) -> int:
        """
        Extract the seed from the response.

        The WebUI ``info`` field is currently always an empty string.
        The ``parameters`` field echoes the outer request body, so we read
        ``seed`` from there.
        """
        return body.get("parameters", {}).get("seed", fallback)

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def txt2image(
        self,
        params: Txt2ImageParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Generate an image from a text prompt via ``POST /sdapi/v1/txt2img``.

        Args:
            params:     Generation parameters.
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        payload = {
            "prompt":          params.prompt,
            "negative_prompt": params.negative_prompt,
            "width":           params.width,
            "height":          params.height,
            "steps":           params.steps,
            "cfg_scale":       params.cfg_scale,
            "seed":            params.seed,
            "sampler_name":    params.sampler_name,
            "scheduler":       params.scheduler,
            "batch_size":      params.batch_size,
        }

        http = self._get_http()
        resp = await http.post("/sdapi/v1/txt2img", json=payload)
        resp.raise_for_status()
        body = resp.json()

        b64 = body["images"][0]
        image_path, w, h = await asyncio.to_thread(
            self._save_b64_image, b64, output_dir
        )
        seed = self._parse_seed(body, params.seed)

        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            seed=seed,
            prompt_id=Path(image_path).stem,
        )

    async def image_edit(
        self,
        params: ImageEditParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Edit a reference image via ``POST /sdapi/v1/img2img``.

        All images in *params.image_paths* are tiled into a horizontal grid
        before being sent to the server.

        Args:
            params:     Edit parameters (prompt + one or more image paths).
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        init_image_b64 = await asyncio.to_thread(
            self._images_to_b64_grid, params.image_paths
        )

        payload = {
            "prompt":              params.prompt,
            "negative_prompt":     params.negative_prompt,
            "init_images":         [init_image_b64],
            "denoising_strength":  params.denoising_strength,
            "width":               params.width,
            "height":              params.height,
            "steps":               params.steps,
            "cfg_scale":           params.cfg_scale,
            "seed":                params.seed,
            "sampler_name":        params.sampler_name,
            "scheduler":           params.scheduler,
        }

        http = self._get_http()
        resp = await http.post("/sdapi/v1/img2img", json=payload)
        resp.raise_for_status()
        body = resp.json()

        b64 = body["images"][0]
        image_path, w, h = await asyncio.to_thread(
            self._save_b64_image, b64, output_dir
        )
        seed = self._parse_seed(body, params.seed)

        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            seed=seed,
            prompt_id=Path(image_path).stem,
        )


# =============================================================================
# Convenience async functions
# =============================================================================

async def txt2image(
    prompt: str,
    output_dir: str = "./output",
    server_url: str = "http://gx10-cbc5:1234",
    **kwargs,
) -> ImageGenerationResult:
    """Generate an image from a text prompt with a single async call."""
    config = StableDiffusionCppConfig(server_url=server_url)
    params = Txt2ImageParams(prompt=prompt, **kwargs)
    async with StableDiffusionCppClient(config) as client:
        return await client.txt2image(params, output_dir=output_dir)


async def image_edit(
    prompt: str,
    image_paths: List[str],
    output_dir: str = "./output",
    server_url: str = "http://gx10-cbc5:1234",
    **kwargs,
) -> ImageGenerationResult:
    """Edit one or several reference images with a single async call."""
    config = StableDiffusionCppConfig(server_url=server_url)
    params = ImageEditParams(prompt=prompt, image_paths=image_paths, **kwargs)
    async with StableDiffusionCppClient(config) as client:
        return await client.image_edit(params, output_dir=output_dir)