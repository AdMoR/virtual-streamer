"""
Flux Kontext Image Generation Client

Client for the Flux Kontext image-edit inference server.

Abstract interface
------------------
:class:`ImageGenerationClientInterface` defines two operations:

  - :meth:`txt2image` — generate an image from a text prompt only
  - :meth:`image_edit` — edit / transform one or several reference images
                          guided by a text prompt

Both methods accept typed Pydantic parameter models and return an
:class:`ImageGenerationResult`.

Concrete implementation
-----------------------
:class:`FluxKontextClient` implements the interface by calling the REST API
exposed by ``virtual_streamer/image_generation/flux_kontext_server/app.py``.

Usage
-----
    from virtual_streamer.image_generation.flux_kontext_client import (
        FluxKontextClient, FluxKontextConfig,
        Txt2ImageParams, ImageEditParams,
    )

    # Text-to-image
    async with FluxKontextClient(FluxKontextConfig()) as client:
        result = await client.txt2image(
            Txt2ImageParams(prompt="A fox in a snowy forest"),
            output_dir="./output",
        )
        print(result.image_path)

    # Image editing (one or several reference images)
    async with FluxKontextClient(FluxKontextConfig()) as client:
        result = await client.image_edit(
            ImageEditParams(
                prompt="Make the sky purple and add stars",
                image_paths=["photo.jpg"],
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

class FluxKontextConfig(BaseModel):
    """Connection settings for the Flux Kontext inference server."""

    server_url: str = Field(
        default="http://localhost:8010",
        description="Base URL of the running Flux Kontext server",
    )
    timeout: float = Field(
        default=300.0,
        description="HTTP timeout in seconds",
    )


# =============================================================================
# Parameter models
# =============================================================================

class Txt2ImageParams(BaseModel):
    """Parameters for text-to-image generation (no conditioning image)."""

    prompt: str = Field(description="Text prompt describing the desired image")
    width: int = Field(default=680, ge=64, le=2048)
    height: int = Field(default=496, ge=64, le=2048)
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)
    seed: int = Field(default=-1, description="-1 for a random seed")


class ImageEditParams(BaseModel):
    """Parameters for image editing conditioned on one or several reference images."""

    prompt: str = Field(description="Editing instruction")
    image_paths: List[str] = Field(
        description="Local paths to one or more reference images",
        min_length=1,
    )
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=2.5, ge=0.0, le=20.0)
    seed: int = Field(default=-1, description="-1 for a random seed")


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
        Edit or transform one or several reference images guided by a text prompt.

        When multiple images are supplied they are tiled into a grid before
        being sent to the server.

        Args:
            params:     Edit parameters (prompt, image_paths, steps…)
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
# Flux Kontext implementation
# =============================================================================

class FluxKontextClient(ImageGenerationClientInterface):
    """
    Image generation / editing client backed by the Flux Kontext REST server.

    The HTTP client is created lazily on first use and reused across calls.
    """

    def __init__(self, config: Optional[FluxKontextConfig] = None) -> None:
        self.config = config or FluxKontextConfig()
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
    def _image_to_b64(path: str) -> str:
        with Image.open(path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
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
        Generate an image from a text prompt via the ``/txt2image`` endpoint.

        Args:
            params:     Generation parameters.
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        http = self._get_http()

        payload = {
            "prompt": params.prompt,
            "width": params.width,
            "height": params.height,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "seed": params.seed,
        }

        resp = await http.post("/txt2image", json=payload)
        resp.raise_for_status()
        body = resp.json()

        image_path, w, h = self._save_b64_image(body["image_b64"], output_dir)
        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            seed=body["seed"],
            prompt_id=Path(image_path).stem,
        )

    async def image_edit(
        self,
        params: ImageEditParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Edit reference images via the ``/edit`` endpoint.

        All images in *params.image_paths* are base64-encoded and sent to the
        server, which tiles them into a grid when more than one is provided.

        Args:
            params:     Edit parameters (prompt + one or more image paths).
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        http = self._get_http()

        images_b64 = await asyncio.to_thread(
            lambda: [self._image_to_b64(p) for p in params.image_paths]
        )

        payload = {
            "prompt": params.prompt,
            "images_b64": images_b64,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "seed": params.seed,
        }

        resp = await http.post("/edit", json=payload)
        resp.raise_for_status()
        body = resp.json()

        image_path, w, h = self._save_b64_image(body["image_b64"], output_dir)
        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            seed=body["seed"],
            prompt_id=Path(image_path).stem,
        )


# =============================================================================
# Convenience async functions
# =============================================================================

async def txt2image(
    prompt: str,
    output_dir: str = "./output",
    server_url: str = "http://localhost:8010",
    **kwargs,
) -> ImageGenerationResult:
    """Generate an image from a text prompt with a single async call."""
    config = FluxKontextConfig(server_url=server_url)
    params = Txt2ImageParams(prompt=prompt, **kwargs)
    async with FluxKontextClient(config) as client:
        return await client.txt2image(params, output_dir=output_dir)


async def image_edit(
    prompt: str,
    image_paths: List[str],
    output_dir: str = "./output",
    server_url: str = "http://localhost:8010",
    **kwargs,
) -> ImageGenerationResult:
    """Edit one or several reference images with a single async call."""
    config = FluxKontextConfig(server_url=server_url)
    params = ImageEditParams(prompt=prompt, image_paths=image_paths, **kwargs)
    async with FluxKontextClient(config) as client:
        return await client.image_edit(params, output_dir=output_dir)
