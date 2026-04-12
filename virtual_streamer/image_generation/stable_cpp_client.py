"""
Stable Diffusion CPP Image Generation Client

Client for a stable-diffusion.cpp server using the OpenAI-compatible API.

Abstract interface
------------------
:class:`ImageGenerationClientInterface` defines two operations:

  - :meth:`txt2image` — generate an image from a text prompt only
  - :meth:`image_edit` — edit / transform reference images guided by a text
                          prompt

Both methods accept typed Pydantic parameter models and return an
:class:`ImageGenerationResult`.

Concrete implementation
-----------------------
:class:`StableDiffusionCppClient` implements the interface using the
OpenAI-compatible endpoints:

  POST /v1/images/generations  — text-to-image (JSON body)
  POST /v1/images/edits        — image editing  (multipart/form-data)

Advanced stable-diffusion.cpp parameters (steps, cfg_scale, seed, sampler,
scheduler, negative_prompt, …) are injected into the prompt string using the
``<sd_cpp_extra_args>`` tag understood by sd.cpp.

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

    # Image editing
    async with StableDiffusionCppClient(StableDiffusionCppConfig()) as client:
        result = await client.image_edit(
            ImageEditParams(
                prompt="Make the sky purple and add stars",
                image_paths=["photo.jpg"],
                extra_args={"denoising_strength": 0.75},
            ),
            output_dir="./output",
        )
        print(result.image_path)
"""

import asyncio
import base64
import io
import json
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
    """Parameters for text-to-image generation (POST /v1/images/generations)."""

    prompt: str = Field(description="Text prompt describing the desired image")
    negative_prompt: str = Field(default="")
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=20, ge=1, le=200)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    seed: int = Field(default=-1, description="-1 for a random seed")
    sampler_name: str = Field(default="euler_a")
    scheduler: str = Field(default="default")
    n: int = Field(default=1, ge=1, le=8, description="Number of images to generate")
    output_format: str = Field(default="png")
    output_compression: int = Field(default=100, ge=0, le=100)
    extra_args: Optional[dict] = Field(
        default=None,
        description="Additional sd.cpp-specific params injected via sd_cpp_extra_args tag",
    )


class ImageEditParams(BaseModel):
    """Parameters for image editing (POST /v1/images/edits)."""

    prompt: str = Field(description="Editing instruction")
    image_paths: List[str] = Field(
        description="Local paths to one or more reference images",
        min_length=1,
    )
    negative_prompt: str = Field(default="")
    steps: int = Field(default=20, ge=1, le=200)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    seed: int = Field(default=-1, description="-1 for a random seed")
    sampler_name: str = Field(default="euler_a")
    scheduler: str = Field(default="default")
    n: int = Field(default=1, ge=1, le=8, description="Number of images to generate")
    output_format: str = Field(default="png")
    output_compression: int = Field(default=100, ge=0, le=100)
    extra_args: Optional[dict] = Field(
        default=None,
        description="Additional sd.cpp-specific params (e.g. denoising_strength)",
    )


# =============================================================================
# Result model
# =============================================================================

class ImageGenerationResult(BaseModel):
    """Result returned after a successful generation or edit."""

    image_path: str = Field(description="Local path to the saved output image")
    width: int
    height: int
    seed: int = Field(default=-1)
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
    ) -> ImageGenerationResult: ...

    @abstractmethod
    async def image_edit(
        self,
        params: ImageEditParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> "ImageGenerationClientInterface":
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()


# =============================================================================
# Stable Diffusion CPP OpenAI-compatible implementation
# =============================================================================

class StableDiffusionCppClient(ImageGenerationClientInterface):
    """
    Image generation / editing client using the OpenAI-compatible API of a
    stable-diffusion.cpp server.

    Advanced sd.cpp parameters are injected into the prompt using the
    ``<sd_cpp_extra_args>{...}</sd_cpp_extra_args>`` tag.
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
    def _build_prompt(prompt: str, extra: dict) -> str:
        """Append sd_cpp_extra_args tag to *prompt* with *extra* as JSON."""
        return f"{prompt}\n<sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>"

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
        Generate an image from a text prompt via ``POST /v1/images/generations``.

        Args:
            params:     Generation parameters.
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        extra = {
            "negative_prompt": params.negative_prompt,
            "steps": params.steps,
            "cfg_scale": params.cfg_scale,
            "seed": params.seed,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            **(params.extra_args or {}),
        }
        full_prompt = self._build_prompt(params.prompt, extra)

        payload = {
            "prompt": full_prompt,
            "n": params.n,
            "size": f"{params.width}x{params.height}",
            "output_format": params.output_format,
            "output_compression": params.output_compression,
        }

        http = self._get_http()
        resp = await http.post("/v1/images/generations", json=payload)
        resp.raise_for_status()
        body = resp.json()

        b64 = body["data"][0]["b64_json"]
        image_path, w, h = await asyncio.to_thread(
            self._save_b64_image, b64, output_dir
        )
        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            seed=params.seed,
            prompt_id=Path(image_path).stem,
        )

    async def image_edit(
        self,
        params: ImageEditParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Edit reference images via ``POST /v1/images/edits`` (multipart).

        Each path in *params.image_paths* is uploaded as an ``image[]`` field.

        Args:
            params:     Edit parameters (prompt + one or more image paths).
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        extra = {
            "negative_prompt": params.negative_prompt,
            "steps": params.steps,
            "cfg_scale": params.cfg_scale,
            "seed": params.seed,
            "sampler_name": params.sampler_name,
            "scheduler": params.scheduler,
            **(params.extra_args or {}),
        }
        full_prompt = self._build_prompt(params.prompt, extra)

        # Build multipart form — read image files in a thread
        def _build_multipart():
            files = []
            for path in params.image_paths:
                data = Path(path).read_bytes()
                files.append(("image[]", (Path(path).name, data, "image/png")))
            return files

        image_files = await asyncio.to_thread(_build_multipart)

        form_data = {
            "prompt": full_prompt,
            "n": str(params.n),
            "size": f"{512}x{512}",  # sd.cpp uses input image size for edits
            "output_format": params.output_format,
            "output_compression": str(params.output_compression),
        }

        http = self._get_http()
        resp = await http.post("/v1/images/edits", data=form_data, files=image_files)
        resp.raise_for_status()
        body = resp.json()

        b64 = body["data"][0]["b64_json"]
        image_path, w, h = await asyncio.to_thread(
            self._save_b64_image, b64, output_dir
        )
        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            seed=params.seed,
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