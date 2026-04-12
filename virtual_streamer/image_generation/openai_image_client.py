"""
OpenAI GPT Image Generation Client

Client for OpenAI image generation using the Responses API with the
``image_generation`` tool.

Abstract interface
------------------
Implements :class:`ImageGenerationClientInterface` with two operations:

  - :meth:`txt2image` — generate an image from a text prompt
  - :meth:`image_edit` — generate an image conditioned on reference images
                          and a text prompt

Both methods return an :class:`ImageGenerationResult`.

Concrete implementation
-----------------------
:class:`OpenAIImageClient` calls ``POST https://api.openai.com/v1/responses``
with ``tools=[{"type": "image_generation"}]``.  Reference images are passed as
``input_image`` content blocks with base64 data URLs.

Usage
-----
    from virtual_streamer.image_generation.openai_image_client import (
        OpenAIImageClient, OpenAIImageConfig,
        OpenAITxt2ImageParams, OpenAIImageEditParams,
    )

    # Text-to-image
    async with OpenAIImageClient(OpenAIImageConfig(api_key="sk-...")) as client:
        result = await client.txt2image(
            OpenAITxt2ImageParams(prompt="A fox in a snowy forest"),
            output_dir="./output",
        )
        print(result.image_path)

    # Image editing conditioned on reference images
    async with OpenAIImageClient(OpenAIImageConfig(api_key="sk-...")) as client:
        result = await client.image_edit(
            OpenAIImageEditParams(
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
import mimetypes
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

class OpenAIImageConfig(BaseModel):
    """Connection / authentication settings for the OpenAI API."""

    api_key: str = Field(description="OpenAI API key (sk-…)")
    model: str = Field(
        default="gpt-image-1",
        description="Model to use for image generation via the Responses API",
    )
    timeout: float = Field(default=300.0, description="HTTP timeout in seconds")


# =============================================================================
# Parameter models
# =============================================================================

class OpenAITxt2ImageParams(BaseModel):
    """Parameters for text-to-image generation."""

    prompt: str = Field(description="Text prompt describing the desired image")
    model: Optional[str] = Field(
        default=None,
        description="Override the model from config (e.g. 'gpt-4.1')",
    )


class OpenAIImageEditParams(BaseModel):
    """Parameters for image generation conditioned on reference images."""

    prompt: str = Field(description="Editing instruction or generation prompt")
    image_paths: List[str] = Field(
        description="Local paths to one or more reference images",
        min_length=1,
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the model from config",
    )


# =============================================================================
# Result model (shared with other clients)
# =============================================================================

class ImageGenerationResult(BaseModel):
    """Result returned after a successful generation or edit."""

    image_path: str = Field(description="Local path to the saved output image")
    width: int
    height: int
    seed: int = Field(default=-1, description="OpenAI does not expose a seed")
    prompt_id: str = Field(description="Unique identifier for this generation")


# =============================================================================
# Abstract interface (mirrored from stable_cpp_client for independence)
# =============================================================================

class ImageGenerationClientInterface(ABC):

    @abstractmethod
    async def txt2image(
        self,
        params: OpenAITxt2ImageParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult: ...

    @abstractmethod
    async def image_edit(
        self,
        params: OpenAIImageEditParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> "ImageGenerationClientInterface":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


# =============================================================================
# OpenAI implementation
# =============================================================================

class OpenAIImageClient(ImageGenerationClientInterface):
    """
    Image generation client backed by the OpenAI Responses API.

    Both :meth:`txt2image` and :meth:`image_edit` call
    ``POST /v1/responses`` with ``tools=[{"type": "image_generation"}]``.

    The AsyncOpenAI client is created at construction time and reused.
    """

    def __init__(self, config: OpenAIImageConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            timeout=config.timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, override: Optional[str]) -> str:
        return override or self.config.model

    @staticmethod
    def _image_to_data_url(path: str) -> str:
        """Encode a local image file as a base64 data URL."""
        mime, _ = mimetypes.guess_type(path)
        if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
            # Normalise to PNG for unsupported types
            with Image.open(path) as img:
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                raw = buf.getvalue()
            mime = "image/png"
        else:
            raw = Path(path).read_bytes()
        b64 = base64.b64encode(raw).decode()
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _save_b64_image(b64: str, output_dir: str) -> tuple[str, int, int]:
        """Decode a base64 PNG/JPEG, save to *output_dir*, return (path, w, h)."""
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = f"{uuid.uuid4().hex}.png"
        fpath = out / fname
        img.save(fpath)
        return str(fpath), img.width, img.height

    @staticmethod
    def _extract_image_b64(response) -> str:
        """Pull the first image_generation_call result from a Responses API response."""
        for item in response.output:
            if item.type == "image_generation_call":
                return item.result
        raise RuntimeError(
            "OpenAI response contained no image_generation_call output. "
            f"Output types: {[o.type for o in response.output]}"
        )

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._client.close()

    async def txt2image(
        self,
        params: OpenAITxt2ImageParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Generate an image from a text prompt via the OpenAI Responses API.

        Args:
            params:     Generation parameters (prompt, optional model override).
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            openai.APIError: On API-level errors.
            RuntimeError:    If the response contains no image output.
        """
        response = await self._client.responses.create(
            model=self._resolve_model(params.model),
            input=params.prompt,
            tools=[{"type": "image_generation"}],
        )

        b64 = self._extract_image_b64(response)
        image_path, w, h = await asyncio.to_thread(
            self._save_b64_image, b64, output_dir
        )
        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            prompt_id=Path(image_path).stem,
        )

    async def image_edit(
        self,
        params: OpenAIImageEditParams,
        output_dir: str = "./output",
    ) -> ImageGenerationResult:
        """
        Generate an image conditioned on reference images and a text prompt.

        Each path in *params.image_paths* is encoded as a base64 data URL and
        passed as an ``input_image`` content block alongside the prompt.

        Args:
            params:     Edit parameters (prompt + one or more image paths).
            output_dir: Directory to save the output PNG.

        Returns:
            :class:`ImageGenerationResult`.

        Raises:
            openai.APIError: On API-level errors.
            RuntimeError:    If the response contains no image output.
        """
        data_urls = await asyncio.to_thread(
            lambda: [self._image_to_data_url(p) for p in params.image_paths]
        )

        content = [{"type": "input_text", "text": params.prompt}]
        for url in data_urls:
            content.append({"type": "input_image", "image_url": url})

        response = await self._client.responses.create(
            model=self._resolve_model(params.model),
            input=[{"role": "user", "content": content}],
            tools=[{"type": "image_generation"}],
        )

        b64 = self._extract_image_b64(response)
        image_path, w, h = await asyncio.to_thread(
            self._save_b64_image, b64, output_dir
        )
        return ImageGenerationResult(
            image_path=image_path,
            width=w,
            height=h,
            prompt_id=Path(image_path).stem,
        )


# =============================================================================
# Convenience async functions
# =============================================================================

async def txt2image(
    prompt: str,
    api_key: str,
    output_dir: str = "./output",
    model: str = "gpt-image-1",
) -> ImageGenerationResult:
    """Generate an image from a text prompt with a single async call."""
    config = OpenAIImageConfig(api_key=api_key, model=model)
    params = OpenAITxt2ImageParams(prompt=prompt)
    async with OpenAIImageClient(config) as client:
        return await client.txt2image(params, output_dir=output_dir)


async def image_edit(
    prompt: str,
    image_paths: List[str],
    api_key: str,
    output_dir: str = "./output",
    model: str = "gpt-image-1",
) -> ImageGenerationResult:
    """Generate an image conditioned on reference images with a single async call."""
    config = OpenAIImageConfig(api_key=api_key, model=model)
    params = OpenAIImageEditParams(prompt=prompt, image_paths=image_paths)
    async with OpenAIImageClient(config) as client:
        return await client.image_edit(params, output_dir=output_dir)