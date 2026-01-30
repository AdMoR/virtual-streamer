"""
Webservice Client

Async HTTP client for calling Virtual Streamer API endpoints.
Consolidates TTS, Wav2Lip, and STT API calls in one reusable client.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx


@dataclass
class APIConfig:
    """Configuration for webservice API calls."""

    base_url: str = field(
        default_factory=lambda: os.environ.get("API_BASE_URL", "http://localhost:8000")
    )
    timeout: float = 120.0
    character_id: Optional[str] = None  # Optional default character


class WebserviceClient:
    """
    Async HTTP client for calling the Virtual Streamer webservice API.

    Handles TTS, Wav2Lip, and STT API calls with proper error handling.
    
    Example:
        async with WebserviceClient(config) as client:
            audio_path = await client.generate_tts("Hello!", "fred")
            video_path = await client.generate_wav2lip(audio_path, source_video, "fred")
            srt_path = await client.transcribe_to_srt(audio_path)
    """

    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize the WebserviceClient.
        
        Args:
            config: API configuration (defaults to environment-based config)
        """
        self.config = config or APIConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "WebserviceClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def generate_tts(
        self,
        text: str,
        character_id: Optional[str] = None,
        entry_id: str = "",
    ) -> str:
        """
        Call TTS API to generate audio from text.

        Args:
            text: Dialog text to synthesize
            character_id: Character ID to use for voice (falls back to config default)
            entry_id: Optional entry ID for tracking

        Returns:
            Path to generated audio file
            
        Raises:
            httpx.HTTPStatusError: If API call fails
        """
        cid = character_id or self.config.character_id
        if not cid:
            raise ValueError("character_id must be provided or set in config")
            
        client = self._get_client()
        response = await client.post(
            "/api/v1/tts/generate",
            json={
                "entry_id": entry_id or f"tts_{datetime.now().timestamp()}",
                "character_id": cid,
                "text": text,
                "timestamp": 0,
            },
            timeout=30 * 60,  # TTS can be slow
        )
        response.raise_for_status()
        data = response.json()
        return data["audio_path"]

    async def generate_wav2lip(
        self,
        audio_path: str,
        video_path: str,
        character_id: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Call Wav2Lip API to generate lip-synced video.

        Args:
            audio_path: Path to audio file
            video_path: Path to source video
            character_id: Character ID for face detection (falls back to config default)
            output_dir: Optional output directory

        Returns:
            Path to generated lip-synced video
            
        Raises:
            httpx.HTTPStatusError: If API call fails
        """
        cid = character_id or self.config.character_id
        if not cid:
            raise ValueError("character_id must be provided or set in config")
            
        client = self._get_client()
        response = await client.post(
            "/api/v1/wav2lip/generate",
            json={
                "audio_path": audio_path,
                "video": {
                    "storage_path": video_path,
                    "collection_ids": [],
                },
                "options": {
                    "subtitles_enabled": False,
                    "subtitle_style": None,
                },
                "character_id": cid,
                "output_dir": output_dir,
            },
            timeout=30 * 60,  # Wav2Lip can be slow
        )
        response.raise_for_status()
        data = response.json()
        return data["raw_video_path"]

    async def transcribe_to_srt(self, audio_path: str) -> str:
        """
        Call STT API to generate SRT subtitles from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to generated SRT file
            
        Raises:
            httpx.HTTPStatusError: If API call fails
        """
        client = self._get_client()
        with open(audio_path, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_path), f, "audio/wav")}
            response = await client.post(
                "/api/v1/stt/transcribe-to-srt",
                files=files,
                timeout=30 * 60,  # STT can be slow
            )
        response.raise_for_status()
        data = response.json()
        return data["srt_path"]

    async def transcribe(self, audio_path: str) -> str:
        """
        Call STT API to transcribe audio to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
            
        Raises:
            httpx.HTTPStatusError: If API call fails
        """
        client = self._get_client()
        with open(audio_path, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_path), f, "audio/wav")}
            response = await client.post(
                "/api/v1/stt/transcribe",
                files=files,
                timeout=30 * 60,
            )
        response.raise_for_status()
        data = response.json()
        return data["text"]
