"""
Character API Client

HTTP client for calling the low-level Character API endpoints.
Use this client for code running outside the main API server.
"""

import os
from typing import List, Optional

import httpx

from virtual_streamer.video_server.models import Character, VoiceSample


class CharacterClient:
    """
    Async HTTP client for Character API operations.
    
    Use this client for external code that needs to interact with the
    Character API. For code running inside the API server, use
    EntityRepository directly instead.
    
    Example:
        async with CharacterClient() as client:
            character = await client.get_character("fred")
            characters = await client.list_characters()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Character API client.

        Args:
            base_url: Base URL for the API (defaults to API_BASE_URL env var or http://localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = (
            base_url
            or os.environ.get("API_BASE_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "CharacterClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _dict_to_character(self, data: dict) -> Character:
        """Convert API response dict to Character model."""
        voice_samples = [
            VoiceSample(
                sample_storage_path=s.get("sample_storage_path", ""),
                transcript=s.get("transcript", ""),
            )
            for s in data.get("voice_samples", [])
        ]
        
        return Character(
            character_id=data["character_id"],
            name=data["name"],
            description=data.get("description"),
            video_clip_path=data.get("video_clip_path", ""),
            voice_samples=voice_samples,
            video_search_tag=data.get("video_search_tag"),
            identity_images=data.get("identity_images", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    async def get_character(self, character_id: str) -> Character:
        """
        Fetch a character by ID.

        Args:
            character_id: Character identifier

        Returns:
            Character object

        Raises:
            httpx.HTTPStatusError: If character not found (404) or other HTTP error
        """
        client = self._get_client()
        url = f"{self.base_url}/api/v1/characters/{character_id}"
        
        response = await client.get(url)
        response.raise_for_status()
        
        data = response.json()
        return self._dict_to_character(data)

    async def list_characters(self, limit: int = 100) -> List[Character]:
        """
        List all characters.

        Args:
            limit: Maximum number of characters to return

        Returns:
            List of Character objects

        Raises:
            httpx.HTTPStatusError: On HTTP error
        """
        client = self._get_client()
        url = f"{self.base_url}/api/v1/characters"
        params = {"limit": limit}
        
        response = await client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return [self._dict_to_character(item) for item in data]


# Convenience functions for one-off calls

async def get_character(character_id: str, base_url: Optional[str] = None) -> Character:
    """
    Fetch a character by ID (convenience function).
    
    For multiple calls, use CharacterClient as a context manager instead.

    Args:
        character_id: Character identifier
        base_url: Optional base URL override

    Returns:
        Character object
    """
    async with CharacterClient(base_url=base_url) as client:
        return await client.get_character(character_id)


async def list_characters(limit: int = 100, base_url: Optional[str] = None) -> List[Character]:
    """
    List all characters (convenience function).
    
    For multiple calls, use CharacterClient as a context manager instead.

    Args:
        limit: Maximum number of characters to return
        base_url: Optional base URL override

    Returns:
        List of Character objects
    """
    async with CharacterClient(base_url=base_url) as client:
        return await client.list_characters(limit)

