"""
API Dependencies and Utilities

Common dependencies and utilities used across API layers.
"""

import os
from pathlib import Path
from typing import Optional, List
from fastapi import HTTPException

from virtual_streamer.video_server.models import Character
from virtual_streamer.utils.minio_client import get_storage_client, MinIOClient
from virtual_streamer.utils.storage_interface import StorageInterface


class PathResolver:
    """
    Resolves paths between different services.

    The entity service stores paths relative to DATA_DIR (e.g., "audios/file.wav").
    Other services need to resolve these to absolute paths.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize path resolver.

        Args:
            data_dir: Base data directory. Defaults to DATA_DIR env var or /data
        """
        self.data_dir = Path(data_dir or os.environ.get("DATA_DIR", "/data"))

    def resolve(self, path: str) -> str:
        """
        Resolve a path that may be relative to DATA_DIR.

        Args:
            path: Path to resolve (relative or absolute)

        Returns:
            Absolute path
        """
        if not path:
            return path

        path_obj = Path(path)

        # If already absolute, return as-is
        if path_obj.is_absolute():
            return str(path_obj)

        # Otherwise, resolve relative to data_dir
        resolved = self.data_dir / path
        return str(resolved)

    def resolve_audio(self, relative_path: str) -> str:
        """
        Resolve an audio file path from entity service.

        Args:
            relative_path: Path like "audios/file.wav"

        Returns:
            Absolute path like "/data/audios/file.wav"
        """
        return self.resolve(relative_path)

    def resolve_video(self, relative_path: str) -> str:
        """
        Resolve a video file path from entity service.

        Args:
            relative_path: Path like "clips/video.mp4"

        Returns:
            Absolute path like "/data/clips/video.mp4"
        """
        return self.resolve(relative_path)

    def exists(self, path: str) -> bool:
        """
        Check if a resolved path exists.

        Args:
            path: Path to check (will be resolved first)

        Returns:
            True if file exists
        """
        resolved = self.resolve(path)
        return Path(resolved).exists()


# Global path resolver instance
_path_resolver: Optional[PathResolver] = None


def get_path_resolver() -> PathResolver:
    """
    Get or create the global path resolver instance.

    Returns:
        PathResolver instance
    """
    global _path_resolver

    if _path_resolver is None:
        _path_resolver = PathResolver()

    return _path_resolver


def resolve_path(path: str) -> str:
    """
    Convenience function to resolve a path.

    Args:
        path: Path to resolve

    Returns:
        Absolute path
    """
    resolver = get_path_resolver()
    return resolver.resolve(path)


# Character storage functions

# Storage configuration
_CHARACTER_PREFIX = "characters/"


async def get_character_data(character_id: str) -> Character:
    """
    Fetch character data from MinIO storage.

    This replaces the old HTTP call to entity_api service.

    Args:
        character_id: Character identifier

    Returns:
        Character object

    Raises:
        HTTPException: If character not found
    """
    storage = get_storage_client()
    key = f"{_CHARACTER_PREFIX}{character_id}.json"

    try:
        data = await storage.get_json(key)
        if data is None:
            raise HTTPException(
                status_code=404, detail=f"Character '{character_id}' not found"
            )

        # Ensure backward compatibility
        data["video_clip_path"] = data.get("video_clip_path", "")
        return Character(**data)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Character '{character_id}' not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading character '{character_id}': {str(e)}",
        )


async def list_characters(limit: int = 100) -> List[Character]:
    """
    List all characters from MinIO storage.

    Args:
        limit: Maximum number of characters to return

    Returns:
        List of Character objects
    """
    storage = get_storage_client()
    keys = await storage.list_objects(_CHARACTER_PREFIX)

    characters = []
    count = 0
    for key in keys:
        if key.endswith(".json"):
            try:
                data = await storage.get_json(key)
                if data:
                    # Ensure backward compatibility
                    data["video_clip_path"] = data.get("video_clip_path", "")
                    characters.append(Character(**data))
                    count += 1
                    if count >= limit:
                        break
            except Exception as e:
                print(f"Warning: Error loading character from {key}: {e}")
                continue

    return characters
