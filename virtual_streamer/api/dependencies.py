"""
API Dependencies and Utilities

Common dependencies and utilities used across API layers.
"""

import os
from pathlib import Path
from typing import Optional

from virtual_streamer.utils.minio_client import get_storage_client, MinIOClient


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


class StoragePathResolver:
    """
    Resolves MinIO storage keys to local file paths by downloading files.
    
    This enables services running in containers to access files stored in MinIO
    by downloading them to a local cache directory.
    """

    def __init__(
        self,
        storage: Optional[MinIOClient] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize storage path resolver.

        Args:
            storage: MinIO client instance (defaults to global client)
            cache_dir: Local cache directory (defaults to STORAGE_CACHE_DIR env var or /tmp/storage_cache)
        """
        self.storage = storage or get_storage_client()
        self.cache_dir = Path(
            cache_dir or os.environ.get("STORAGE_CACHE_DIR", "/tmp/storage_cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def resolve_file(self, storage_key: str) -> str:
        """
        Download file from MinIO and return local path.
        
        Works for any file type (video, audio, etc.). Files are cached locally
        to avoid repeated downloads.

        Args:
            storage_key: MinIO key like "clips/fred.mp4" or "audios/sample.wav"

        Returns:
            Local path like "/tmp/storage_cache/clips/fred.mp4"
            
        Raises:
            FileNotFoundError: If the file doesn't exist in storage
        """
        if not storage_key:
            raise ValueError("storage_key cannot be empty")

        local_path = self.cache_dir / storage_key

        # Check if already cached
        if local_path.exists():
            return str(local_path)

        # Check if file exists in storage
        if not await self.storage.object_exists(storage_key):
            raise FileNotFoundError(f"File not found in storage: {storage_key}")

        # Download from MinIO
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await self.storage.download_file(storage_key, str(local_path))

        return str(local_path)

    def clear_cache(self, storage_key: Optional[str] = None) -> None:
        """
        Clear cached files.

        Args:
            storage_key: Specific key to clear, or None to clear all
        """
        if storage_key:
            local_path = self.cache_dir / storage_key
            if local_path.exists():
                local_path.unlink()
        else:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global storage resolver instance
_storage_resolver: Optional[StoragePathResolver] = None


def get_storage_resolver() -> StoragePathResolver:
    """
    Get or create the global storage path resolver instance.

    Returns:
        StoragePathResolver instance
    """
    global _storage_resolver

    if _storage_resolver is None:
        _storage_resolver = StoragePathResolver()

    return _storage_resolver
