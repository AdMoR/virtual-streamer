"""
API Dependencies and Utilities

Common dependencies and utilities used across API layers.
"""

import os
from pathlib import Path
from typing import Optional


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



