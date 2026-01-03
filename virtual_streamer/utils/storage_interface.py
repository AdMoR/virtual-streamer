"""
Abstract Storage Interface for Virtual Streamer.

This module defines a generic object storage interface that can be implemented
by different storage backends (MinIO, S3, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class StorageInterface(ABC):
    """
    Abstract interface for object storage operations.
    
    All storage backends (MinIO, S3, etc.) should implement this interface
    to ensure consistent behavior across the application.
    """

    @abstractmethod
    async def put_object(
        self, key: str, data: bytes, content_type: Optional[str] = None
    ) -> str:
        """
        Store binary data with the given key.
        
        Args:
            key: Object key/path in storage
            data: Binary data to store
            content_type: Optional MIME type (e.g., "application/json", "video/mp4")
            
        Returns:
            The key where the object was stored
        """
        pass

    @abstractmethod
    async def get_object(self, key: str) -> Optional[bytes]:
        """
        Retrieve binary data by key.
        
        Args:
            key: Object key/path in storage
            
        Returns:
            Binary data if found, None otherwise
        """
        pass

    @abstractmethod
    async def put_json(self, key: str, data: Dict[str, Any]) -> str:
        """
        Store a dictionary as JSON.
        
        Args:
            key: Object key/path in storage
            data: Dictionary to store as JSON
            
        Returns:
            The key where the object was stored
        """
        pass

    @abstractmethod
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and parse JSON data by key.
        
        Args:
            key: Object key/path in storage
            
        Returns:
            Parsed dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> None:
        """
        Delete an object by key.
        
        Args:
            key: Object key/path to delete
        """
        pass

    @abstractmethod
    async def list_objects(self, prefix: str) -> List[str]:
        """
        List all object keys matching a prefix.
        
        Args:
            prefix: Key prefix to filter by
            
        Returns:
            List of matching object keys
        """
        pass

    @abstractmethod
    async def upload_file(self, local_path: str, key: str) -> str:
        """
        Upload a local file to storage.
        
        Args:
            local_path: Path to local file
            key: Destination key in storage
            
        Returns:
            The key where the file was stored
        """
        pass

    @abstractmethod
    async def download_file(self, key: str, local_path: str) -> str:
        """
        Download a file from storage to local filesystem.
        
        Args:
            key: Object key in storage
            local_path: Destination path on local filesystem
            
        Returns:
            The local path where file was saved
        """
        pass

    @abstractmethod
    def get_url(self, key: str) -> str:
        """
        Get a URL for accessing an object.
        
        Args:
            key: Object key in storage
            
        Returns:
            URL string for accessing the object
        """
        pass

    @abstractmethod
    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            key: Object key to check
            
        Returns:
            True if object exists, False otherwise
        """
        pass

