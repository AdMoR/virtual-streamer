"""
MinIO Storage Client for Virtual Streamer.

This module implements the StorageInterface using MinIO (S3-compatible) storage.
It uses boto3 with a custom endpoint URL to connect to MinIO.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import partial

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from virtual_streamer.utils.storage_interface import StorageInterface


class MinIOClient(StorageInterface):
    """
    Storage client using MinIO (S3-compatible object storage).
    
    Configuration via environment variables:
    - MINIO_ENDPOINT: MinIO server URL (default: http://minio:9000)
    - MINIO_ACCESS_KEY: Access key (default: minioadmin)
    - MINIO_SECRET_KEY: Secret key (default: minioadmin)
    - MINIO_BUCKET: Bucket name (default: virtual-streamer)
    - MINIO_SECURE: Use HTTPS (default: false)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        secure: Optional[bool] = None,
    ):
        """
        Initialize MinIO client.
        
        Args:
            endpoint: MinIO server URL (overrides env var)
            access_key: Access key (overrides env var)
            secret_key: Secret key (overrides env var)
            bucket: Bucket name (overrides env var)
            secure: Use HTTPS (overrides env var)
        """
        self.endpoint = endpoint or os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
        self.access_key = access_key or os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = bucket or os.environ.get("MINIO_BUCKET", "virtual-streamer")
        
        if secure is not None:
            self.secure = secure
        else:
            self.secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
        
        # Create boto3 S3 client with MinIO endpoint
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",  # MinIO doesn't care, but boto3 needs it
        )
        
        self._bucket_ensured = False
        print(f"Initialized MinIOClient: endpoint={self.endpoint}, bucket={self.bucket}")

    async def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        if self._bucket_ensured:
            return
            
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(self._client.head_bucket, Bucket=self.bucket)
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchBucket"):
                await loop.run_in_executor(
                    None,
                    partial(self._client.create_bucket, Bucket=self.bucket)
                )
                print(f"Created bucket: {self.bucket}")
            else:
                raise
        
        self._bucket_ensured = True

    async def put_object(
        self, key: str, data: bytes, content_type: Optional[str] = None
    ) -> str:
        """Store binary data with the given key."""
        await self._ensure_bucket()
        
        loop = asyncio.get_event_loop()
        kwargs = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
            
        await loop.run_in_executor(
            None,
            partial(self._client.put_object, **kwargs)
        )
        print(f"Successfully stored object: {key}")
        return key

    async def get_object(self, key: str) -> Optional[bytes]:
        """Retrieve binary data by key."""
        await self._ensure_bucket()
        
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                partial(self._client.get_object, Bucket=self.bucket, Key=key)
            )
            return response["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise

    async def put_json(self, key: str, data: Dict[str, Any]) -> str:
        """Store a dictionary as JSON."""
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        return await self.put_object(key, json_bytes, content_type="application/json")

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve and parse JSON data by key."""
        data = await self.get_object(key)
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))

    async def delete_object(self, key: str) -> None:
        """Delete an object by key."""
        await self._ensure_bucket()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._client.delete_object, Bucket=self.bucket, Key=key)
        )
        print(f"Successfully deleted object: {key}")

    async def list_objects(self, prefix: str) -> List[str]:
        """List all object keys matching a prefix."""
        await self._ensure_bucket()
        
        loop = asyncio.get_event_loop()
        keys = []
        
        paginator = self._client.get_paginator("list_objects_v2")
        
        def _list_pages():
            result = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        result.append(obj["Key"])
            return result
        
        keys = await loop.run_in_executor(None, _list_pages)
        return keys

    async def upload_file(self, local_path: str, key: str) -> str:
        """Upload a local file to storage."""
        await self._ensure_bucket()
        
        loop = asyncio.get_event_loop()
        
        # Determine content type based on file extension
        content_type = self._guess_content_type(local_path)
        
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        
        await loop.run_in_executor(
            None,
            partial(
                self._client.upload_file,
                local_path,
                self.bucket,
                key,
                ExtraArgs=extra_args if extra_args else None,
            )
        )
        print(f"Successfully uploaded file: {local_path} -> {key}")
        return key

    async def download_file(self, key: str, local_path: str) -> str:
        """Download a file from storage to local filesystem."""
        await self._ensure_bucket()
        
        # Ensure parent directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._client.download_file, self.bucket, key, local_path)
        )
        print(f"Successfully downloaded file: {key} -> {local_path}")
        return local_path

    def get_url(self, key: str) -> str:
        """Get a URL for accessing an object."""
        # Generate presigned URL that expires in 1 hour
        url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=3600,
        )
        return url

    async def object_exists(self, key: str) -> bool:
        """Check if an object exists."""
        await self._ensure_bucket()
        
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(self._client.head_object, Bucket=self.bucket, Key=key)
            )
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise

    @staticmethod
    def _guess_content_type(filepath: str) -> Optional[str]:
        """Guess content type based on file extension."""
        ext = Path(filepath).suffix.lower()
        content_types = {
            ".json": "application/json",
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".srt": "text/plain",
            ".txt": "text/plain",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }
        return content_types.get(ext)


# Global instance for convenience
_minio_client: Optional[MinIOClient] = None


def get_storage_client() -> MinIOClient:
    """
    Get or create the global MinIO storage client instance.
    
    Returns:
        MinIOClient instance
    """
    global _minio_client
    
    if _minio_client is None:
        _minio_client = MinIOClient()
    
    return _minio_client


def reset_storage_client() -> None:
    """Reset the global storage client instance (useful for testing)."""
    global _minio_client
    _minio_client = None

