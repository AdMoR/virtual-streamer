"""
Unit tests for Storage Interface and MinIO Client.

Tests cover:
- StorageInterface abstract methods
- MinIOClient initialization
- MinIOClient operations (mocked)
- get_storage_client singleton pattern
- reset_storage_client functionality
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

from virtual_streamer.utils.storage_interface import StorageInterface
from virtual_streamer.utils.minio_client import (
    MinIOClient,
    get_storage_client,
    reset_storage_client,
)


# ============================================================================
# StorageInterface Tests
# ============================================================================


class TestStorageInterfaceAbstract:
    """Test StorageInterface is properly abstract."""

    def test_cannot_instantiate_directly(self):
        """Test that StorageInterface cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            StorageInterface()

    def test_concrete_implementation_requires_all_methods(self):
        """Test that subclass must implement all abstract methods."""
        
        class IncompleteStorage(StorageInterface):
            async def put_object(self, key, data, content_type=None):
                pass
        
        with pytest.raises(TypeError, match="abstract"):
            IncompleteStorage()

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that a fully implemented subclass can be created."""
        
        class CompleteStorage(StorageInterface):
            async def put_object(self, key, data, content_type=None):
                return key
            async def get_object(self, key):
                return None
            async def put_json(self, key, data):
                return key
            async def get_json(self, key):
                return None
            async def delete_object(self, key):
                pass
            async def list_objects(self, prefix):
                return []
            async def upload_file(self, local_path, key):
                return key
            async def download_file(self, key, local_path):
                return local_path
            def get_url(self, key):
                return f"http://test/{key}"
            async def object_exists(self, key):
                return False
        
        storage = CompleteStorage()
        assert storage is not None


# ============================================================================
# MinIOClient Initialization Tests
# ============================================================================


class TestMinIOClientInit:
    """Test MinIOClient initialization."""

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_default_initialization(self, mock_boto3):
        """Test MinIOClient initializes with default values."""
        client = MinIOClient()
        
        assert client.endpoint == "http://minio:9000"
        assert client.access_key == "minioadmin"
        assert client.secret_key == "minioadmin"
        assert client.bucket == "virtual-streamer"
        assert client.secure is False

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_custom_initialization(self, mock_boto3):
        """Test MinIOClient initializes with custom values."""
        client = MinIOClient(
            endpoint="http://custom:9000",
            access_key="custom_key",
            secret_key="custom_secret",
            bucket="custom-bucket",
            secure=True,
        )
        
        assert client.endpoint == "http://custom:9000"
        assert client.access_key == "custom_key"
        assert client.secret_key == "custom_secret"
        assert client.bucket == "custom-bucket"
        assert client.secure is True

    @patch.dict(os.environ, {
        "MINIO_ENDPOINT": "http://env:9000",
        "MINIO_ACCESS_KEY": "env_key",
        "MINIO_SECRET_KEY": "env_secret",
        "MINIO_BUCKET": "env-bucket",
        "MINIO_SECURE": "true",
    })
    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_environment_variable_loading(self, mock_boto3):
        """Test MinIOClient loads from environment variables."""
        client = MinIOClient()
        
        assert client.endpoint == "http://env:9000"
        assert client.access_key == "env_key"
        assert client.secret_key == "env_secret"
        assert client.bucket == "env-bucket"
        assert client.secure is True

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_creates_boto3_client(self, mock_boto3):
        """Test that MinIOClient creates a boto3 S3 client."""
        client = MinIOClient()
        
        mock_boto3.client.assert_called_once()
        call_args = mock_boto3.client.call_args
        assert call_args[0][0] == "s3"
        assert "endpoint_url" in call_args[1]


# ============================================================================
# MinIOClient Operations Tests (Mocked)
# ============================================================================


class TestMinIOClientOperations:
    """Test MinIOClient operations with mocked boto3."""

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        return MagicMock()

    @pytest.fixture
    def minio_client(self, mock_s3_client):
        """Create a MinIOClient with mocked S3 client."""
        with patch('virtual_streamer.utils.minio_client.boto3') as mock_boto3:
            mock_boto3.client.return_value = mock_s3_client
            client = MinIOClient()
            client._bucket_ensured = True  # Skip bucket check
            return client

    @pytest.mark.asyncio
    async def test_put_object(self, minio_client, mock_s3_client):
        """Test put_object uploads data."""
        minio_client._client = mock_s3_client
        
        result = await minio_client.put_object("test/key", b"test data", "text/plain")
        
        assert result == "test/key"
        mock_s3_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_object_found(self, minio_client, mock_s3_client):
        """Test get_object returns data when found."""
        minio_client._client = mock_s3_client
        mock_body = MagicMock()
        mock_body.read.return_value = b"test data"
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        
        result = await minio_client.get_object("test/key")
        
        assert result == b"test data"

    @pytest.mark.asyncio
    async def test_get_object_not_found(self, minio_client, mock_s3_client):
        """Test get_object returns None when not found."""
        from botocore.exceptions import ClientError
        
        minio_client._client = mock_s3_client
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        
        result = await minio_client.get_object("missing/key")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_put_json(self, minio_client, mock_s3_client):
        """Test put_json stores JSON data."""
        minio_client._client = mock_s3_client
        
        data = {"name": "test", "value": 42}
        result = await minio_client.put_json("test/data.json", data)
        
        assert result == "test/data.json"
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "application/json"

    @pytest.mark.asyncio
    async def test_get_json(self, minio_client, mock_s3_client):
        """Test get_json retrieves and parses JSON."""
        minio_client._client = mock_s3_client
        mock_body = MagicMock()
        mock_body.read.return_value = b'{"name": "test", "value": 42}'
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        
        result = await minio_client.get_json("test/data.json")
        
        assert result == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_delete_object(self, minio_client, mock_s3_client):
        """Test delete_object removes an object."""
        minio_client._client = mock_s3_client
        
        await minio_client.delete_object("test/key")
        
        mock_s3_client.delete_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_objects(self, minio_client, mock_s3_client):
        """Test list_objects returns matching keys."""
        minio_client._client = mock_s3_client
        
        # Mock paginator
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "test/file1.json"}, {"Key": "test/file2.json"}]}
        ]
        mock_s3_client.get_paginator.return_value = mock_paginator
        
        result = await minio_client.list_objects("test/")
        
        assert result == ["test/file1.json", "test/file2.json"]

    @pytest.mark.asyncio
    async def test_object_exists_true(self, minio_client, mock_s3_client):
        """Test object_exists returns True when object exists."""
        minio_client._client = mock_s3_client
        
        result = await minio_client.object_exists("test/key")
        
        assert result is True
        mock_s3_client.head_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_object_exists_false(self, minio_client, mock_s3_client):
        """Test object_exists returns False when object doesn't exist."""
        from botocore.exceptions import ClientError
        
        minio_client._client = mock_s3_client
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )
        
        result = await minio_client.object_exists("missing/key")
        
        assert result is False

    def test_get_url(self, minio_client, mock_s3_client):
        """Test get_url generates a presigned URL."""
        minio_client._client = mock_s3_client
        mock_s3_client.generate_presigned_url.return_value = "http://test/url"
        
        result = minio_client.get_url("test/key")
        
        assert result == "http://test/url"
        mock_s3_client.generate_presigned_url.assert_called_once()


# ============================================================================
# MinIOClient Content Type Tests
# ============================================================================


class TestMinIOClientContentType:
    """Test content type detection."""

    def test_guess_content_type_json(self):
        """Test JSON content type detection."""
        assert MinIOClient._guess_content_type("file.json") == "application/json"

    def test_guess_content_type_mp4(self):
        """Test MP4 content type detection."""
        assert MinIOClient._guess_content_type("file.mp4") == "video/mp4"

    def test_guess_content_type_wav(self):
        """Test WAV content type detection."""
        assert MinIOClient._guess_content_type("file.wav") == "audio/wav"

    def test_guess_content_type_unknown(self):
        """Test unknown extension returns None."""
        assert MinIOClient._guess_content_type("file.xyz") is None

    def test_guess_content_type_case_insensitive(self):
        """Test content type detection is case insensitive."""
        assert MinIOClient._guess_content_type("file.JSON") == "application/json"
        assert MinIOClient._guess_content_type("file.MP4") == "video/mp4"


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalStorageInstance:
    """Test global storage client singleton."""

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_get_storage_client_returns_instance(self, mock_boto3):
        """Test get_storage_client returns a MinIOClient."""
        reset_storage_client()
        
        client = get_storage_client()
        
        assert isinstance(client, MinIOClient)
        reset_storage_client()

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_get_storage_client_singleton(self, mock_boto3):
        """Test get_storage_client returns same instance."""
        reset_storage_client()
        
        client1 = get_storage_client()
        client2 = get_storage_client()
        
        assert client1 is client2
        reset_storage_client()

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_reset_storage_client(self, mock_boto3):
        """Test reset_storage_client creates new instance."""
        reset_storage_client()
        
        client1 = get_storage_client()
        reset_storage_client()
        client2 = get_storage_client()
        
        assert client1 is not client2
        reset_storage_client()


# ============================================================================
# File Upload/Download Tests
# ============================================================================


class TestMinIOClientFileOperations:
    """Test file upload and download operations."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            filepath = f.name
        yield filepath
        if os.path.exists(filepath):
            os.unlink(filepath)

    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        return MagicMock()

    @pytest.fixture
    def minio_client(self, mock_s3_client):
        """Create a MinIOClient with mocked S3 client."""
        with patch('virtual_streamer.utils.minio_client.boto3') as mock_boto3:
            mock_boto3.client.return_value = mock_s3_client
            client = MinIOClient()
            client._bucket_ensured = True
            return client

    @pytest.mark.asyncio
    async def test_upload_file(self, minio_client, mock_s3_client, temp_file):
        """Test upload_file uploads a local file."""
        minio_client._client = mock_s3_client
        
        result = await minio_client.upload_file(temp_file, "uploads/test.txt")
        
        assert result == "uploads/test.txt"
        mock_s3_client.upload_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file(self, minio_client, mock_s3_client, temp_file):
        """Test download_file downloads to local path."""
        minio_client._client = mock_s3_client
        download_path = temp_file + ".downloaded"
        
        try:
            result = await minio_client.download_file("uploads/test.txt", download_path)
            
            assert result == download_path
            mock_s3_client.download_file.assert_called_once()
        finally:
            if os.path.exists(download_path):
                os.unlink(download_path)


# ============================================================================
# Bucket Ensure Tests
# ============================================================================


class TestMinIOClientBucketManagement:
    """Test bucket creation and management."""

    @pytest.mark.asyncio
    @patch('virtual_streamer.utils.minio_client.boto3')
    async def test_ensure_bucket_creates_when_missing(self, mock_boto3):
        """Test that bucket is created when it doesn't exist."""
        from botocore.exceptions import ClientError
        
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        
        client = MinIOClient()
        await client._ensure_bucket()
        
        mock_s3.create_bucket.assert_called_once()

    @pytest.mark.asyncio
    @patch('virtual_streamer.utils.minio_client.boto3')
    async def test_ensure_bucket_skips_when_exists(self, mock_boto3):
        """Test that bucket creation is skipped when bucket exists."""
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        
        client = MinIOClient()
        await client._ensure_bucket()
        
        mock_s3.create_bucket.assert_not_called()


# ============================================================================
# Integration with StorageInterface
# ============================================================================


class TestMinIOClientImplementsInterface:
    """Test that MinIOClient properly implements StorageInterface."""

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_implements_storage_interface(self, mock_boto3):
        """Test MinIOClient is a StorageInterface."""
        client = MinIOClient()
        
        assert isinstance(client, StorageInterface)

    @patch('virtual_streamer.utils.minio_client.boto3')
    def test_has_all_required_methods(self, mock_boto3):
        """Test MinIOClient has all required interface methods."""
        client = MinIOClient()
        
        # Check all required methods exist
        assert hasattr(client, 'put_object')
        assert hasattr(client, 'get_object')
        assert hasattr(client, 'put_json')
        assert hasattr(client, 'get_json')
        assert hasattr(client, 'delete_object')
        assert hasattr(client, 'list_objects')
        assert hasattr(client, 'upload_file')
        assert hasattr(client, 'download_file')
        assert hasattr(client, 'get_url')
        assert hasattr(client, 'object_exists')
        
        # Check they are callable
        assert callable(client.put_object)
        assert callable(client.get_object)
        assert callable(client.put_json)
        assert callable(client.get_json)
        assert callable(client.delete_object)
        assert callable(client.list_objects)
        assert callable(client.upload_file)
        assert callable(client.download_file)
        assert callable(client.get_url)
        assert callable(client.object_exists)


# ============================================================================
# Pytest Configuration
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

