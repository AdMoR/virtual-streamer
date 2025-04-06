import os
import json
import shutil
from pathlib import Path
from fastapi import HTTPException, status
from typing import Optional, List, Dict, Any

class LocalFSClient:
    """
    Mimics the AsyncS3Client interface but operates on the local filesystem.
    Keys are treated as relative paths within the base_path.
    """
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).resolve()
        # Ensure the base directory exists
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating base directory {self.base_path}: {e}")
            # Depending on use case, might want to raise an exception here
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to initialize local storage base path: {e}")
        print(f"Initialized LocalFSClient with base path: {self.base_path}")

    def _get_full_path(self, key: str) -> Path:
        """Constructs the full, absolute path for a given key and ensures it's within the base path."""
        # Normalize the key to prevent path traversal issues (e.g., '../')
        # os.path.normpath handles ., .., and normalizes slashes
        normalized_key = os.path.normpath(key)
        # Ensure the key doesn't try to escape the base path
        full_path = (self.base_path / normalized_key).resolve()

        # Security check: Ensure the resolved path is still within the base_path
        if self.base_path not in full_path.parents and full_path != self.base_path:
             # Check if the path itself is the base path (e.g. key was empty or '.')
            if str(full_path).startswith(str(self.base_path)):
                 # Allow paths directly under base_path
                 pass
            else:
                print(f"Attempted path traversal detected: Key '{key}' resolved outside base path '{self.base_path}'")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid key leading outside base storage path.")

        return full_path

    async def s3_put_json(self, key: str, data: Dict[str, Any]):
        """Saves a dictionary as a JSON file to the local filesystem."""
        full_path = self._get_full_path(key)
        try:
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str) # Use default=str for datetime etc.
            print(f"Successfully wrote JSON to: {full_path}")
        except OSError as e:
            print(f"Error writing JSON to local file {full_path}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Local filesystem write error: {e}")
        except TypeError as e:
            print(f"Error serializing data to JSON for key {key}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"JSON serialization error: {e}")

    async def s3_get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Loads and parses a JSON file from the local filesystem."""
        full_path = self._get_full_path(key)
        if not full_path.is_file():
            print(f"JSON file not found at: {full_path}")
            return None
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return json.loads(content)
        except FileNotFoundError:
            # Should be caught by is_file() check, but handle defensively
            print(f"JSON file not found (race condition?) at: {full_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from local file {full_path}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Local file is not valid JSON: {e}")
        except OSError as e:
            print(f"Error reading local file {full_path}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Local filesystem read error: {e}")

    async def s3_delete_object(self, key: str):
        """Deletes a file from the local filesystem."""
        full_path = self._get_full_path(key)
        try:
            if full_path.is_file():
                os.remove(full_path)
                print(f"Successfully deleted file: {full_path}")
            elif full_path.exists():
                print(f"Path exists but is not a file, not deleting: {full_path}")
                # Or raise error if deleting non-files is unexpected
            else:
                 print(f"File not found, nothing to delete: {full_path}")
                 pass # Deleting non-existent is often treated as success (idempotent)
        except OSError as e:
            print(f"Error deleting local file {full_path}: {e}")
            # Decide if this should be a critical error or just logged
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Local filesystem delete error: {e}")

    async def s3_list_keys(self, prefix: str) -> List[str]:
        """Lists keys (relative paths) matching a prefix within the base path."""
        prefix_path = self._get_full_path(prefix)
        keys = []

        # Ensure the prefix path is within the base path before walking
        if self.base_path not in prefix_path.parents and prefix_path != self.base_path:
             # Check if the path itself is the base path (e.g. prefix was empty or '.')
            if str(prefix_path).startswith(str(self.base_path)):
                 pass # Allow listing from base_path or subdirs
            else:
                print(f"List prefix '{prefix}' resolves outside base path '{self.base_path}'. Returning empty list.")
                return [] # Prefix is outside the allowed area

        # Check if the prefix itself exists and is a directory
        if not prefix_path.is_dir():
             # If it's a file matching the prefix, maybe return just that?
             # Or if it doesn't exist, return empty list.
             # Current behavior: only list contents if prefix is a directory.
             print(f"Prefix path is not a directory or does not exist: {prefix_path}")
             # Check if the prefix might exactly match a file
             if prefix_path.is_file() and str(prefix_path).startswith(str(self.base_path / prefix)):
                 try:
                     relative_path = prefix_path.relative_to(self.base_path)
                     return [str(relative_path).replace(os.sep, '/')] # Normalize to forward slashes like S3 keys
                 except ValueError:
                     return [] # Should not happen if logic is correct
             return []


        try:
            for item in prefix_path.rglob('*'): # Recursive glob
                if item.is_file():
                    # Convert absolute path back to relative key
                    try:
                        relative_path = item.relative_to(self.base_path)
                        # Normalize slashes to be consistent with S3 keys (forward slash)
                        keys.append(str(relative_path).replace(os.sep, '/'))
                    except ValueError:
                        # This shouldn't happen if item is within prefix_path which is within base_path
                        print(f"Warning: Found item {item} outside base path during listing?")
                        continue
        except OSError as e:
            print(f"Error listing local directory {prefix_path}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Local filesystem list error: {e}")

        return keys
