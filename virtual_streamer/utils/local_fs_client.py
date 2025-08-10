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
        self.base_path.mkdir(parents=True, exist_ok=True)
        print(f"Initialized LocalFSClient with base path: {self.base_path}")

    def _get_full_path(self, key: str, prefix: str="") -> Path:
        """Constructs the full, absolute path for a given key and ensures it's within the base path."""
        full_path = (self.base_path / prefix / key).resolve()
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    async def s3_put_json(self, key: str, data: Dict[str, Any]):
        """Saves a dictionary as a JSON file to the local filesystem."""
        full_path = self._get_full_path(key)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str) # Use default=str for datetime etc.
        print(f"Successfully wrote JSON to: {full_path}")

    async def s3_put_file(self, file_path: str, s3_prefix: str):
        """Saves a dictionary as a JSON file to the local filesystem."""
        full_path = self._get_full_path(os.path.basename(file_path), prefix=s3_prefix)
        shutil.copyfile(file_path, full_path)
        print(f"Successfully wrote file to: {full_path}")
        return str(full_path)

    async def s3_get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Loads and parses a JSON file from the local filesystem."""
        full_path = self._get_full_path(key)
        print("Full path = ", full_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return json.loads(content)

    async def s3_delete_object(self, key: str):
        """Deletes a file from the local filesystem."""
        full_path = self._get_full_path(key)
        if full_path.is_file():
            os.remove(full_path)
            print(f"Successfully deleted file: {full_path}")
        elif full_path.exists():
            print(f"Path exists but is not a file, not deleting: {full_path}")
            # Or raise error if deleting non-files is unexpected
        else:
             print(f"File not found, nothing to delete: {full_path}")
             pass # Deleting non-existent is often treated as success (idempotent)


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

        return keys
