"""
Video Search Client for the Video Embedding Server API.

Provides a client for searching video segments by text similarity
using VideoPrism embeddings stored in Qdrant.
"""

import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class TagInfo:
    """Tag information associated with a video segment."""

    name: str
    start: float
    end: float

    @classmethod
    def from_dict(cls, data: dict) -> "TagInfo":
        """Create TagInfo from API response dictionary."""
        return cls(
            name=data["name"],
            start=data["start"],
            end=data["end"],
        )


@dataclass
class VideoSearchResult:
    """Result from a video search query."""

    segment_id: str
    video_id: str
    segment_index: int
    duration: float
    path: str
    tags: list[TagInfo]
    similarity: float

    @classmethod
    def from_dict(cls, data: dict) -> "VideoSearchResult":
        """Create VideoSearchResult from API response dictionary."""
        return cls(
            segment_id=data["segment_id"],
            video_id=data["video_id"],
            segment_index=data["segment_index"],
            duration=data["duration"],
            path=data["path"],
            tags=[TagInfo.from_dict(t) for t in data.get("tags", [])],
            similarity=data["similarity"],
        )


class VideoSearchClient:
    """Client for the Video Embedding Similarity Server.

    Provides methods for searching video segments by text similarity
    using VideoPrism embeddings with optional tag filtering.

    Attributes:
        server_url: Base URL of the Video Embedding Server.

    Example:
        >>> client = VideoSearchClient()
        >>> results = client.search("person running", "my_videos", top_k=5)
        >>> for r in results:
        ...     print(f"{r.video_id}: {r.similarity:.4f}")
    """

    DEFAULT_SERVER_URL = "http://localhost:8003"

    def __init__(self, server_url: Optional[str] = None):
        """Initialize the Video Search Client.

        Args:
            server_url: Base URL of the Video Embedding Server.
                       Defaults to VIDEO_SEARCH_SERVER_URL env var or localhost:8003.
        """
        if server_url is None:
            server_url = os.environ.get(
                "VIDEO_SEARCH_SERVER_URL", self.DEFAULT_SERVER_URL
            )
        self.server_url = server_url.rstrip("/")

    def health(self) -> dict:
        """Check server health status.

        Returns:
            Dictionary with status, model, qdrant_host, and collections.

        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        response = requests.get(f"{self.server_url}/health")
        response.raise_for_status()
        return response.json()

    def list_collections(self) -> list[str]:
        """List all available Qdrant collections.

        Returns:
            List of collection names.

        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        response = requests.get(f"{self.server_url}/collections")
        response.raise_for_status()
        return response.json()["collections"]

    def list_tags(self, collection: str) -> list[str]:
        """List all unique tags in a collection.

        Args:
            collection: Name of the collection.

        Returns:
            List of tag names.

        Raises:
            requests.exceptions.RequestException: If the request fails.
            requests.exceptions.HTTPError: If collection not found (404).
        """
        response = requests.get(f"{self.server_url}/tags/{collection}")
        response.raise_for_status()
        return response.json()["tags"]

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        prompt_template: str = "a video of {}.",
        tags: Optional[list[str]] = None,
        tag_mode: str = "all",
    ) -> list[VideoSearchResult]:
        """Search for video segments matching a text query.

        Args:
            query: Natural language search query.
            collection: Qdrant collection to search.
            top_k: Number of results to return (default: 5).
            prompt_template: Template to wrap query (use {} as placeholder).
            tags: Optional list of tags to filter by.
            tag_mode: "all" (AND) or "any" (OR) for tag filtering.

        Returns:
            List of VideoSearchResult objects sorted by similarity (descending).

        Raises:
            requests.exceptions.RequestException: If the request fails.
            requests.exceptions.HTTPError: If collection not found (404) or
                invalid tag_mode (400).

        Example:
            >>> results = client.search(
            ...     query="person dancing",
            ...     collection="my_videos",
            ...     top_k=10,
            ...     tags=["person:john"],
            ...     tag_mode="all",
            ... )
        """
        payload = {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "prompt_template": prompt_template,
        }

        if tags:
            payload["tags"] = tags
            payload["tag_mode"] = tag_mode

        response = requests.post(f"{self.server_url}/search", json=payload)
        response.raise_for_status()

        results_data = response.json()["results"]
        return [VideoSearchResult.from_dict(r) for r in results_data]

    def search_raw(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        prompt_template: str = "a video of {}.",
        tags: Optional[list[str]] = None,
        tag_mode: str = "all",
    ) -> list[dict]:
        """Search for video segments, returning raw dictionary results.

        Same as search() but returns raw dictionaries instead of dataclass objects.
        Useful when you need the raw API response format.

        Args:
            query: Natural language search query.
            collection: Qdrant collection to search.
            top_k: Number of results to return (default: 5).
            prompt_template: Template to wrap query (use {} as placeholder).
            tags: Optional list of tags to filter by.
            tag_mode: "all" (AND) or "any" (OR) for tag filtering.

        Returns:
            List of result dictionaries from the API.
        """
        payload = {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "prompt_template": prompt_template,
        }

        if tags:
            payload["tags"] = tags
            payload["tag_mode"] = tag_mode

        response = requests.post(f"{self.server_url}/search", json=payload)
        response.raise_for_status()

        return response.json()["results"]



