"""
News models - Re-exported from video_server.models for backward compatibility.

The canonical definitions are in virtual_streamer.video_server.models.
This module provides re-exports so existing imports continue to work.
"""

from virtual_streamer.video_server.models import (
    NewsSource,
    ArticleContent,
    ArticleMetadata,
    NewsArticle,
    NewsContext,
)

__all__ = [
    "NewsSource",
    "ArticleContent",
    "ArticleMetadata",
    "NewsArticle",
    "NewsContext",
]
