"""
News Feed Reader Module.

Provides RSS feed reading and article storage for story generation.
"""

from virtual_streamer.news.models import (
    NewsSource,
    ArticleContent,
    ArticleMetadata,
    NewsArticle,
    NewsContext,
)
from virtual_streamer.news.fetcher import RSSFetcher
from virtual_streamer.news.store import ArticleStore
from virtual_streamer.news.selector import (
    ArticleSelector,
    NewestSelector,
    RandomSelector,
    SourcePrioritySelector,
    UnusedSelector,
)

__all__ = [
    # Models
    "NewsSource",
    "ArticleContent",
    "ArticleMetadata",
    "NewsArticle",
    "NewsContext",
    # Fetcher
    "RSSFetcher",
    # Store
    "ArticleStore",
    # Selectors
    "ArticleSelector",
    "NewestSelector",
    "RandomSelector",
    "SourcePrioritySelector",
    "UnusedSelector",
]
