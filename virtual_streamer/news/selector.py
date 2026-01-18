"""
Article Selectors.

Provides different strategies for selecting articles for story generation.
"""

import random
from abc import ABC, abstractmethod
from typing import Optional

from virtual_streamer.video_server.models import ArticleMetadata, NewsSource


class ArticleSelector(ABC):
    """
    Abstract interface for article selection strategies.
    
    Implement this interface to create custom selection logic.
    """

    @abstractmethod
    def select(self, articles: list[ArticleMetadata]) -> Optional[ArticleMetadata]:
        """
        Select one article from the list.
        
        Args:
            articles: List of article metadata to select from
            
        Returns:
            Selected ArticleMetadata, or None if list is empty
        """
        ...


class NewestSelector(ArticleSelector):
    """Select the most recently published article."""

    def select(self, articles: list[ArticleMetadata]) -> Optional[ArticleMetadata]:
        if not articles:
            return None
        return max(articles, key=lambda a: a.published_at)


class RandomSelector(ArticleSelector):
    """Select a random article."""

    def select(self, articles: list[ArticleMetadata]) -> Optional[ArticleMetadata]:
        if not articles:
            return None
        return random.choice(articles)


class SourcePrioritySelector(ArticleSelector):
    """
    Select newest article from preferred sources first.
    
    Falls back to other sources if preferred sources have no articles.
    
    Usage:
        selector = SourcePrioritySelector([
            NewsSource.LE_MONDE,
            NewsSource.FRANCE_INFO,
        ])
        article = selector.select(articles)
    """

    def __init__(self, priority: list[NewsSource]):
        """
        Initialize with source priority list.
        
        Args:
            priority: List of NewsSource in order of preference
        """
        self.priority = priority

    def select(self, articles: list[ArticleMetadata]) -> Optional[ArticleMetadata]:
        if not articles:
            return None

        # Try each source in priority order
        for source in self.priority:
            source_articles = [a for a in articles if a.source == source]
            if source_articles:
                return max(source_articles, key=lambda a: a.published_at)

        # Fallback to newest from any source
        return max(articles, key=lambda a: a.published_at)


class UnusedSelector(ArticleSelector):
    """
    Select the newest article that hasn't been used yet.
    
    Useful to ensure variety in story generation.
    """

    def select(self, articles: list[ArticleMetadata]) -> Optional[ArticleMetadata]:
        if not articles:
            return None

        # Filter to unused articles
        unused = [a for a in articles if a.used_in_story is None]
        
        if unused:
            return max(unused, key=lambda a: a.published_at)
        
        # If all used, return newest anyway
        return max(articles, key=lambda a: a.published_at)


# Future implementations (stubs):
# class LLMCuratedSelector(ArticleSelector):
#     """Use LLM to select the most interesting/relevant article."""
#     ...

# class ManualSelector(ArticleSelector):
#     """Allow manual selection via API endpoint."""
#     ...
