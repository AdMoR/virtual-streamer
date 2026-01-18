"""
RSS Feed Fetcher.

Fetches and parses RSS feeds from French news sources.
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional

import feedparser

from virtual_streamer.video_server.models import (
    ArticleContent,
    ArticleMetadata,
    NewsSource,
)

logger = logging.getLogger(__name__)

# Source URL to enum mapping
RSS_SOURCES: dict[str, NewsSource] = {
    "https://www.lemonde.fr/rss/une.xml": NewsSource.LE_MONDE,
    "https://www.francetvinfo.fr/titres.rss": NewsSource.FRANCE_INFO,
    "https://news.google.com/rss?hl=fr&gl=FR&ceid=FR:fr": NewsSource.GOOGLE_NEWS_FR,
    "https://www.bfmtv.com/rss/news-24-7/": NewsSource.BFM_TV,
    "https://www.20minutes.fr/feeds/rss-une.xml": NewsSource.TWENTY_MINUTES,
}


class RSSFetcher:
    """
    Fetches news articles from RSS feeds.
    
    Usage:
        fetcher = RSSFetcher()
        articles = await fetcher.fetch_all()
        
        # Or fetch specific sources
        fetcher = RSSFetcher(sources=["https://www.lemonde.fr/rss/une.xml"])
        articles = await fetcher.fetch_all()
    """

    def __init__(self, sources: Optional[list[str]] = None):
        """
        Initialize the RSS fetcher.
        
        Args:
            sources: List of RSS feed URLs. Defaults to all configured sources.
        """
        self.sources = sources or list(RSS_SOURCES.keys())

    def _generate_id(self, link: str) -> str:
        """Generate unique ID from article URL."""
        return hashlib.sha256(link.encode()).hexdigest()[:16]

    def _parse_date(self, entry) -> datetime:
        """Parse publication date from feed entry."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        return datetime.utcnow()

    def _get_storage_key(self, article_id: str, published_at: datetime) -> str:
        """Generate object storage path."""
        date_str = published_at.strftime("%Y-%m-%d")
        return f"articles/{date_str}/{article_id}.json"

    def _get_source_enum(self, url: str) -> NewsSource:
        """Get the NewsSource enum for a URL."""
        return RSS_SOURCES.get(url, NewsSource.GOOGLE_NEWS_FR)

    async def fetch_source(
        self, url: str
    ) -> list[tuple[ArticleMetadata, ArticleContent]]:
        """
        Fetch and parse a single RSS source.
        
        Args:
            url: RSS feed URL
            
        Returns:
            List of (metadata, content) tuples
        """
        source = self._get_source_enum(url)
        
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.error(f"Failed to parse feed {url}: {e}")
            return []

        if feed.bozo and feed.bozo_exception:
            logger.warning(f"Feed {url} has issues: {feed.bozo_exception}")

        articles: list[tuple[ArticleMetadata, ArticleContent]] = []
        
        for entry in feed.entries:
            try:
                link = entry.get("link", "")
                if not link:
                    continue

                article_id = self._generate_id(link)
                published_at = self._parse_date(entry)

                # Extract categories/tags
                categories = []
                if hasattr(entry, "tags"):
                    categories = [t.term for t in entry.tags if hasattr(t, "term")]

                content = ArticleContent(
                    title=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", "")),
                    link=link,
                    categories=categories,
                    author=entry.get("author"),
                )

                metadata = ArticleMetadata(
                    id=article_id,
                    title=content.title[:200],  # Truncate for DB
                    source=source,
                    published_at=published_at,
                    object_storage_key=self._get_storage_key(article_id, published_at),
                )

                articles.append((metadata, content))
                
            except Exception as e:
                logger.warning(f"Failed to parse entry from {url}: {e}")
                continue

        logger.info(f"Fetched {len(articles)} articles from {source.value}")
        return articles

    async def fetch_all(self) -> list[tuple[ArticleMetadata, ArticleContent]]:
        """
        Fetch articles from all configured sources.
        
        Returns:
            List of (metadata, content) tuples from all sources
        """
        all_articles: list[tuple[ArticleMetadata, ArticleContent]] = []
        
        for url in self.sources:
            try:
                articles = await self.fetch_source(url)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                continue

        logger.info(f"Fetched {len(all_articles)} total articles from {len(self.sources)} sources")
        return all_articles
