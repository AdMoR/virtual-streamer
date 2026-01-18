"""
Hybrid Article Store.

Stores article metadata in SQLite for fast queries,
and article content in object storage for efficient storage.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

from virtual_streamer.video_server.models import (
    ArticleContent,
    ArticleMetadata,
    NewsArticle,
    NewsSource,
)
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.storage_interface import StorageInterface

logger = logging.getLogger(__name__)


class ArticleStore:
    """
    Hybrid storage: SQLite for metadata, object storage for content.
    
    SQLite schema:
        articles(id, title, source, published_at, fetched_at, object_storage_key, used_in_story)
    
    Object storage structure:
        {bucket}/articles/{YYYY-MM-DD}/{article_id}.json
    
    Usage:
        store = ArticleStore()
        
        # Save articles
        new_count = await store.save_articles(articles)
        
        # Query metadata (fast, SQLite only)
        recent = store.get_recent_metadata(hours=24)
        
        # Fetch full article when needed
        article = await store.get_full_article(metadata)
    """

    def __init__(
        self,
        db_path: str = "./data/news_articles.db",
        storage_client: Optional[StorageInterface] = None,
    ):
        """
        Initialize the article store.
        
        Args:
            db_path: Path to SQLite database file
            storage_client: Object storage client (defaults to MinIO)
        """
        self.db_path = db_path
        self.storage = storage_client or get_storage_client()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    fetched_at TIMESTAMP NOT NULL,
                    object_storage_key TEXT NOT NULL,
                    used_in_story TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_published_at 
                ON articles(published_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source 
                ON articles(source)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_used_in_story 
                ON articles(used_in_story)
            """)
        
        logger.info(f"Initialized article database at {self.db_path}")

    @contextmanager
    def _get_conn(self):
        """Get a database connection with automatic commit/close."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    async def save_article(
        self, metadata: ArticleMetadata, content: ArticleContent
    ) -> bool:
        """
        Save article: content to object storage, metadata to SQLite.
        
        Args:
            metadata: Article metadata
            content: Article content
            
        Returns:
            True if new article was saved, False if duplicate
        """
        # Check for duplicate
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM articles WHERE id = ?", (metadata.id,)
            ).fetchone()
            if existing:
                logger.debug(f"Article {metadata.id} already exists, skipping")
                return False

        # Store content in object storage
        await self.storage.put_json(
            metadata.object_storage_key, 
            content.model_dump()
        )

        # Store metadata in SQLite
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO articles 
                (id, title, source, published_at, fetched_at, object_storage_key)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.id,
                    metadata.title,
                    metadata.source.value,
                    metadata.published_at.isoformat(),
                    metadata.fetched_at.isoformat(),
                    metadata.object_storage_key,
                ),
            )

        logger.debug(f"Saved article {metadata.id}: {metadata.title[:50]}...")
        return True

    async def save_articles(
        self, articles: list[tuple[ArticleMetadata, ArticleContent]]
    ) -> int:
        """
        Save multiple articles.
        
        Args:
            articles: List of (metadata, content) tuples
            
        Returns:
            Count of new articles saved
        """
        new_count = 0
        for metadata, content in articles:
            if await self.save_article(metadata, content):
                new_count += 1
        
        logger.info(f"Saved {new_count} new articles (out of {len(articles)} total)")
        return new_count

    def _row_to_metadata(self, row: sqlite3.Row) -> ArticleMetadata:
        """Convert a database row to ArticleMetadata."""
        return ArticleMetadata(
            id=row["id"],
            title=row["title"],
            source=NewsSource(row["source"]),
            published_at=datetime.fromisoformat(row["published_at"]),
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            object_storage_key=row["object_storage_key"],
            used_in_story=row["used_in_story"],
        )

    def get_recent_metadata(
        self,
        hours: int = 24,
        source: Optional[NewsSource] = None,
        limit: int = 50,
    ) -> list[ArticleMetadata]:
        """
        Get metadata for recent articles (fast, no object storage call).
        
        Args:
            hours: How many hours back to look
            source: Optional filter by news source
            limit: Maximum number of articles to return
            
        Returns:
            List of ArticleMetadata objects
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        query = "SELECT * FROM articles WHERE published_at > ?"
        params: list = [cutoff.isoformat()]

        if source:
            query += " AND source = ?"
            params.append(source.value)

        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_metadata(row) for row in rows]

    async def get_full_article(self, metadata: ArticleMetadata) -> NewsArticle:
        """
        Fetch full article by loading content from object storage.
        
        Args:
            metadata: Article metadata (must have object_storage_key)
            
        Returns:
            Complete NewsArticle with content
        """
        content_data = await self.storage.get_json(metadata.object_storage_key)
        if content_data is None:
            raise ValueError(
                f"Article content not found at {metadata.object_storage_key}"
            )
        
        content = ArticleContent(**content_data)
        return NewsArticle(metadata=metadata, content=content)

    def get_unused_metadata(self, limit: int = 10) -> list[ArticleMetadata]:
        """
        Get articles not yet used for story generation.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of unused ArticleMetadata objects
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM articles 
                WHERE used_in_story IS NULL 
                ORDER BY published_at DESC 
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_metadata(row) for row in rows]

    def mark_used(self, article_id: str, story_id: str) -> None:
        """
        Mark article as used in a story.
        
        Args:
            article_id: Article ID to mark
            story_id: ID of the story that used this article
        """
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE articles SET used_in_story = ? WHERE id = ?",
                (story_id, article_id),
            )
        
        logger.info(f"Marked article {article_id} as used in story {story_id}")

    def get_article_count(self) -> int:
        """Get total count of articles in the store."""
        with self._get_conn() as conn:
            result = conn.execute("SELECT COUNT(*) FROM articles").fetchone()
            return result[0] if result else 0

    def get_metadata_by_id(self, article_id: str) -> Optional[ArticleMetadata]:
        """
        Get article metadata by ID.
        
        Args:
            article_id: Article ID
            
        Returns:
            ArticleMetadata if found, None otherwise
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM articles WHERE id = ?", (article_id,)
            ).fetchone()

        if row is None:
            return None
        
        return self._row_to_metadata(row)
