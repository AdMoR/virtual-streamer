"""
Low-level API: News Article Management

Handles article fetching, storage, and retrieval.
Uses hybrid storage (SQLite + object storage).
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional

from virtual_streamer.video_server.models import (
    ArticleMetadata,
    NewsArticle,
    NewsSource,
    NewsContext,
)
from virtual_streamer.news.store import ArticleStore
from virtual_streamer.news.fetcher import RSSFetcher

# Router setup
router = APIRouter(prefix="/articles", tags=["News Articles"])

# Lazy-initialized store
_store: Optional[ArticleStore] = None


def get_store() -> ArticleStore:
    """Get or create the article store singleton."""
    global _store
    if _store is None:
        _store = ArticleStore()
    return _store


@router.post("/fetch", status_code=status.HTTP_200_OK)
async def fetch_articles(sources: Optional[List[str]] = None):
    """
    Fetch fresh articles from RSS feeds.
    
    Args:
        sources: Optional list of RSS feed URLs to fetch from.
                 If not provided, fetches from all configured sources.
    
    Returns:
        Count of fetched and new articles
    """
    fetcher = RSSFetcher(sources=sources)
    articles = await fetcher.fetch_all()

    store = get_store()
    new_count = await store.save_articles(articles)

    return {"fetched": len(articles), "new": new_count}


@router.get("", response_model=List[ArticleMetadata])
async def list_articles(
    hours: int = Query(24, ge=1, le=168, description="How many hours back to look"),
    source: Optional[NewsSource] = Query(None, description="Filter by news source"),
    limit: int = Query(50, ge=1, le=500, description="Max articles to return"),
):
    """
    List recent article metadata.
    
    Returns lightweight metadata without full content for fast queries.
    Use GET /articles/{article_id} for full content.
    """
    store = get_store()
    return store.get_recent_metadata(hours=hours, source=source, limit=limit)


@router.get("/unused", response_model=List[ArticleMetadata])
async def list_unused_articles(
    limit: int = Query(10, ge=1, le=100, description="Max articles to return"),
):
    """
    List articles not yet used for story generation.
    
    Useful for selecting articles to generate new stories from.
    """
    store = get_store()
    return store.get_unused_metadata(limit=limit)


@router.get("/count", status_code=status.HTTP_200_OK)
async def get_article_count():
    """Get total count of articles in the store."""
    store = get_store()
    return {"count": store.get_article_count()}


@router.get("/{article_id}", response_model=NewsArticle)
async def get_article(article_id: str):
    """
    Get full article with content.
    
    Fetches metadata from SQLite and content from object storage.
    """
    store = get_store()
    metadata = store.get_metadata_by_id(article_id)

    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Article not found",
        )

    return await store.get_full_article(metadata)


@router.get("/{article_id}/context", response_model=NewsContext)
async def get_article_context(article_id: str):
    """
    Get article formatted as story generation context.
    
    Returns a NewsContext object ready to be used in story generation prompts.
    """
    store = get_store()
    metadata = store.get_metadata_by_id(article_id)

    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Article not found",
        )

    article = await store.get_full_article(metadata)
    return NewsContext.from_article(article)


@router.post("/{article_id}/mark-used", status_code=status.HTTP_200_OK)
async def mark_article_used(article_id: str, story_id: str = Query(..., description="ID of the story that used this article")):
    """
    Mark an article as used in a story.
    
    This helps track which articles have been used and enables
    the /unused endpoint to filter them out.
    """
    store = get_store()
    
    # Verify article exists
    metadata = store.get_metadata_by_id(article_id)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Article not found",
        )
    
    store.mark_used(article_id, story_id)
    return {"status": "ok", "article_id": article_id, "story_id": story_id}


@router.get("/{article_id}/metadata", response_model=ArticleMetadata)
async def get_article_metadata(article_id: str):
    """
    Get article metadata only (fast, no object storage call).
    
    Use this when you only need metadata like title, source, and dates.
    """
    store = get_store()
    metadata = store.get_metadata_by_id(article_id)

    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Article not found",
        )

    return metadata
