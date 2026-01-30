"""
Streaming Store for managing stream configurations, programmations, and playlists.

Provides MySQL-backed storage for streaming data with support for:
- Stream configuration CRUD
- Media programmation scheduling
- Playlist management with fallback logic
"""

import os
import json
import uuid
import aiomysql
from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import Optional, List, Dict, Any

from virtual_streamer.streaming.models import (
    StreamConfig,
    MediaProgrammation,
    PlaylistEntry,
    PlaylistStatus,
)


class StreamingStoreInterface(ABC):
    """Abstract interface for streaming data operations."""

    # Stream Config
    @abstractmethod
    async def create_stream(self, data: Dict[str, Any]) -> StreamConfig:
        """Create a new stream configuration."""
        pass

    @abstractmethod
    async def get_stream(self, stream_id: str) -> Optional[StreamConfig]:
        """Get a stream by ID."""
        pass

    @abstractmethod
    async def list_streams(self) -> List[StreamConfig]:
        """List all streams."""
        pass

    @abstractmethod
    async def update_stream(self, stream_id: str, data: Dict[str, Any]) -> Optional[StreamConfig]:
        """Update a stream configuration."""
        pass

    @abstractmethod
    async def delete_stream(self, stream_id: str) -> bool:
        """Delete a stream and all associated data."""
        pass

    # Programmation
    @abstractmethod
    async def create_programmation(self, data: Dict[str, Any]) -> MediaProgrammation:
        """Create a new media programmation."""
        pass

    @abstractmethod
    async def get_programmation(self, prog_id: str) -> Optional[MediaProgrammation]:
        """Get a programmation by ID."""
        pass

    @abstractmethod
    async def get_active_programmation(
        self, stream_id: str, at_time: time
    ) -> Optional[MediaProgrammation]:
        """Get the active programmation for a stream at a given time."""
        pass

    @abstractmethod
    async def list_programmations(self, stream_id: str) -> List[MediaProgrammation]:
        """List all programmations for a stream."""
        pass

    @abstractmethod
    async def update_programmation(
        self, prog_id: str, data: Dict[str, Any]
    ) -> Optional[MediaProgrammation]:
        """Update a programmation."""
        pass

    @abstractmethod
    async def delete_programmation(self, prog_id: str) -> bool:
        """Delete a programmation and its playlist."""
        pass

    # Playlist
    @abstractmethod
    async def add_to_playlist(
        self, prog_id: str, video_key: str, metadata: Dict[str, Any] = None
    ) -> PlaylistEntry:
        """Add a video to a programmation's playlist."""
        pass

    @abstractmethod
    async def get_next_video(self, prog_id: str) -> Optional[PlaylistEntry]:
        """Get the next video to play (pending first, then random played)."""
        pass

    @abstractmethod
    async def mark_as_playing(self, entry_id: str) -> None:
        """Mark a playlist entry as currently playing."""
        pass

    @abstractmethod
    async def mark_as_played(self, entry_id: str) -> None:
        """Mark a playlist entry as played."""
        pass

    @abstractmethod
    async def get_playlist(
        self, prog_id: str, status: Optional[str] = None
    ) -> List[PlaylistEntry]:
        """Get playlist entries for a programmation."""
        pass

    @abstractmethod
    async def get_playlist_entry(self, entry_id: str) -> Optional[PlaylistEntry]:
        """Get a specific playlist entry."""
        pass


class MySQLStreamingStore(StreamingStoreInterface):
    """MySQL-backed streaming store."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        database: str = None,
    ):
        self.host = host or os.environ.get("MYSQL_HOST", "localhost")
        self.port = port or int(os.environ.get("MYSQL_PORT", "3306"))
        self.user = user or os.environ.get("MYSQL_USER", "virtual_streamer")
        self.password = password or os.environ.get("MYSQL_PASSWORD", "")
        self.database = database or os.environ.get("MYSQL_DATABASE", "virtual_streamer")
        self._pool: Optional[aiomysql.Pool] = None
        print(f"Initialized MySQLStreamingStore for {self.host}:{self.port}/{self.database}")

    async def _get_pool(self) -> aiomysql.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                autocommit=True,
            )
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()

    # ========== Stream Config ==========

    async def create_stream(self, data: Dict[str, Any]) -> StreamConfig:
        """Create a new stream configuration."""
        pool = await self._get_pool()
        stream_id = data.get("stream_id", str(uuid.uuid4()))
        now = datetime.utcnow()

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO stream_configs (stream_id, name, description, is_active, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        stream_id,
                        data["name"],
                        data.get("description"),
                        data.get("is_active", True),
                        now,
                        now,
                    ),
                )

        return StreamConfig(
            stream_id=stream_id,
            name=data["name"],
            description=data.get("description"),
            is_active=data.get("is_active", True),
            created_at=now,
            updated_at=now,
        )

    async def get_stream(self, stream_id: str) -> Optional[StreamConfig]:
        """Get a stream by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT stream_id, name, description, is_active, created_at, updated_at
                    FROM stream_configs WHERE stream_id = %s
                    """,
                    (stream_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return self._row_to_stream_config(row)

    async def list_streams(self) -> List[StreamConfig]:
        """List all streams."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT stream_id, name, description, is_active, created_at, updated_at
                    FROM stream_configs ORDER BY created_at DESC
                    """
                )
                rows = await cur.fetchall()
                return [self._row_to_stream_config(row) for row in rows]

    async def update_stream(
        self, stream_id: str, data: Dict[str, Any]
    ) -> Optional[StreamConfig]:
        """Update a stream configuration."""
        pool = await self._get_pool()

        updates = []
        params = []

        if "name" in data:
            updates.append("name = %s")
            params.append(data["name"])
        if "description" in data:
            updates.append("description = %s")
            params.append(data["description"])
        if "is_active" in data:
            updates.append("is_active = %s")
            params.append(data["is_active"])

        if not updates:
            return await self.get_stream(stream_id)

        updates.append("updated_at = %s")
        params.append(datetime.utcnow())
        params.append(stream_id)

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE stream_configs SET {', '.join(updates)} WHERE stream_id = %s",
                    params,
                )

        return await self.get_stream(stream_id)

    async def delete_stream(self, stream_id: str) -> bool:
        """Delete a stream and all associated data (cascading)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM stream_configs WHERE stream_id = %s", (stream_id,)
                )
                return cur.rowcount > 0

    def _row_to_stream_config(self, row) -> StreamConfig:
        """Convert a database row to StreamConfig."""
        stream_id, name, description, is_active, created_at, updated_at = row
        return StreamConfig(
            stream_id=stream_id,
            name=name,
            description=description,
            is_active=bool(is_active),
            created_at=created_at,
            updated_at=updated_at,
        )

    # ========== Programmation ==========

    async def create_programmation(self, data: Dict[str, Any]) -> MediaProgrammation:
        """Create a new media programmation."""
        pool = await self._get_pool()
        prog_id = data.get("programmation_id", str(uuid.uuid4()))
        now = datetime.utcnow()

        # Handle time objects
        start_time = data["start_time"]
        end_time = data["end_time"]
        if isinstance(start_time, str):
            start_time = time.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = time.fromisoformat(end_time)

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO media_programmations 
                    (programmation_id, stream_id, story_template_id, name, start_time, end_time, priority, is_active, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        prog_id,
                        data["stream_id"],
                        data["story_template_id"],
                        data["name"],
                        start_time,
                        end_time,
                        data.get("priority", 0),
                        data.get("is_active", True),
                        now,
                    ),
                )

        return MediaProgrammation(
            programmation_id=prog_id,
            stream_id=data["stream_id"],
            story_template_id=data["story_template_id"],
            name=data["name"],
            start_time=start_time,
            end_time=end_time,
            priority=data.get("priority", 0),
            is_active=data.get("is_active", True),
            created_at=now,
        )

    async def get_programmation(self, prog_id: str) -> Optional[MediaProgrammation]:
        """Get a programmation by ID."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT programmation_id, stream_id, story_template_id, name, 
                           start_time, end_time, priority, is_active, created_at
                    FROM media_programmations WHERE programmation_id = %s
                    """,
                    (prog_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return self._row_to_programmation(row)

    async def get_active_programmation(
        self, stream_id: str, at_time: time
    ) -> Optional[MediaProgrammation]:
        """
        Get the active programmation for a stream at a given time.
        
        If multiple programmations overlap, return the one with highest priority.
        Returns None if no programmation is active.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT programmation_id, stream_id, story_template_id, name,
                           start_time, end_time, priority, is_active, created_at
                    FROM media_programmations
                    WHERE stream_id = %s
                      AND is_active = TRUE
                      AND start_time <= %s
                      AND end_time > %s
                    ORDER BY priority DESC
                    LIMIT 1
                    """,
                    (stream_id, at_time, at_time),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return self._row_to_programmation(row)

    async def list_programmations(self, stream_id: str) -> List[MediaProgrammation]:
        """List all programmations for a stream."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT programmation_id, stream_id, story_template_id, name,
                           start_time, end_time, priority, is_active, created_at
                    FROM media_programmations
                    WHERE stream_id = %s
                    ORDER BY priority DESC, start_time ASC
                    """,
                    (stream_id,),
                )
                rows = await cur.fetchall()
                return [self._row_to_programmation(row) for row in rows]

    async def update_programmation(
        self, prog_id: str, data: Dict[str, Any]
    ) -> Optional[MediaProgrammation]:
        """Update a programmation."""
        pool = await self._get_pool()

        updates = []
        params = []

        for field in ["stream_id", "story_template_id", "name", "priority", "is_active"]:
            if field in data:
                updates.append(f"{field} = %s")
                params.append(data[field])

        for field in ["start_time", "end_time"]:
            if field in data:
                updates.append(f"{field} = %s")
                value = data[field]
                if isinstance(value, str):
                    value = time.fromisoformat(value)
                params.append(value)

        if not updates:
            return await self.get_programmation(prog_id)

        params.append(prog_id)

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE media_programmations SET {', '.join(updates)} WHERE programmation_id = %s",
                    params,
                )

        return await self.get_programmation(prog_id)

    async def delete_programmation(self, prog_id: str) -> bool:
        """Delete a programmation and its playlist (cascading)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM media_programmations WHERE programmation_id = %s",
                    (prog_id,),
                )
                return cur.rowcount > 0

    def _row_to_programmation(self, row) -> MediaProgrammation:
        """Convert a database row to MediaProgrammation."""
        (
            prog_id,
            stream_id,
            story_template_id,
            name,
            start_time,
            end_time,
            priority,
            is_active,
            created_at,
        ) = row
        
        # Handle timedelta from MySQL TIME column
        if hasattr(start_time, 'total_seconds'):
            total_secs = int(start_time.total_seconds())
            start_time = time(total_secs // 3600, (total_secs % 3600) // 60, total_secs % 60)
        if hasattr(end_time, 'total_seconds'):
            total_secs = int(end_time.total_seconds())
            end_time = time(total_secs // 3600, (total_secs % 3600) // 60, total_secs % 60)
        
        return MediaProgrammation(
            programmation_id=prog_id,
            stream_id=stream_id,
            story_template_id=story_template_id,
            name=name,
            start_time=start_time,
            end_time=end_time,
            priority=priority,
            is_active=bool(is_active),
            created_at=created_at,
        )

    # ========== Playlist ==========

    async def add_to_playlist(
        self, prog_id: str, video_key: str, metadata: Dict[str, Any] = None
    ) -> PlaylistEntry:
        """Add a video to a programmation's playlist."""
        pool = await self._get_pool()
        entry_id = str(uuid.uuid4())
        now = datetime.utcnow()
        metadata = metadata or {}

        # Get next play_order
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COALESCE(MAX(play_order), -1) + 1 FROM playlist_entries WHERE programmation_id = %s",
                    (prog_id,),
                )
                row = await cur.fetchone()
                play_order = row[0] if row else 0

                await cur.execute(
                    """
                    INSERT INTO playlist_entries 
                    (entry_id, programmation_id, video_storage_key, status, play_order, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry_id,
                        prog_id,
                        video_key,
                        PlaylistStatus.PENDING.value,
                        play_order,
                        json.dumps(metadata),
                        now,
                    ),
                )

        return PlaylistEntry(
            entry_id=entry_id,
            programmation_id=prog_id,
            video_storage_key=video_key,
            status=PlaylistStatus.PENDING,
            play_order=play_order,
            metadata=metadata,
            created_at=now,
            played_at=None,
        )

    async def get_next_video(self, prog_id: str) -> Optional[PlaylistEntry]:
        """
        Get the next video to play from a programmation's playlist.
        
        Priority:
        1. Pending videos (by play_order, then created_at)
        2. Random from played videos (fallback/replay)
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Try pending first
                await cur.execute(
                    """
                    SELECT entry_id, programmation_id, video_storage_key, status,
                           play_order, metadata, created_at, played_at
                    FROM playlist_entries
                    WHERE programmation_id = %s AND status = 'pending'
                    ORDER BY play_order ASC, created_at ASC
                    LIMIT 1
                    """,
                    (prog_id,),
                )
                row = await cur.fetchone()

                if row:
                    return self._row_to_playlist_entry(row)

                # Fallback: random from played
                await cur.execute(
                    """
                    SELECT entry_id, programmation_id, video_storage_key, status,
                           play_order, metadata, created_at, played_at
                    FROM playlist_entries
                    WHERE programmation_id = %s AND status = 'played'
                    ORDER BY RAND()
                    LIMIT 1
                    """,
                    (prog_id,),
                )
                row = await cur.fetchone()

                if row:
                    return self._row_to_playlist_entry(row)

                return None

    async def mark_as_playing(self, entry_id: str) -> None:
        """Mark a playlist entry as currently playing."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE playlist_entries SET status = %s WHERE entry_id = %s",
                    (PlaylistStatus.PLAYING.value, entry_id),
                )

    async def mark_as_played(self, entry_id: str) -> None:
        """Mark a playlist entry as played."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE playlist_entries SET status = %s, played_at = %s WHERE entry_id = %s",
                    (PlaylistStatus.PLAYED.value, datetime.utcnow(), entry_id),
                )

    async def get_playlist(
        self, prog_id: str, status: Optional[str] = None
    ) -> List[PlaylistEntry]:
        """Get playlist entries for a programmation."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                if status:
                    await cur.execute(
                        """
                        SELECT entry_id, programmation_id, video_storage_key, status,
                               play_order, metadata, created_at, played_at
                        FROM playlist_entries
                        WHERE programmation_id = %s AND status = %s
                        ORDER BY play_order ASC, created_at ASC
                        """,
                        (prog_id, status),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT entry_id, programmation_id, video_storage_key, status,
                               play_order, metadata, created_at, played_at
                        FROM playlist_entries
                        WHERE programmation_id = %s
                        ORDER BY play_order ASC, created_at ASC
                        """,
                        (prog_id,),
                    )
                rows = await cur.fetchall()
                return [self._row_to_playlist_entry(row) for row in rows]

    async def get_playlist_entry(self, entry_id: str) -> Optional[PlaylistEntry]:
        """Get a specific playlist entry."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT entry_id, programmation_id, video_storage_key, status,
                           play_order, metadata, created_at, played_at
                    FROM playlist_entries
                    WHERE entry_id = %s
                    """,
                    (entry_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return self._row_to_playlist_entry(row)

    async def get_entries_played_since(
        self, stream_id: str, since: datetime
    ) -> List[PlaylistEntry]:
        """
        Get playlist entries played after a given timestamp.
        Joins with programmations to filter by stream.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT pe.entry_id, pe.programmation_id, pe.video_storage_key, 
                           pe.status, pe.play_order, pe.metadata, pe.created_at, pe.played_at
                    FROM playlist_entries pe
                    JOIN media_programmations mp ON pe.programmation_id = mp.programmation_id
                    WHERE mp.stream_id = %s 
                      AND pe.status = 'played'
                      AND pe.played_at > %s
                    ORDER BY pe.played_at ASC
                    """,
                    (stream_id, since),
                )
                rows = await cur.fetchall()
                return [self._row_to_playlist_entry(row) for row in rows]

    def _row_to_playlist_entry(self, row) -> PlaylistEntry:
        """Convert a database row to PlaylistEntry."""
        (
            entry_id,
            prog_id,
            video_key,
            status,
            play_order,
            metadata,
            created_at,
            played_at,
        ) = row

        # Parse JSON metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        return PlaylistEntry(
            entry_id=entry_id,
            programmation_id=prog_id,
            video_storage_key=video_key,
            status=PlaylistStatus(status),
            play_order=play_order,
            metadata=metadata,
            created_at=created_at,
            played_at=played_at,
        )


# Global store instance (lazy initialized)
_streaming_store: Optional[StreamingStoreInterface] = None


async def get_streaming_store() -> StreamingStoreInterface:
    """Get or create the global streaming store instance."""
    global _streaming_store
    if _streaming_store is None:
        _streaming_store = MySQLStreamingStore()
    return _streaming_store


def reset_streaming_store() -> None:
    """Reset the global streaming store instance (useful for testing)."""
    global _streaming_store
    _streaming_store = None
