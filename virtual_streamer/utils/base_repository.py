"""
Shared MySQL plumbing for repositories.

Every repository talks to the same MySQL database: configuration comes from
the MYSQL_* environment variables, the connection pool is lazily created on
first use and shared across repository instances (one pool per credentials/
database tuple, not one per repository), and each repository ensures its own
tables once via the _create_tables() hook.

Subclasses implement _create_tables(cur) and build their queries on the
_execute/_fetch_one/_fetch_all helpers instead of handling pool/connection/
cursor acquisition themselves.
"""

import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

import aiomysql

logger = logging.getLogger(__name__)

# One pool per (host, port, user, database), shared by every repository.
_shared_pools: dict = {}


class BaseMySQLRepository:
    """Lazy shared-pool initialisation, autocommit=True, _create_tables() on first connection."""

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
        logger.debug(
            "Initialized %s for %s:%s/%s",
            type(self).__name__, self.host, self.port, self.database,
        )

    @property
    def _pool_key(self) -> tuple:
        return (self.host, self.port, self.user, self.database)

    async def _get_pool(self) -> aiomysql.Pool:
        if self._pool is None:
            pool = _shared_pools.get(self._pool_key)
            if pool is None:
                await self._ensure_database()
                pool = await aiomysql.create_pool(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    db=self.database,
                    autocommit=True,
                )
                _shared_pools[self._pool_key] = pool
            self._pool = pool
            await self._ensure_tables()
        return self._pool

    async def _ensure_database(self):
        """Create the database if it doesn't exist."""
        conn = await aiomysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            autocommit=True,
        )
        try:
            async with conn.cursor() as cur:
                await cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}`")
            logger.info("Ensured database '%s' exists", self.database)
        finally:
            conn.close()

    async def _ensure_tables(self):
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await self._create_tables(cur)
        logger.info("Ensured tables exist for %s", type(self).__name__)

    async def _create_tables(self, cur) -> None:
        """Run CREATE TABLE IF NOT EXISTS / migration statements on the given cursor."""
        raise NotImplementedError

    # ── Query helpers ───────────────────────────────────────────────────────────

    async def _execute(self, query: str, params: Optional[Sequence[Any]] = None) -> int:
        """Execute a statement; returns the affected row count."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return cur.rowcount

    async def _fetch_one(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> Optional[Tuple]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return await cur.fetchone()

    async def _fetch_all(
        self, query: str, params: Optional[Sequence[Any]] = None
    ) -> List[Tuple]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return list(await cur.fetchall())

    async def close(self):
        """Close the shared pool for this database (affects all repositories using it)."""
        pool = _shared_pools.pop(self._pool_key, None)
        if pool is not None:
            pool.close()
            await pool.wait_closed()
        self._pool = None
