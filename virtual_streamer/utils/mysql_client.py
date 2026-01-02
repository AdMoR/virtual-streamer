"""
MySQL storage client for entity server.
Implements the same interface as LocalFSClient/AsyncS3Client,
storing JSON documents in a MySQL JSON column.
"""

import os
import json
import aiomysql
from typing import Optional, List, Dict, Any


class MySQLClient:
    """
    Storage client using MySQL with JSON column.
    Implements the same interface as LocalFSClient/AsyncS3Client.
    """

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
        print(f"Initialized MySQLClient for {self.host}:{self.port}/{self.database}")

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
            await self._ensure_table()
        return self._pool

    async def _ensure_table(self):
        """Create storage table if it doesn't exist."""
        pool = self._pool
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS entity_storage (
                        `key` VARCHAR(512) PRIMARY KEY,
                        `value` JSON NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                """)

    async def s3_put_json(self, key: str, data: Dict[str, Any]):
        """Store JSON data with the given key."""
        pool = await self._get_pool()
        json_str = json.dumps(data, default=str)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO entity_storage (`key`, `value`)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE `value` = VALUES(`value`), updated_at = NOW()
                    """,
                    (key, json_str),
                )
        print(f"Successfully stored JSON with key: {key}")

    async def s3_get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve JSON data by key."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT `value` FROM entity_storage WHERE `key` = %s",
                    (key,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                # MySQL JSON column returns dict directly or string depending on driver
                value = row[0]
                if isinstance(value, str):
                    return json.loads(value)
                return value

    async def s3_delete_object(self, key: str):
        """Delete an entry by key."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM entity_storage WHERE `key` = %s",
                    (key,),
                )
        print(f"Successfully deleted key: {key}")

    async def s3_list_keys(self, prefix: str) -> List[str]:
        """List all keys matching a prefix."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT `key` FROM entity_storage WHERE `key` LIKE %s",
                    (f"{prefix}%",),
                )
                rows = await cur.fetchall()
                return [row[0] for row in rows]

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()

