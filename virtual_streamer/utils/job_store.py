"""
Job Store for tracking async job status.

Supports both in-memory storage (for development) and MySQL (for production).
"""

import os
import json
import aiomysql
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any


class JobStoreInterface(ABC):
    """Abstract interface for job storage."""

    @abstractmethod
    async def create_job(self, job_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new job record."""
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        pass

    @abstractmethod
    async def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a job's status/result."""
        pass

    @abstractmethod
    async def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent jobs."""
        pass

    @abstractmethod
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        pass

    @abstractmethod
    async def count_pending_jobs(self, story_template_id: str) -> int:
        """Count pending/running jobs for a specific story template."""
        pass


class InMemoryJobStore(JobStoreInterface):
    """In-memory job store for development/testing."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        print("Initialized InMemoryJobStore (jobs will be lost on restart)")

    async def create_job(self, job_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        job = {
            "job_id": job_id,
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": now,
            "request": request,
        }
        self._jobs[job_id] = job
        return job

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    async def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if job_id not in self._jobs:
            return None

        job = self._jobs[job_id]
        if status is not None:
            job["status"] = status
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        return job

    async def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        jobs = list(self._jobs.values())
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        return jobs[:limit]

    async def delete_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    async def count_pending_jobs(self, story_template_id: str) -> int:
        """Count pending/running jobs for a specific story template."""
        count = 0
        for job in self._jobs.values():
            if job["status"] in ("pending", "running"):
                request = job.get("request", {})
                if request.get("story_template_id") == story_template_id:
                    count += 1
        return count


class MySQLJobStore(JobStoreInterface):
    """MySQL-backed job store for production persistence."""

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
        print(f"Initialized MySQLJobStore for {self.host}:{self.port}/{self.database}")

    async def _get_pool(self) -> aiomysql.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            await self._ensure_database()
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

    async def _ensure_database(self):
        """Create database if it doesn't exist."""
        conn = await aiomysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            autocommit=True,
        )
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}`"
                )
            print(f"Ensured database '{self.database}' exists")
        finally:
            conn.close()

    async def _ensure_table(self):
        """Create jobs table if it doesn't exist."""
        pool = self._pool
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id VARCHAR(36) PRIMARY KEY,
                        status ENUM('pending', 'running', 'completed', 'failed') NOT NULL DEFAULT 'pending',
                        request JSON,
                        result JSON,
                        error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_status (status)
                    )
                """)
                print("Ensured 'jobs' table exists")

    async def create_job(self, job_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        pool = await self._get_pool()
        now = datetime.utcnow()
        request_json = json.dumps(request, default=str)

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO jobs (job_id, status, request, created_at)
                    VALUES (%s, 'pending', %s, %s)
                    """,
                    (job_id, request_json, now),
                )

        return {
            "job_id": job_id,
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": now.isoformat(),
            "request": request,
        }

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT job_id, status, request, result, error, created_at
                    FROM jobs WHERE job_id = %s
                    """,
                    (job_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    async def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()

        # Build dynamic UPDATE query
        updates = []
        params = []

        if status is not None:
            updates.append("status = %s")
            params.append(status)
        if result is not None:
            updates.append("result = %s")
            params.append(json.dumps(result, default=str))
        if error is not None:
            updates.append("error = %s")
            params.append(error)

        if not updates:
            return await self.get_job(job_id)

        params.append(job_id)

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = %s",
                    params,
                )

        return await self.get_job(job_id)

    async def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT job_id, status, request, result, error, created_at
                    FROM jobs ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
                return [self._row_to_dict(row) for row in rows]

    async def delete_job(self, job_id: str) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
                return cur.rowcount > 0

    async def count_pending_jobs(self, story_template_id: str) -> int:
        """Count pending/running jobs for a specific story template."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*) FROM jobs
                    WHERE status IN ('pending', 'running')
                      AND JSON_EXTRACT(request, '$.story_template_id') = %s
                    """,
                    (story_template_id,),
                )
                row = await cur.fetchone()
                return row[0] if row else 0

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a job dictionary."""
        job_id, status, request, result, error, created_at = row

        # Parse JSON fields
        if isinstance(request, str):
            request = json.loads(request)
        if isinstance(result, str):
            result = json.loads(result)

        return {
            "job_id": job_id,
            "status": status,
            "request": request,
            "result": result,
            "error": error,
            "created_at": created_at.isoformat() if created_at else None,
        }

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()


def get_job_store() -> JobStoreInterface:
    """Factory function to create job store based on configuration."""
    backend = os.environ.get("JOB_STORAGE_BACKEND", "memory").lower()

    if backend == "mysql":
        return MySQLJobStore()
    else:
        return InMemoryJobStore()


# Global job store instance (lazy initialized)
_job_store: Optional[JobStoreInterface] = None


async def get_global_job_store() -> JobStoreInterface:
    """Get or create the global job store instance."""
    global _job_store
    if _job_store is None:
        _job_store = get_job_store()
    return _job_store
