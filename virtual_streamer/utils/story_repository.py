"""
Story Repository — MySQL-backed storage for Story, Scene and ConditioningImageArtifact.

Mirrors the EntityRepository pattern: lazy pool initialisation, autocommit=True,
and a _ensure_tables() call on first connection. Binary files remain in MinIO;
only metadata and MinIO keys are stored here.

Tables created (in dependency order):
  stories                     — one row per generated story
  scenes                      — one row per video segment within a story
  conditioning_image_artifacts — provenance of the SD conditioning image per scene
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiomysql


class StoryRepository:
    """
    MySQL-backed repository for Story, Scene and ConditioningImageArtifact entities.
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
        print(f"Initialized StoryRepository for {self.host}:{self.port}/{self.database}")

    async def _get_pool(self) -> aiomysql.Pool:
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
            await self._ensure_tables()
        return self._pool

    async def _ensure_database(self):
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
            print(f"Ensured database '{self.database}' exists")
        finally:
            conn.close()

    async def _ensure_tables(self):
        pool = self._pool
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS stories (
                        story_id          CHAR(36)     NOT NULL PRIMARY KEY,
                        story_template_id VARCHAR(255) NOT NULL,
                        title             TEXT         NOT NULL,
                        story_plan        TEXT         NOT NULL,
                        status            ENUM('PENDING','GENERATING','COMPLETED','FAILED')
                                                       NOT NULL DEFAULT 'PENDING',
                        raw_agent_output  JSON         NOT NULL,
                        final_video_key   VARCHAR(512),
                        created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
                                          ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (story_template_id)
                            REFERENCES story_templates(id) ON DELETE RESTRICT
                    )
                """)

                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS scenes (
                        scene_id          CHAR(36)     NOT NULL PRIMARY KEY,
                        story_id          CHAR(36)     NOT NULL,
                        scene_index       INT          NOT NULL,
                        prompt            TEXT         NOT NULL,
                        video_segment_key VARCHAR(512),
                        guiding_video_key VARCHAR(512),
                        audio_key         VARCHAR(512),
                        speaker_id        VARCHAR(255),
                        spoken_line       TEXT,
                        location_id       VARCHAR(255),
                        duration_seconds  FLOAT,
                        raw_scene_data    JSON         NOT NULL,
                        created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (story_id)    REFERENCES stories(story_id)  ON DELETE CASCADE,
                        FOREIGN KEY (speaker_id)  REFERENCES characters(id)     ON DELETE SET NULL,
                        FOREIGN KEY (location_id) REFERENCES locations(id)      ON DELETE SET NULL
                    )
                """)

                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS conditioning_image_artifacts (
                        artifact_id          CHAR(36)     NOT NULL PRIMARY KEY,
                        scene_id             CHAR(36)     NOT NULL,
                        final_image_key      VARCHAR(512) NOT NULL,
                        character_image_keys JSON         NOT NULL,
                        location_image_key   VARCHAR(512),
                        flux_prompt_json     JSON         NOT NULL,
                        created_at           TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (scene_id) REFERENCES scenes(scene_id) ON DELETE CASCADE
                    )
                """)

                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS segment_candidates (
                        candidate_id      CHAR(36)     NOT NULL PRIMARY KEY,
                        scene_id          CHAR(36)     NOT NULL,
                        seed              BIGINT       NOT NULL,
                        video_key         VARCHAR(512),
                        image_key         VARCHAR(512),
                        generation_params JSON         NOT NULL,
                        judge_verdict     JSON,
                        judge_score       FLOAT,
                        judge_passed      TINYINT(1),
                        selected          TINYINT(1)   NOT NULL DEFAULT 0,
                        selection_source  ENUM('judge','human','fallback')
                                                       NOT NULL DEFAULT 'judge',
                        duration_seconds  FLOAT,
                        created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (scene_id) REFERENCES scenes(scene_id) ON DELETE CASCADE
                    )
                """)

                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS judge_feedback (
                        feedback_id    CHAR(36)     NOT NULL PRIMARY KEY,
                        candidate_id   CHAR(36)     NOT NULL,
                        user           VARCHAR(255) NOT NULL,
                        human_passed   TINYINT(1),
                        human_score    FLOAT,
                        artifact_tags  JSON,
                        comment        TEXT,
                        created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (candidate_id)
                            REFERENCES segment_candidates(candidate_id) ON DELETE CASCADE
                    )
                """)

                print("Ensured story tables exist")

    # ── Story CRUD ─────────────────────────────────────────────────────────────

    async def create_story(
        self,
        story_id: str,
        story_template_id: str,
        title: str,
        story_plan: str,
        raw_agent_output: dict,
        status: str = "PENDING",
    ) -> dict:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO stories
                        (story_id, story_template_id, title, story_plan, status, raw_agent_output)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (story_id, story_template_id, title, story_plan, status,
                     json.dumps(raw_agent_output, ensure_ascii=False)),
                )
        return await self.get_story(story_id)

    async def get_story(self, story_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT story_id, story_template_id, title, story_plan, status,
                           raw_agent_output, final_video_key, created_at, updated_at
                    FROM stories WHERE story_id = %s
                    """,
                    (story_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "story_id": row[0],
                    "story_template_id": row[1],
                    "title": row[2],
                    "story_plan": row[3],
                    "status": row[4],
                    "raw_agent_output": json.loads(row[5]) if row[5] else {},
                    "final_video_key": row[6],
                    "created_at": row[7].isoformat() if row[7] else None,
                    "updated_at": row[8].isoformat() if row[8] else None,
                }

    async def update_story_status(
        self,
        story_id: str,
        status: str,
        final_video_key: Optional[str] = None,
    ) -> Optional[dict]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                if final_video_key is not None:
                    await cur.execute(
                        "UPDATE stories SET status = %s, final_video_key = %s WHERE story_id = %s",
                        (status, final_video_key, story_id),
                    )
                else:
                    await cur.execute(
                        "UPDATE stories SET status = %s WHERE story_id = %s",
                        (status, story_id),
                    )
        return await self.get_story(story_id)

    async def list_stories(
        self,
        story_template_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                if story_template_id:
                    await cur.execute(
                        """
                        SELECT story_id, story_template_id, title, status,
                               final_video_key, created_at, updated_at
                        FROM stories WHERE story_template_id = %s
                        ORDER BY created_at DESC LIMIT %s
                        """,
                        (story_template_id, limit),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT story_id, story_template_id, title, status,
                               final_video_key, created_at, updated_at
                        FROM stories ORDER BY created_at DESC LIMIT %s
                        """,
                        (limit,),
                    )
                rows = await cur.fetchall()
                return [
                    {
                        "story_id": r[0],
                        "story_template_id": r[1],
                        "title": r[2],
                        "status": r[3],
                        "final_video_key": r[4],
                        "created_at": r[5].isoformat() if r[5] else None,
                        "updated_at": r[6].isoformat() if r[6] else None,
                    }
                    for r in rows
                ]

    async def delete_story(self, story_id: str) -> bool:
        """Delete a story; scenes, candidates and feedback cascade via FKs."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM stories WHERE story_id = %s", (story_id,))
                return cur.rowcount > 0

    # ── Scene CRUD ─────────────────────────────────────────────────────────────

    async def create_scene(
        self,
        scene_id: str,
        story_id: str,
        scene_index: int,
        prompt: str,
        raw_scene_data: dict,
        speaker_id: Optional[str] = None,
        spoken_line: Optional[str] = None,
        location_id: Optional[str] = None,
    ) -> dict:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO scenes
                        (scene_id, story_id, scene_index, prompt, raw_scene_data,
                         speaker_id, spoken_line, location_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (scene_id, story_id, scene_index, prompt,
                     json.dumps(raw_scene_data, ensure_ascii=False),
                     speaker_id, spoken_line, location_id),
                )
        return await self.get_scene(scene_id)

    async def get_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT scene_id, story_id, scene_index, prompt,
                           video_segment_key, guiding_video_key, audio_key,
                           speaker_id, spoken_line, location_id,
                           duration_seconds, raw_scene_data, created_at
                    FROM scenes WHERE scene_id = %s
                    """,
                    (scene_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "scene_id": row[0],
                    "story_id": row[1],
                    "scene_index": row[2],
                    "prompt": row[3],
                    "video_segment_key": row[4],
                    "guiding_video_key": row[5],
                    "audio_key": row[6],
                    "speaker_id": row[7],
                    "spoken_line": row[8],
                    "location_id": row[9],
                    "duration_seconds": row[10],
                    "raw_scene_data": json.loads(row[11]) if row[11] else {},
                    "created_at": row[12].isoformat() if row[12] else None,
                }

    async def update_scene_artifacts(
        self,
        scene_id: str,
        video_segment_key: Optional[str] = None,
        audio_key: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        guiding_video_key: Optional[str] = None,
    ) -> Optional[dict]:
        updates = []
        values = []
        if video_segment_key is not None:
            updates.append("video_segment_key = %s")
            values.append(video_segment_key)
        if audio_key is not None:
            updates.append("audio_key = %s")
            values.append(audio_key)
        if duration_seconds is not None:
            updates.append("duration_seconds = %s")
            values.append(duration_seconds)
        if guiding_video_key is not None:
            updates.append("guiding_video_key = %s")
            values.append(guiding_video_key)
        if not updates:
            return await self.get_scene(scene_id)
        values.append(scene_id)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE scenes SET {', '.join(updates)} WHERE scene_id = %s",
                    values,
                )
        return await self.get_scene(scene_id)

    async def list_scenes_for_story(self, story_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT scene_id, story_id, scene_index, prompt,
                           video_segment_key, guiding_video_key, audio_key,
                           speaker_id, spoken_line, location_id,
                           duration_seconds, created_at
                    FROM scenes WHERE story_id = %s ORDER BY scene_index ASC
                    """,
                    (story_id,),
                )
                rows = await cur.fetchall()
                return [
                    {
                        "scene_id": r[0],
                        "story_id": r[1],
                        "scene_index": r[2],
                        "prompt": r[3],
                        "video_segment_key": r[4],
                        "guiding_video_key": r[5],
                        "audio_key": r[6],
                        "speaker_id": r[7],
                        "spoken_line": r[8],
                        "location_id": r[9],
                        "duration_seconds": r[10],
                        "created_at": r[11].isoformat() if r[11] else None,
                    }
                    for r in rows
                ]

    # ── ConditioningImageArtifact CRUD ─────────────────────────────────────────

    async def create_conditioning_image_artifact(
        self,
        artifact_id: str,
        scene_id: str,
        final_image_key: str,
        character_image_keys: List[str],
        flux_prompt_json: dict,
        location_image_key: Optional[str] = None,
    ) -> dict:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO conditioning_image_artifacts
                        (artifact_id, scene_id, final_image_key,
                         character_image_keys, location_image_key, flux_prompt_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (artifact_id, scene_id, final_image_key,
                     json.dumps(character_image_keys),
                     location_image_key,
                     json.dumps(flux_prompt_json, ensure_ascii=False)),
                )
        return await self.get_conditioning_image_artifact(artifact_id)

    async def get_conditioning_image_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT artifact_id, scene_id, final_image_key,
                           character_image_keys, location_image_key,
                           flux_prompt_json, created_at
                    FROM conditioning_image_artifacts WHERE artifact_id = %s
                    """,
                    (artifact_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                return {
                    "artifact_id": row[0],
                    "scene_id": row[1],
                    "final_image_key": row[2],
                    "character_image_keys": json.loads(row[3]) if row[3] else [],
                    "location_image_key": row[4],
                    "flux_prompt_json": json.loads(row[5]) if row[5] else {},
                    "created_at": row[6].isoformat() if row[6] else None,
                }

    async def get_artifacts_for_scene(self, scene_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT artifact_id, scene_id, final_image_key,
                           character_image_keys, location_image_key,
                           flux_prompt_json, created_at
                    FROM conditioning_image_artifacts WHERE scene_id = %s
                    ORDER BY created_at ASC
                    """,
                    (scene_id,),
                )
                rows = await cur.fetchall()
                return [
                    {
                        "artifact_id": r[0],
                        "scene_id": r[1],
                        "final_image_key": r[2],
                        "character_image_keys": json.loads(r[3]) if r[3] else [],
                        "location_image_key": r[4],
                        "flux_prompt_json": json.loads(r[5]) if r[5] else {},
                        "created_at": r[6].isoformat() if r[6] else None,
                    }
                    for r in rows
                ]

    # ── SegmentCandidate CRUD ──────────────────────────────────────────────────

    _CANDIDATE_COLS = (
        "candidate_id, scene_id, seed, video_key, image_key, generation_params, "
        "judge_verdict, judge_score, judge_passed, selected, selection_source, "
        "duration_seconds, created_at"
    )

    @staticmethod
    def _candidate_row_to_dict(r) -> Dict[str, Any]:
        return {
            "candidate_id": r[0],
            "scene_id": r[1],
            "seed": r[2],
            "video_key": r[3],
            "image_key": r[4],
            "generation_params": json.loads(r[5]) if r[5] else {},
            "judge_verdict": json.loads(r[6]) if r[6] else None,
            "judge_score": r[7],
            "judge_passed": bool(r[8]) if r[8] is not None else None,
            "selected": bool(r[9]),
            "selection_source": r[10],
            "duration_seconds": r[11],
            "created_at": r[12].isoformat() if r[12] else None,
        }

    async def create_candidate(
        self,
        candidate_id: str,
        scene_id: str,
        seed: int,
        generation_params: dict,
        video_key: Optional[str] = None,
        image_key: Optional[str] = None,
        judge_verdict: Optional[dict] = None,
        duration_seconds: Optional[float] = None,
        selected: bool = False,
        selection_source: str = "judge",
    ) -> dict:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO segment_candidates
                        (candidate_id, scene_id, seed, video_key, image_key,
                         generation_params, judge_verdict, judge_score, judge_passed,
                         selected, selection_source, duration_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        candidate_id, scene_id, seed, video_key, image_key,
                        json.dumps(generation_params, ensure_ascii=False),
                        json.dumps(judge_verdict, ensure_ascii=False) if judge_verdict else None,
                        (judge_verdict or {}).get("score"),
                        (judge_verdict or {}).get("passed"),
                        int(selected), selection_source, duration_seconds,
                    ),
                )
        return await self.get_candidate(candidate_id)

    async def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {self._CANDIDATE_COLS} FROM segment_candidates WHERE candidate_id = %s",
                    (candidate_id,),
                )
                row = await cur.fetchone()
                return self._candidate_row_to_dict(row) if row else None

    async def list_candidates_for_scene(self, scene_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT {self._CANDIDATE_COLS} FROM segment_candidates
                    WHERE scene_id = %s ORDER BY created_at ASC
                    """,
                    (scene_id,),
                )
                return [self._candidate_row_to_dict(r) for r in await cur.fetchall()]

    async def set_selected_candidate(
        self, candidate_id: str, selection_source: str = "human"
    ) -> Optional[dict]:
        """Mark *candidate_id* as the selected take for its scene (unselects siblings)."""
        candidate = await self.get_candidate(candidate_id)
        if candidate is None:
            return None
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE segment_candidates SET selected = 0 WHERE scene_id = %s",
                    (candidate["scene_id"],),
                )
                await cur.execute(
                    """
                    UPDATE segment_candidates
                    SET selected = 1, selection_source = %s
                    WHERE candidate_id = %s
                    """,
                    (selection_source, candidate_id),
                )
        return await self.get_candidate(candidate_id)

    async def get_selected_candidates_for_story(self, story_id: str) -> List[Dict[str, Any]]:
        """Selected candidate per scene, ordered by scene_index. Scenes without candidates are absent."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT {', '.join('c.' + col.strip() for col in self._CANDIDATE_COLS.split(','))},
                           s.scene_index
                    FROM segment_candidates c
                    JOIN scenes s ON s.scene_id = c.scene_id
                    WHERE s.story_id = %s AND c.selected = 1
                    ORDER BY s.scene_index ASC
                    """,
                    (story_id,),
                )
                rows = await cur.fetchall()
                out = []
                for r in rows:
                    d = self._candidate_row_to_dict(r)
                    d["scene_index"] = r[13]
                    out.append(d)
                return out

    # ── JudgeFeedback CRUD ─────────────────────────────────────────────────────

    async def create_judge_feedback(
        self,
        feedback_id: str,
        candidate_id: str,
        user: str,
        human_passed: Optional[bool] = None,
        human_score: Optional[float] = None,
        artifact_tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
    ) -> dict:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO judge_feedback
                        (feedback_id, candidate_id, user, human_passed, human_score,
                         artifact_tags, comment)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        feedback_id, candidate_id, user,
                        int(human_passed) if human_passed is not None else None,
                        human_score,
                        json.dumps(artifact_tags or []),
                        comment,
                    ),
                )
        return {
            "feedback_id": feedback_id,
            "candidate_id": candidate_id,
            "user": user,
            "human_passed": human_passed,
            "human_score": human_score,
            "artifact_tags": artifact_tags or [],
            "comment": comment,
        }

    async def export_judge_feedback(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Human labels joined with the judge verdict — training data for judge improvement."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT f.feedback_id, f.candidate_id, f.user, f.human_passed,
                           f.human_score, f.artifact_tags, f.comment, f.created_at,
                           c.video_key, c.seed, c.judge_verdict, c.judge_score,
                           c.judge_passed, s.prompt, s.spoken_line
                    FROM judge_feedback f
                    JOIN segment_candidates c ON c.candidate_id = f.candidate_id
                    JOIN scenes s ON s.scene_id = c.scene_id
                    ORDER BY f.created_at DESC LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
                return [
                    {
                        "feedback_id": r[0],
                        "candidate_id": r[1],
                        "user": r[2],
                        "human_passed": bool(r[3]) if r[3] is not None else None,
                        "human_score": r[4],
                        "artifact_tags": json.loads(r[5]) if r[5] else [],
                        "comment": r[6],
                        "created_at": r[7].isoformat() if r[7] else None,
                        "video_key": r[8],
                        "seed": r[9],
                        "judge_verdict": json.loads(r[10]) if r[10] else None,
                        "judge_score": r[11],
                        "judge_passed": bool(r[12]) if r[12] is not None else None,
                        "scene_prompt": r[13],
                        "spoken_line": r[14],
                    }
                    for r in rows
                ]

    async def close(self):
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_story_repository: Optional[StoryRepository] = None


def get_story_repository() -> StoryRepository:
    global _story_repository
    if _story_repository is None:
        _story_repository = StoryRepository()
    return _story_repository
