"""
Eval Repository — MySQL-backed storage for the generic evaluation bench.

Built on BaseMySQLRepository (shared lazy pool, autocommit=True, table
creation on first connection). Binary artifacts remain in MinIO (prefix
eval/{run_id}/); only metadata and keys are stored here.

Model:
  eval_benches  — named suite for one model kind, with a JSON dataset of cases
  eval_runs     — one execution of a bench against a model/config (comparison unit)
  eval_samples  — one generation per case: MinIO artifact or JSON output
  eval_feedback — human label on a sample (pass/fail, score, tags, comment)
"""

import json
from typing import Any, Dict, List, Optional

from virtual_streamer.utils.base_repository import BaseMySQLRepository

MODEL_KINDS = ("image", "video", "tts", "llm_agent")


class EvalRepository(BaseMySQLRepository):
    """MySQL-backed repository for eval benches, runs, samples and feedback."""

    async def _create_tables(self, cur):
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_benches (
                bench_id    CHAR(36)     NOT NULL PRIMARY KEY,
                name        VARCHAR(255) NOT NULL UNIQUE,
                model_kind  ENUM('image','video','tts','llm_agent') NOT NULL,
                description TEXT,
                judge_agent VARCHAR(255),
                cases       JSON         NOT NULL,
                created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id       CHAR(36)     NOT NULL PRIMARY KEY,
                bench_id     CHAR(36)     NOT NULL,
                label        VARCHAR(255),
                model_id     VARCHAR(255) NOT NULL,
                model_config JSON         NOT NULL,
                status       ENUM('running','completed','failed')
                                          NOT NULL DEFAULT 'running',
                error        TEXT,
                created_at   TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (bench_id) REFERENCES eval_benches(bench_id) ON DELETE CASCADE
            )
        """)

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_samples (
                sample_id       CHAR(36)     NOT NULL PRIMARY KEY,
                run_id          CHAR(36)     NOT NULL,
                case_id         VARCHAR(128) NOT NULL,
                input_params    JSON         NOT NULL,
                artifact_key    VARCHAR(512),
                output_json     JSON,
                auto_score      FLOAT,
                auto_verdict    JSON,
                candidate_id    CHAR(36),
                status          ENUM('pending','ok','error')
                                             NOT NULL DEFAULT 'pending',
                error           TEXT,
                latency_seconds FLOAT,
                created_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
            )
        """)

        await cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_feedback (
                feedback_id              CHAR(36)     NOT NULL PRIMARY KEY,
                sample_id                CHAR(36)     NOT NULL,
                user                     VARCHAR(255) NOT NULL,
                passed                   TINYINT(1),
                score                    FLOAT,
                tags                     JSON,
                comment                  TEXT,
                preferred_over_sample_id CHAR(36),
                created_at               TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_id)
                    REFERENCES eval_samples(sample_id) ON DELETE CASCADE
            )
        """)

    # ── Bench CRUD ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bench_row_to_dict(r) -> Dict[str, Any]:
        return {
            "bench_id": r[0],
            "name": r[1],
            "model_kind": r[2],
            "description": r[3],
            "judge_agent": r[4],
            "cases": json.loads(r[5]) if r[5] else [],
            "created_at": r[6].isoformat() if r[6] else None,
        }

    _BENCH_COLS = "bench_id, name, model_kind, description, judge_agent, cases, created_at"

    async def upsert_bench(
        self,
        bench_id: str,
        name: str,
        model_kind: str,
        cases: List[dict],
        description: Optional[str] = None,
        judge_agent: Optional[str] = None,
    ) -> dict:
        """Insert a bench, or update an existing bench with the same name."""
        await self._execute(
            """
            INSERT INTO eval_benches
                (bench_id, name, model_kind, description, judge_agent, cases)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_kind = VALUES(model_kind),
                description = VALUES(description),
                judge_agent = VALUES(judge_agent),
                cases = VALUES(cases)
            """,
            (bench_id, name, model_kind, description, judge_agent,
             json.dumps(cases, ensure_ascii=False)),
        )
        return await self.get_bench_by_name(name)

    async def get_bench(self, bench_id: str) -> Optional[Dict[str, Any]]:
        row = await self._fetch_one(
            f"SELECT {self._BENCH_COLS} FROM eval_benches WHERE bench_id = %s",
            (bench_id,),
        )
        return self._bench_row_to_dict(row) if row else None

    async def get_bench_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        row = await self._fetch_one(
            f"SELECT {self._BENCH_COLS} FROM eval_benches WHERE name = %s",
            (name,),
        )
        return self._bench_row_to_dict(row) if row else None

    async def list_benches(self) -> List[Dict[str, Any]]:
        rows = await self._fetch_all(
            f"SELECT {self._BENCH_COLS} FROM eval_benches ORDER BY model_kind, name"
        )
        return [self._bench_row_to_dict(r) for r in rows]

    # ── Run CRUD ────────────────────────────────────────────────────────────────

    async def create_run(
        self,
        run_id: str,
        bench_id: str,
        model_id: str,
        model_config: dict,
        label: Optional[str] = None,
    ) -> dict:
        await self._execute(
            """
            INSERT INTO eval_runs (run_id, bench_id, label, model_id, model_config)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_id, bench_id, label, model_id,
             json.dumps(model_config, ensure_ascii=False)),
        )
        return await self.get_run(run_id)

    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = await self._fetch_one(
            """
            SELECT run_id, bench_id, label, model_id, model_config,
                   status, error, created_at
            FROM eval_runs WHERE run_id = %s
            """,
            (run_id,),
        )
        if row is None:
            return None
        return {
            "run_id": row[0],
            "bench_id": row[1],
            "label": row[2],
            "model_id": row[3],
            "model_config": json.loads(row[4]) if row[4] else {},
            "status": row[5],
            "error": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
        }

    async def finish_run(self, run_id: str, status: str, error: Optional[str] = None):
        await self._execute(
            "UPDATE eval_runs SET status = %s, error = %s WHERE run_id = %s",
            (status, error, run_id),
        )

    async def list_runs_for_bench(self, bench_id: str) -> List[Dict[str, Any]]:
        """Runs for a bench, each with aggregate sample/feedback scores."""
        rows = await self._fetch_all(
            """
            SELECT r.run_id, r.label, r.model_id, r.model_config, r.status,
                   r.error, r.created_at,
                   COUNT(DISTINCT s.sample_id)                        AS n_samples,
                   SUM(s.status = 'error')                            AS n_errors,
                   AVG(s.auto_score)                                  AS avg_auto_score,
                   AVG(s.latency_seconds)                             AS avg_latency,
                   COUNT(f.feedback_id)                               AS n_feedback,
                   AVG(f.score)                                       AS avg_human_score,
                   AVG(f.passed)                                      AS human_pass_rate
            FROM eval_runs r
            LEFT JOIN eval_samples s ON s.run_id = r.run_id
            LEFT JOIN eval_feedback f ON f.sample_id = s.sample_id
            WHERE r.bench_id = %s
            GROUP BY r.run_id
            ORDER BY r.created_at DESC
            """,
            (bench_id,),
        )
        return [
            {
                "run_id": r[0],
                "label": r[1],
                "model_id": r[2],
                "model_config": json.loads(r[3]) if r[3] else {},
                "status": r[4],
                "error": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
                "n_samples": int(r[7] or 0),
                "n_errors": int(r[8] or 0),
                "avg_auto_score": float(r[9]) if r[9] is not None else None,
                "avg_latency": float(r[10]) if r[10] is not None else None,
                "n_feedback": int(r[11] or 0),
                "avg_human_score": float(r[12]) if r[12] is not None else None,
                "human_pass_rate": float(r[13]) if r[13] is not None else None,
            }
            for r in rows
        ]

    # ── Sample CRUD ─────────────────────────────────────────────────────────────

    _SAMPLE_COLS = (
        "sample_id, run_id, case_id, input_params, artifact_key, output_json, "
        "auto_score, auto_verdict, candidate_id, status, error, latency_seconds, created_at"
    )

    @staticmethod
    def _sample_row_to_dict(r) -> Dict[str, Any]:
        return {
            "sample_id": r[0],
            "run_id": r[1],
            "case_id": r[2],
            "input_params": json.loads(r[3]) if r[3] else {},
            "artifact_key": r[4],
            "output_json": json.loads(r[5]) if r[5] else None,
            "auto_score": r[6],
            "auto_verdict": json.loads(r[7]) if r[7] else None,
            "candidate_id": r[8],
            "status": r[9],
            "error": r[10],
            "latency_seconds": r[11],
            "created_at": r[12].isoformat() if r[12] else None,
        }

    async def create_sample(
        self,
        sample_id: str,
        run_id: str,
        case_id: str,
        input_params: dict,
        artifact_key: Optional[str] = None,
        output_json: Optional[Any] = None,
        auto_score: Optional[float] = None,
        auto_verdict: Optional[dict] = None,
        candidate_id: Optional[str] = None,
        status: str = "ok",
        error: Optional[str] = None,
        latency_seconds: Optional[float] = None,
    ) -> dict:
        await self._execute(
            """
            INSERT INTO eval_samples
                (sample_id, run_id, case_id, input_params, artifact_key,
                 output_json, auto_score, auto_verdict, candidate_id,
                 status, error, latency_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                sample_id, run_id, case_id,
                json.dumps(input_params, ensure_ascii=False),
                artifact_key,
                json.dumps(output_json, ensure_ascii=False)
                if output_json is not None else None,
                auto_score,
                json.dumps(auto_verdict, ensure_ascii=False) if auto_verdict else None,
                candidate_id, status, error, latency_seconds,
            ),
        )
        return await self.get_sample(sample_id)

    async def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        row = await self._fetch_one(
            f"SELECT {self._SAMPLE_COLS} FROM eval_samples WHERE sample_id = %s",
            (sample_id,),
        )
        return self._sample_row_to_dict(row) if row else None

    async def list_samples_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        rows = await self._fetch_all(
            f"""
            SELECT {self._SAMPLE_COLS} FROM eval_samples
            WHERE run_id = %s ORDER BY created_at ASC
            """,
            (run_id,),
        )
        return [self._sample_row_to_dict(r) for r in rows]

    async def sample_exists_for_candidate(self, candidate_id: str) -> bool:
        row = await self._fetch_one(
            "SELECT 1 FROM eval_samples WHERE candidate_id = %s LIMIT 1",
            (candidate_id,),
        )
        return row is not None

    # ── Feedback ────────────────────────────────────────────────────────────────

    async def create_feedback(
        self,
        feedback_id: str,
        sample_id: str,
        user: str,
        passed: Optional[bool] = None,
        score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
        preferred_over_sample_id: Optional[str] = None,
    ) -> dict:
        await self._execute(
            """
            INSERT INTO eval_feedback
                (feedback_id, sample_id, user, passed, score, tags,
                 comment, preferred_over_sample_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                feedback_id, sample_id, user,
                int(passed) if passed is not None else None,
                score,
                json.dumps(tags or []),
                comment,
                preferred_over_sample_id,
            ),
        )
        return {
            "feedback_id": feedback_id,
            "sample_id": sample_id,
            "user": user,
            "passed": passed,
            "score": score,
            "tags": tags or [],
            "comment": comment,
            "preferred_over_sample_id": preferred_over_sample_id,
        }

    async def list_feedback_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        rows = await self._fetch_all(
            """
            SELECT f.feedback_id, f.sample_id, f.user, f.passed, f.score,
                   f.tags, f.comment, f.preferred_over_sample_id, f.created_at
            FROM eval_feedback f
            JOIN eval_samples s ON s.sample_id = f.sample_id
            WHERE s.run_id = %s ORDER BY f.created_at ASC
            """,
            (run_id,),
        )
        return [
            {
                "feedback_id": r[0],
                "sample_id": r[1],
                "user": r[2],
                "passed": bool(r[3]) if r[3] is not None else None,
                "score": r[4],
                "tags": json.loads(r[5]) if r[5] else [],
                "comment": r[6],
                "preferred_over_sample_id": r[7],
                "created_at": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]

    async def export_feedback(
        self, bench_id: Optional[str] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Human labels joined with sample, run and bench context — training data."""
        where = "WHERE b.bench_id = %s" if bench_id else ""
        params = (bench_id, limit) if bench_id else (limit,)
        rows = await self._fetch_all(
            f"""
            SELECT f.feedback_id, f.user, f.passed, f.score, f.tags,
                   f.comment, f.created_at,
                   s.sample_id, s.case_id, s.input_params, s.artifact_key,
                   s.output_json, s.auto_score, s.auto_verdict,
                   r.run_id, r.model_id, r.label,
                   b.bench_id, b.name, b.model_kind
            FROM eval_feedback f
            JOIN eval_samples s ON s.sample_id = f.sample_id
            JOIN eval_runs r ON r.run_id = s.run_id
            JOIN eval_benches b ON b.bench_id = r.bench_id
            {where}
            ORDER BY f.created_at DESC LIMIT %s
            """,
            params,
        )
        return [
            {
                "feedback_id": r[0],
                "user": r[1],
                "passed": bool(r[2]) if r[2] is not None else None,
                "score": r[3],
                "tags": json.loads(r[4]) if r[4] else [],
                "comment": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
                "sample_id": r[7],
                "case_id": r[8],
                "input_params": json.loads(r[9]) if r[9] else {},
                "artifact_key": r[10],
                "output_json": json.loads(r[11]) if r[11] else None,
                "auto_score": r[12],
                "auto_verdict": json.loads(r[13]) if r[13] else None,
                "run_id": r[14],
                "model_id": r[15],
                "run_label": r[16],
                "bench_id": r[17],
                "bench_name": r[18],
                "model_kind": r[19],
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_eval_repository: Optional[EvalRepository] = None


def get_eval_repository() -> EvalRepository:
    global _eval_repository
    if _eval_repository is None:
        _eval_repository = EvalRepository()
    return _eval_repository
