"""
Low-level API: Database Browser

Simple endpoints to browse MySQL tables for debugging and administration.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
import aiomysql

router = APIRouter(prefix="/db", tags=["Database Browser"])


async def get_connection():
    """Create a MySQL connection using environment variables."""
    return await aiomysql.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "virtual_streamer"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        db=os.environ.get("MYSQL_DATABASE", "virtual_streamer"),
        autocommit=True,
    )


@router.get("/tables", response_model=List[str])
async def list_tables():
    """List all tables in the database."""
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SHOW TABLES")
            tables = await cur.fetchall()
            return [t[0] for t in tables]
    finally:
        conn.close()


@router.get("/tables/{table_name}/schema")
async def get_table_schema(table_name: str) -> List[Dict[str, Any]]:
    """Get the schema (columns) of a table."""
    conn = await get_connection()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            # Validate table exists to prevent SQL injection
            await cur.execute("SHOW TABLES LIKE %s", (table_name,))
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
            
            await cur.execute(f"DESCRIBE `{table_name}`")
            columns = await cur.fetchall()
            return [dict(col) for col in columns]
    finally:
        conn.close()


@router.get("/tables/{table_name}/rows")
async def get_table_rows(
    table_name: str,
    limit: int = Query(default=50, le=1000),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    """Get rows from a table with pagination."""
    conn = await get_connection()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            # Validate table exists to prevent SQL injection
            await cur.execute("SHOW TABLES LIKE %s", (table_name,))
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
            
            # Get total count
            await cur.execute(f"SELECT COUNT(*) as cnt FROM `{table_name}`")
            count_result = await cur.fetchone()
            total = count_result["cnt"]
            
            # Get rows
            await cur.execute(f"SELECT * FROM `{table_name}` LIMIT %s OFFSET %s", (limit, offset))
            rows = await cur.fetchall()
            
            # Convert datetime objects to strings for JSON serialization
            serialized_rows = []
            for row in rows:
                serialized_row = {}
                for key, value in row.items():
                    if hasattr(value, 'isoformat'):
                        serialized_row[key] = value.isoformat()
                    elif isinstance(value, bytes):
                        serialized_row[key] = value.decode('utf-8', errors='replace')
                    else:
                        serialized_row[key] = value
                serialized_rows.append(serialized_row)
            
            return {
                "table": table_name,
                "total": total,
                "limit": limit,
                "offset": offset,
                "rows": serialized_rows,
            }
    finally:
        conn.close()
