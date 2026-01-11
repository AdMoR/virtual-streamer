#!/usr/bin/env python3
"""
Setup script to create streaming-related database tables.

Usage:
    python scripts/setup_streaming_tables.py
    
    # Or with custom connection
    MYSQL_HOST=localhost MYSQL_PORT=3306 python scripts/setup_streaming_tables.py
    
    # Drop and recreate tables
    python scripts/setup_streaming_tables.py --drop
"""
import asyncio
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiomysql


TABLES_SQL = """
-- Stream configurations
CREATE TABLE IF NOT EXISTS stream_configs (
    stream_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Media programmations (time-slot scheduling)
CREATE TABLE IF NOT EXISTS media_programmations (
    programmation_id VARCHAR(36) PRIMARY KEY,
    stream_id VARCHAR(64) NOT NULL,
    story_template_id VARCHAR(36) NOT NULL,
    name VARCHAR(255) NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    priority INT DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_stream_active (stream_id, is_active),
    INDEX idx_stream_time (stream_id, start_time, end_time),
    FOREIGN KEY (stream_id) REFERENCES stream_configs(stream_id) ON DELETE CASCADE
);

-- Playlist entries
CREATE TABLE IF NOT EXISTS playlist_entries (
    entry_id VARCHAR(36) PRIMARY KEY,
    programmation_id VARCHAR(36) NOT NULL,
    video_storage_key VARCHAR(512) NOT NULL,
    status ENUM('pending', 'playing', 'played', 'skipped') DEFAULT 'pending',
    play_order INT DEFAULT 0,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    played_at TIMESTAMP NULL,
    INDEX idx_prog_status (programmation_id, status),
    INDEX idx_prog_order (programmation_id, play_order),
    FOREIGN KEY (programmation_id) REFERENCES media_programmations(programmation_id) ON DELETE CASCADE
);
"""

DROP_TABLES_SQL = """
DROP TABLE IF EXISTS playlist_entries;
DROP TABLE IF EXISTS media_programmations;
DROP TABLE IF EXISTS stream_configs;
"""


async def get_connection():
    """Create database connection."""
    return await aiomysql.connect(
        host=os.environ.get("MYSQL_HOST", "localhost"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "virtual_streamer"),
        password=os.environ.get("MYSQL_PASSWORD", "streamerpass"),
        db=os.environ.get("MYSQL_DATABASE", "virtual_streamer"),
        autocommit=True,
    )


async def drop_tables():
    """Drop all streaming tables."""
    print("Dropping existing streaming tables...")
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            for statement in DROP_TABLES_SQL.strip().split(';'):
                statement = statement.strip()
                if statement:
                    await cur.execute(statement)
                    print(f"  Dropped: {statement}")
        print("All streaming tables dropped.")
    finally:
        conn.close()


async def setup_tables():
    """Create streaming tables."""
    print("Creating streaming tables...")
    conn = await get_connection()
    try:
        async with conn.cursor() as cur:
            for statement in TABLES_SQL.strip().split(';'):
                statement = statement.strip()
                if statement:
                    await cur.execute(statement)
                    # Extract table name for logging
                    if "CREATE TABLE" in statement.upper():
                        table_name = statement.split("(")[0].split()[-1].strip('`')
                        print(f"  Created: {table_name}")
        print("All streaming tables created successfully!")
    finally:
        conn.close()


async def main(drop: bool = False):
    """Main function."""
    if drop:
        await drop_tables()
    await setup_tables()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup streaming database tables")
    parser.add_argument(
        "--drop", 
        action="store_true", 
        help="Drop existing tables before creating"
    )
    args = parser.parse_args()
    
    asyncio.run(main(drop=args.drop))
