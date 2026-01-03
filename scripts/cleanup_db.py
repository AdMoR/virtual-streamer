#!/usr/bin/env python3
"""
Database Cleanup Utility for Virtual Streamer.

Purges all tables, schema, or the entire database used by the application.
Useful for development, testing, or resetting to a clean state.

Usage:
    # Truncate all tables (keep schema)
    python scripts/cleanup_db.py --truncate

    # Drop all tables (keep database)
    python scripts/cleanup_db.py --drop-tables

    # Drop entire database
    python scripts/cleanup_db.py --drop-database

    # Force without confirmation prompt
    python scripts/cleanup_db.py --drop-tables --force

    # Use custom connection settings
    python scripts/cleanup_db.py --drop-tables \
        --host localhost --port 3306 \
        --user virtual_streamer --password secret \
        --database virtual_streamer
"""

import argparse
import asyncio
import os
import sys

import aiomysql


# Tables in dependency order (children first for proper deletion)
TABLES_IN_ORDER = [
    "clip_collections",      # M:N join table
    "clip_keywords",         # FK to video_clips
    "character_presences",   # FK to video_clips and characters
    "voice_samples",         # FK to characters
    "video_clips",           # Referenced by clip_keywords, character_presences, clip_collections
    "characters",            # Referenced by voice_samples, character_presences
    "collections",           # Referenced by clip_collections
    "jobs",                  # Standalone job tracking table
]


async def get_connection(host: str, port: int, user: str, password: str, database: str = None):
    """Create a database connection."""
    kwargs = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "autocommit": True,
    }
    if database:
        kwargs["db"] = database
    return await aiomysql.connect(**kwargs)


async def get_existing_tables(conn, database: str) -> list[str]:
    """Get list of existing tables in the database."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (database,)
        )
        rows = await cur.fetchall()
        return [row[0] for row in rows]


async def truncate_tables(
    host: str, port: int, user: str, password: str, database: str
) -> dict:
    """Truncate all tables (delete data, keep schema)."""
    results = {"truncated": [], "skipped": [], "errors": []}
    
    try:
        conn = await get_connection(host, port, user, password, database)
    except Exception as e:
        return {"error": f"Connection failed: {e}"}
    
    try:
        existing = await get_existing_tables(conn, database)
        
        async with conn.cursor() as cur:
            # Disable foreign key checks for truncation
            await cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            
            for table in TABLES_IN_ORDER:
                if table in existing:
                    try:
                        await cur.execute(f"TRUNCATE TABLE `{table}`")
                        results["truncated"].append(table)
                        print(f"  ✓ Truncated: {table}")
                    except Exception as e:
                        results["errors"].append({"table": table, "error": str(e)})
                        print(f"  ✗ Error truncating {table}: {e}")
                else:
                    results["skipped"].append(table)
                    print(f"  - Skipped (not found): {table}")
            
            # Re-enable foreign key checks
            await cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    finally:
        conn.close()
    
    return results


async def drop_tables(
    host: str, port: int, user: str, password: str, database: str
) -> dict:
    """Drop all tables (remove schema, keep database)."""
    results = {"dropped": [], "skipped": [], "errors": []}
    
    try:
        conn = await get_connection(host, port, user, password, database)
    except Exception as e:
        return {"error": f"Connection failed: {e}"}
    
    try:
        existing = await get_existing_tables(conn, database)
        
        async with conn.cursor() as cur:
            # Disable foreign key checks for dropping
            await cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            
            for table in TABLES_IN_ORDER:
                if table in existing:
                    try:
                        await cur.execute(f"DROP TABLE IF EXISTS `{table}`")
                        results["dropped"].append(table)
                        print(f"  ✓ Dropped: {table}")
                    except Exception as e:
                        results["errors"].append({"table": table, "error": str(e)})
                        print(f"  ✗ Error dropping {table}: {e}")
                else:
                    results["skipped"].append(table)
                    print(f"  - Skipped (not found): {table}")
            
            # Drop any additional tables not in our list
            remaining = await get_existing_tables(conn, database)
            for table in remaining:
                try:
                    await cur.execute(f"DROP TABLE IF EXISTS `{table}`")
                    results["dropped"].append(table)
                    print(f"  ✓ Dropped (extra): {table}")
                except Exception as e:
                    results["errors"].append({"table": table, "error": str(e)})
                    print(f"  ✗ Error dropping {table}: {e}")
            
            # Re-enable foreign key checks
            await cur.execute("SET FOREIGN_KEY_CHECKS = 1")
    finally:
        conn.close()
    
    return results


async def drop_database(
    host: str, port: int, user: str, password: str, database: str
) -> dict:
    """Drop the entire database."""
    results = {"dropped_database": None, "error": None}
    
    try:
        # Connect without specifying database
        conn = await get_connection(host, port, user, password, database=None)
    except Exception as e:
        return {"error": f"Connection failed: {e}"}
    
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{database}`")
            results["dropped_database"] = database
            print(f"  ✓ Dropped database: {database}")
    except Exception as e:
        results["error"] = str(e)
        print(f"  ✗ Error dropping database: {e}")
    finally:
        conn.close()
    
    return results


async def show_stats(
    host: str, port: int, user: str, password: str, database: str
) -> dict:
    """Show current database statistics."""
    stats = {"tables": {}, "total_rows": 0}
    
    try:
        conn = await get_connection(host, port, user, password, database)
    except Exception as e:
        return {"error": f"Connection failed: {e}"}
    
    try:
        existing = await get_existing_tables(conn, database)
        
        async with conn.cursor() as cur:
            for table in existing:
                try:
                    await cur.execute(f"SELECT COUNT(*) FROM `{table}`")
                    row = await cur.fetchone()
                    count = row[0] if row else 0
                    stats["tables"][table] = count
                    stats["total_rows"] += count
                except Exception:
                    stats["tables"][table] = "error"
    finally:
        conn.close()
    
    return stats


def confirm_action(action: str, database: str) -> bool:
    """Prompt user for confirmation."""
    print(f"\n⚠️  WARNING: This will {action} in database '{database}'")
    print("   This action cannot be undone!\n")
    response = input("Type 'yes' to confirm: ")
    return response.lower() == "yes"


async def main():
    parser = argparse.ArgumentParser(
        description="Database cleanup utility for Virtual Streamer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Actions (choose one):
  --truncate       Delete all data but keep table schema
  --drop-tables    Drop all tables but keep database
  --drop-database  Drop the entire database
  --stats          Show current database statistics (no changes)

Examples:
  # View current state
  python scripts/cleanup_db.py --stats

  # Clear all data (keep schema)
  python scripts/cleanup_db.py --truncate

  # Remove all tables
  python scripts/cleanup_db.py --drop-tables --force

  # Complete reset
  python scripts/cleanup_db.py --drop-database --force
        """,
    )
    
    # Action group (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate all tables (delete data, keep schema)",
    )
    action_group.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop all tables (remove schema, keep database)",
    )
    action_group.add_argument(
        "--drop-database",
        action="store_true",
        help="Drop the entire database",
    )
    action_group.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics without making changes",
    )
    
    # Connection options
    parser.add_argument(
        "--host",
        default=os.environ.get("MYSQL_HOST", "localhost"),
        help="MySQL host (default: MYSQL_HOST env or localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MYSQL_PORT", "3306")),
        help="MySQL port (default: MYSQL_PORT env or 3306)",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("MYSQL_USER", "virtual_streamer"),
        help="MySQL user (default: MYSQL_USER env or virtual_streamer)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("MYSQL_PASSWORD", ""),
        help="MySQL password (default: MYSQL_PASSWORD env)",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("MYSQL_DATABASE", "virtual_streamer"),
        help="Database name (default: MYSQL_DATABASE env or virtual_streamer)",
    )
    
    # Safety options
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Virtual Streamer Database Cleanup")
    print(f"{'='*60}")
    print(f"\nConnection: {args.user}@{args.host}:{args.port}/{args.database}")
    
    # Handle --stats (no destructive action)
    if args.stats:
        print("\nFetching database statistics...")
        stats = await show_stats(
            args.host, args.port, args.user, args.password, args.database
        )
        
        if "error" in stats:
            print(f"\n✗ Error: {stats['error']}")
            sys.exit(1)
        
        print(f"\n{'Table':<25} {'Rows':>10}")
        print("-" * 36)
        for table, count in sorted(stats["tables"].items()):
            print(f"{table:<25} {count:>10}")
        print("-" * 36)
        print(f"{'Total':<25} {stats['total_rows']:>10}")
        sys.exit(0)
    
    # Determine action description
    if args.truncate:
        action_desc = "truncate all tables (delete all data)"
    elif args.drop_tables:
        action_desc = "drop all tables (remove schema)"
    else:
        action_desc = "DROP THE ENTIRE DATABASE"
    
    # Confirm unless --force
    if not args.force:
        if not confirm_action(action_desc, args.database):
            print("\nAborted.")
            sys.exit(0)
    
    print(f"\nExecuting: {action_desc}")
    print("-" * 40)
    
    # Execute the action
    if args.truncate:
        results = await truncate_tables(
            args.host, args.port, args.user, args.password, args.database
        )
        if "error" in results:
            print(f"\n✗ {results['error']}")
            sys.exit(1)
        print(f"\n✓ Truncated {len(results['truncated'])} table(s)")
        
    elif args.drop_tables:
        results = await drop_tables(
            args.host, args.port, args.user, args.password, args.database
        )
        if "error" in results:
            print(f"\n✗ {results['error']}")
            sys.exit(1)
        print(f"\n✓ Dropped {len(results['dropped'])} table(s)")
        
    elif args.drop_database:
        results = await drop_database(
            args.host, args.port, args.user, args.password, args.database
        )
        if results.get("error"):
            print(f"\n✗ {results['error']}")
            sys.exit(1)
        print(f"\n✓ Database '{args.database}' dropped successfully")
    
    print(f"\n{'='*60}")
    print("Cleanup complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())

