"""
Entity Repository for relational database storage.

Provides MySQL-backed storage for Characters, VideoClips, and related entities.
Binary files (audio/video) remain in MinIO - only metadata goes to MySQL.

Built on BaseMySQLRepository (shared lazy pool, autocommit=True, table
creation on first connection).
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from virtual_streamer.utils.base_repository import BaseMySQLRepository

logger = logging.getLogger(__name__)


class EntityRepository(BaseMySQLRepository):
    """
    MySQL-backed repository for entity management.

    Handles Characters, VideoClips, VoiceSamples, CharacterPresences,
    Keywords, and Collections with proper relational structure.
    """

    async def _create_tables(self, cur):
        # Characters table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                video_clip_path VARCHAR(512),
                video_search_tag VARCHAR(255),
                identity_images JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add columns if they don't exist (for existing databases)
        try:
            await cur.execute("""
                ALTER TABLE characters
                ADD COLUMN IF NOT EXISTS video_search_tag VARCHAR(255),
                ADD COLUMN IF NOT EXISTS identity_images JSON
            """)
        except Exception:
            pass  # Columns may already exist

        # Voice samples table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS voice_samples (
                id INT AUTO_INCREMENT PRIMARY KEY,
                character_id VARCHAR(255) NOT NULL,
                storage_path VARCHAR(512) NOT NULL,
                transcript TEXT NOT NULL,
                FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
            )
        """)

        # Video clips table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS video_clips (
                id VARCHAR(255) PRIMARY KEY,
                storage_path VARCHAR(512) NOT NULL,
                duration FLOAT,
                scene_description TEXT,
                source_show VARCHAR(255),
                source_episode VARCHAR(255),
                start_time_in_source FLOAT,
                end_time_in_source FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Character presences table (which characters appear in which clips)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS character_presences (
                id INT AUTO_INCREMENT PRIMARY KEY,
                clip_id VARCHAR(255) NOT NULL,
                character_id VARCHAR(255) NOT NULL,
                start_time FLOAT NOT NULL,
                end_time FLOAT NOT NULL,
                FOREIGN KEY (clip_id) REFERENCES video_clips(id) ON DELETE CASCADE,
                FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
            )
        """)

        # Clip keywords table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS clip_keywords (
                id INT AUTO_INCREMENT PRIMARY KEY,
                clip_id VARCHAR(255) NOT NULL,
                keyword VARCHAR(255) NOT NULL,
                FOREIGN KEY (clip_id) REFERENCES video_clips(id) ON DELETE CASCADE
            )
        """)

        # Collections table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT
            )
        """)

        # Clip-Collections join table (many-to-many)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS clip_collections (
                clip_id VARCHAR(255) NOT NULL,
                collection_id VARCHAR(255) NOT NULL,
                PRIMARY KEY (clip_id, collection_id),
                FOREIGN KEY (clip_id) REFERENCES video_clips(id) ON DELETE CASCADE,
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
            )
        """)

        # Story templates table
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS story_templates (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                prompt TEXT NOT NULL,
                collection VARCHAR(255) NOT NULL,
                target_lines INT DEFAULT 6,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Template-Characters join table (many-to-many)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS template_characters (
                template_id VARCHAR(255) NOT NULL,
                character_id VARCHAR(255) NOT NULL,
                PRIMARY KEY (template_id, character_id),
                FOREIGN KEY (template_id) REFERENCES story_templates(id) ON DELETE CASCADE,
                FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
            )
        """)

        # Locations table (environment descriptions scoped per template)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                story_template_id VARCHAR(255) NOT NULL,
                image_path VARCHAR(512),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (story_template_id) REFERENCES story_templates(id) ON DELETE CASCADE
            )
        """)

        await cur.execute("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'locations'
              AND COLUMN_NAME = 'image_path'
        """)
        if (await cur.fetchone())[0] == 0:
            await cur.execute(
                "ALTER TABLE locations ADD COLUMN image_path VARCHAR(512)"
            )

        # Labeled identity images (view labels + captions) on characters
        await cur.execute("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'characters'
              AND COLUMN_NAME = 'labeled_images'
        """)
        if (await cur.fetchone())[0] == 0:
            await cur.execute(
                "ALTER TABLE characters ADD COLUMN labeled_images JSON"
            )

        # Visual details table (reusable props/logos for reference sheets)
        await cur.execute("""
            CREATE TABLE IF NOT EXISTS visual_details (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                category VARCHAR(64) NOT NULL DEFAULT 'Props',
                story_template_id VARCHAR(255) NOT NULL,
                image_path VARCHAR(512),
                label JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (story_template_id) REFERENCES story_templates(id) ON DELETE CASCADE
            )
        """)

    # ==================== CHARACTER METHODS ====================

    async def create_character(
        self,
        character_id: str,
        name: str,
        description: Optional[str] = None,
        video_clip_path: Optional[str] = None,
        voice_samples: Optional[List[Dict[str, str]]] = None,
        video_search_tag: Optional[str] = None,
        identity_images: Optional[List[str]] = None,
        labeled_images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a new character with optional voice samples, search tag, and identity images."""
        await self._execute(
            """
            INSERT INTO characters (id, name, description, video_clip_path, video_search_tag, identity_images, labeled_images, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (character_id, name, description, video_clip_path, video_search_tag,
             json.dumps(identity_images) if identity_images else None,
             json.dumps(labeled_images) if labeled_images else None,
             datetime.utcnow()),
        )

        if voice_samples:
            for sample in voice_samples:
                await self._execute(
                    """
                    INSERT INTO voice_samples (character_id, storage_path, transcript)
                    VALUES (%s, %s, %s)
                    """,
                    (character_id, sample["storage_path"], sample["transcript"]),
                )

        return await self.get_character(character_id)

    async def get_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get a character by ID with its voice samples."""
        row = await self._fetch_one(
            "SELECT id, name, description, video_clip_path, video_search_tag, identity_images, labeled_images, created_at FROM characters WHERE id = %s",
            (character_id,),
        )
        if row is None:
            return None

        def _parse_json_list(value) -> list:
            if not value:
                return []
            try:
                return json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError):
                return []

        character = {
            "character_id": row[0],
            "name": row[1],
            "description": row[2],
            "video_clip_path": row[3] or "",
            "video_search_tag": row[4],
            "identity_images": _parse_json_list(row[5]),
            "labeled_images": _parse_json_list(row[6]),
            "created_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[7].isoformat() if row[7] else None,  # Use created_at as updated_at
        }

        samples = await self._fetch_all(
            "SELECT storage_path, transcript FROM voice_samples WHERE character_id = %s",
            (character_id,),
        )
        character["voice_samples"] = [
            {"sample_storage_path": s[0], "transcript": s[1]} for s in samples
        ]

        return character

    async def list_characters(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all characters with their voice samples."""
        rows = await self._fetch_all(
            "SELECT id FROM characters ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        characters = []
        for row in rows:
            character = await self.get_character(row[0])
            if character:
                characters.append(character)
        return characters

    async def delete_character(self, character_id: str) -> bool:
        """Delete a character (cascade deletes voice samples)."""
        return await self._execute(
            "DELETE FROM characters WHERE id = %s", (character_id,)
        ) > 0

    async def update_character(
        self,
        character_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        video_clip_path: Optional[str] = None,
        video_search_tag: Optional[str] = None,
        identity_images: Optional[List[str]] = None,
        voice_samples: Optional[List[Dict[str, str]]] = None,
        labeled_images: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing character.

        Only provided fields are updated. Pass None to keep existing value.
        For voice_samples, if provided, it REPLACES all existing samples.
        For identity_images, if provided, it REPLACES all existing images.
        """
        updates = []
        values = []

        if name is not None:
            updates.append("name = %s")
            values.append(name)
        if description is not None:
            updates.append("description = %s")
            values.append(description)
        if video_clip_path is not None:
            updates.append("video_clip_path = %s")
            values.append(video_clip_path)
        if video_search_tag is not None:
            updates.append("video_search_tag = %s")
            values.append(video_search_tag)
        if identity_images is not None:
            updates.append("identity_images = %s")
            values.append(json.dumps(identity_images))
        if labeled_images is not None:
            updates.append("labeled_images = %s")
            values.append(json.dumps(labeled_images))

        if updates:
            values.append(character_id)
            await self._execute(
                f"UPDATE characters SET {', '.join(updates)} WHERE id = %s",
                tuple(values),
            )

        # Replace voice samples if provided
        if voice_samples is not None:
            await self._execute(
                "DELETE FROM voice_samples WHERE character_id = %s",
                (character_id,),
            )
            for sample in voice_samples:
                await self._execute(
                    """
                    INSERT INTO voice_samples (character_id, storage_path, transcript)
                    VALUES (%s, %s, %s)
                    """,
                    (character_id, sample["storage_path"], sample["transcript"]),
                )

        return await self.get_character(character_id)

    # ==================== VIDEO CLIP METHODS ====================

    async def create_video_clip(
        self,
        clip_id: str,
        storage_path: str,
        collection_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new video clip."""
        await self._execute(
            """
            INSERT INTO video_clips (id, storage_path, created_at)
            VALUES (%s, %s, %s)
            """,
            (clip_id, storage_path, datetime.utcnow()),
        )

        # Add to collections if specified
        if collection_ids:
            for collection_id in collection_ids:
                await self._execute(
                    """
                    INSERT IGNORE INTO clip_collections (clip_id, collection_id)
                    VALUES (%s, %s)
                    """,
                    (clip_id, collection_id),
                )

        return await self.get_video_clip(clip_id)

    async def get_video_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get a video clip by ID with all related data."""
        row = await self._fetch_one(
            """
            SELECT id, storage_path, duration, scene_description,
                   source_show, source_episode, start_time_in_source,
                   end_time_in_source, created_at
            FROM video_clips WHERE id = %s
            """,
            (clip_id,),
        )
        if row is None:
            return None

        clip = {
            "clip_id": row[0],
            "storage_path": row[1],
            "created_at": row[8].isoformat() if row[8] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
        }

        # Build metadata if any metadata fields are set
        if any([row[2], row[3], row[4], row[5], row[6], row[7]]):
            metadata = {
                "duration": row[2] or 0.0,
                "scene_description_text": row[3],
                "source_show_name": row[4],
                "source_episode_name": row[5],
                "start_time_in_source": row[6],
                "end_time_in_source": row[7],
            }

            keywords = await self._fetch_all(
                "SELECT keyword FROM clip_keywords WHERE clip_id = %s",
                (clip_id,),
            )
            metadata["scene_keywords"] = [k[0] for k in keywords]

            presences = await self._fetch_all(
                """
                SELECT character_id, start_time, end_time
                FROM character_presences WHERE clip_id = %s
                """,
                (clip_id,),
            )
            metadata["character_presences"] = [
                {"character_id": p[0], "start_time": p[1], "end_time": p[2]}
                for p in presences
            ]

            clip["metadata"] = metadata
        else:
            clip["metadata"] = None

        collections = await self._fetch_all(
            "SELECT collection_id FROM clip_collections WHERE clip_id = %s",
            (clip_id,),
        )
        clip["collection_ids"] = [c[0] for c in collections]

        return clip

    async def update_video_clip_metadata(
        self,
        clip_id: str,
        duration: float,
        scene_description_text: Optional[str] = None,
        scene_keywords: Optional[List[str]] = None,
        character_presences: Optional[List[Dict[str, Any]]] = None,
        source_show_name: Optional[str] = None,
        source_episode_name: Optional[str] = None,
        start_time_in_source: Optional[float] = None,
        end_time_in_source: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update metadata for a video clip."""
        await self._execute(
            """
            UPDATE video_clips SET
                duration = %s,
                scene_description = %s,
                source_show = %s,
                source_episode = %s,
                start_time_in_source = %s,
                end_time_in_source = %s
            WHERE id = %s
            """,
            (
                duration,
                scene_description_text,
                source_show_name,
                source_episode_name,
                start_time_in_source,
                end_time_in_source,
                clip_id,
            ),
        )

        # Replace keywords
        await self._execute("DELETE FROM clip_keywords WHERE clip_id = %s", (clip_id,))
        if scene_keywords:
            for keyword in scene_keywords:
                await self._execute(
                    "INSERT INTO clip_keywords (clip_id, keyword) VALUES (%s, %s)",
                    (clip_id, keyword),
                )

        # Replace character presences
        await self._execute(
            "DELETE FROM character_presences WHERE clip_id = %s", (clip_id,)
        )
        if character_presences:
            for presence in character_presences:
                await self._execute(
                    """
                    INSERT INTO character_presences (clip_id, character_id, start_time, end_time)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        clip_id,
                        presence["character_id"],
                        presence["start_time"],
                        presence["end_time"],
                    ),
                )

        return await self.get_video_clip(clip_id)

    async def list_video_clips(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all video clips."""
        rows = await self._fetch_all(
            "SELECT id FROM video_clips ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        clips = []
        for row in rows:
            clip = await self.get_video_clip(row[0])
            if clip:
                clips.append(clip)
        return clips

    async def delete_video_clip(self, clip_id: str) -> bool:
        """Delete a video clip (cascade deletes related data)."""
        return await self._execute(
            "DELETE FROM video_clips WHERE id = %s", (clip_id,)
        ) > 0

    # ==================== STORY TEMPLATE METHODS ====================

    async def create_story_template(
        self,
        template_id: str,
        name: str,
        prompt: str,
        collection: str,
        target_lines: int = 6,
        character_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new story template with associated characters."""
        await self._execute(
            """
            INSERT INTO story_templates (id, name, prompt, collection, target_lines, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (template_id, name, prompt, collection, target_lines, datetime.utcnow()),
        )

        if character_ids:
            for character_id in character_ids:
                await self._execute(
                    """
                    INSERT INTO template_characters (template_id, character_id)
                    VALUES (%s, %s)
                    """,
                    (template_id, character_id),
                )

        return await self.get_story_template(template_id)

    async def get_story_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a story template by ID with its associated characters."""
        row = await self._fetch_one(
            "SELECT id, name, prompt, collection, target_lines, created_at FROM story_templates WHERE id = %s",
            (template_id,),
        )
        if row is None:
            return None

        template = {
            "template_id": row[0],
            "name": row[1],
            "prompt": row[2],
            "collection": row[3],
            "target_lines": row[4],
            "created_at": row[5].isoformat() if row[5] else None,
            "updated_at": row[5].isoformat() if row[5] else None,
        }

        char_rows = await self._fetch_all(
            "SELECT character_id FROM template_characters WHERE template_id = %s",
            (template_id,),
        )
        template["character_ids"] = [c[0] for c in char_rows]

        return template

    async def list_story_templates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all story templates with their character associations."""
        rows = await self._fetch_all(
            "SELECT id FROM story_templates ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        templates = []
        for row in rows:
            template = await self.get_story_template(row[0])
            if template:
                templates.append(template)
        return templates

    async def update_story_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        prompt: Optional[str] = None,
        collection: Optional[str] = None,
        target_lines: Optional[int] = None,
        character_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing story template."""
        updates = []
        values = []
        if name is not None:
            updates.append("name = %s")
            values.append(name)
        if prompt is not None:
            updates.append("prompt = %s")
            values.append(prompt)
        if collection is not None:
            updates.append("collection = %s")
            values.append(collection)
        if target_lines is not None:
            updates.append("target_lines = %s")
            values.append(target_lines)

        if updates:
            values.append(template_id)
            await self._execute(
                f"UPDATE story_templates SET {', '.join(updates)} WHERE id = %s",
                tuple(values),
            )

        # Update character associations if provided
        if character_ids is not None:
            await self._execute(
                "DELETE FROM template_characters WHERE template_id = %s",
                (template_id,),
            )
            for character_id in character_ids:
                await self._execute(
                    """
                    INSERT INTO template_characters (template_id, character_id)
                    VALUES (%s, %s)
                    """,
                    (template_id, character_id),
                )

        return await self.get_story_template(template_id)

    async def delete_story_template(self, template_id: str) -> bool:
        """Delete a story template (cascade deletes character associations)."""
        return await self._execute(
            "DELETE FROM story_templates WHERE id = %s", (template_id,)
        ) > 0

    async def drop_story_template_tables(self) -> bool:
        """
        Drop story template tables for schema reset.

        Drops both story_templates and template_characters tables,
        allowing them to be recreated with the latest schema.

        Returns:
            True if tables were dropped successfully
        """
        await self._execute("DROP TABLE IF EXISTS template_characters")
        await self._execute("DROP TABLE IF EXISTS story_templates")
        logger.info("Dropped story_templates and template_characters tables")

        # Recreate with current schema
        await self._execute("""
            CREATE TABLE IF NOT EXISTS story_templates (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                prompt TEXT NOT NULL,
                collection VARCHAR(255) NOT NULL,
                target_lines INT DEFAULT 6,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._execute("""
            CREATE TABLE IF NOT EXISTS template_characters (
                template_id VARCHAR(255) NOT NULL,
                character_id VARCHAR(255) NOT NULL,
                PRIMARY KEY (template_id, character_id),
                FOREIGN KEY (template_id) REFERENCES story_templates(id) ON DELETE CASCADE,
                FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
            )
        """)
        logger.info("Recreated story_templates and template_characters tables with latest schema")

        return True

    # ==================== LOCATION METHODS ====================

    _LOCATION_COLS = "id, name, description, story_template_id, image_path, created_at"

    @staticmethod
    def _location_row_to_dict(row) -> Dict[str, Any]:
        return {
            "location_id": row[0],
            "name": row[1],
            "description": row[2],
            "story_template_id": row[3],
            "image_path": row[4],
            "created_at": row[5],
            "updated_at": row[5],
        }

    async def create_location(
        self,
        location_id: str,
        name: str,
        description: str,
        story_template_id: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new location scoped to a story template."""
        await self._execute(
            """
            INSERT INTO locations (id, name, description, story_template_id, image_path, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (location_id, name, description, story_template_id, image_path, datetime.utcnow()),
        )
        return await self.get_location(location_id)

    async def update_location_image(
        self,
        location_id: str,
        image_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Update the identity image path for an existing location."""
        await self._execute(
            "UPDATE locations SET image_path = %s WHERE id = %s",
            (image_path, location_id),
        )
        return await self.get_location(location_id)

    async def get_location(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Get a location by ID."""
        row = await self._fetch_one(
            f"SELECT {self._LOCATION_COLS} FROM locations WHERE id = %s",
            (location_id,),
        )
        return self._location_row_to_dict(row) if row else None

    async def list_locations_by_template(
        self, template_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List all locations for a given story template."""
        rows = await self._fetch_all(
            f"""
            SELECT {self._LOCATION_COLS}
            FROM locations
            WHERE story_template_id = %s
            ORDER BY name
            LIMIT %s
            """,
            (template_id, limit),
        )
        return [self._location_row_to_dict(r) for r in rows]

    async def list_all_locations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all locations across all templates."""
        rows = await self._fetch_all(
            f"""
            SELECT {self._LOCATION_COLS}
            FROM locations
            ORDER BY story_template_id, name
            LIMIT %s
            """,
            (limit,),
        )
        return [self._location_row_to_dict(r) for r in rows]

    async def delete_location(self, location_id: str) -> bool:
        """Delete a location by ID."""
        return await self._execute(
            "DELETE FROM locations WHERE id = %s", (location_id,)
        ) > 0

    # ==================== VISUAL DETAIL METHODS ====================

    _VISUAL_DETAIL_COLS = "id, name, description, category, story_template_id, image_path, label, created_at"

    @staticmethod
    def _visual_detail_row_to_dict(row) -> Dict[str, Any]:
        label = None
        if row[6]:
            try:
                label = json.loads(row[6]) if isinstance(row[6], str) else row[6]
            except (json.JSONDecodeError, TypeError):
                label = None
        return {
            "detail_id": row[0],
            "name": row[1],
            "description": row[2],
            "category": row[3],
            "story_template_id": row[4],
            "image_path": row[5],
            "label": label,
            "created_at": row[7],
            "updated_at": row[7],
        }

    async def create_visual_detail(
        self,
        detail_id: str,
        name: str,
        description: str,
        story_template_id: str,
        category: str = "Props",
        image_path: Optional[str] = None,
        label: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new visual detail scoped to a story template."""
        await self._execute(
            """
            INSERT INTO visual_details (id, name, description, category, story_template_id, image_path, label, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (detail_id, name, description, category, story_template_id,
             image_path, json.dumps(label) if label else None, datetime.utcnow()),
        )
        return await self.get_visual_detail(detail_id)

    async def update_visual_detail_image(
        self,
        detail_id: str,
        image_path: str,
        label: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update the image path (and optional label) of a visual detail."""
        await self._execute(
            "UPDATE visual_details SET image_path = %s, label = %s WHERE id = %s",
            (image_path, json.dumps(label) if label else None, detail_id),
        )
        return await self.get_visual_detail(detail_id)

    async def get_visual_detail(self, detail_id: str) -> Optional[Dict[str, Any]]:
        """Get a visual detail by ID."""
        row = await self._fetch_one(
            f"SELECT {self._VISUAL_DETAIL_COLS} FROM visual_details WHERE id = %s",
            (detail_id,),
        )
        return self._visual_detail_row_to_dict(row) if row else None

    async def list_visual_details_by_template(
        self, template_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List all visual details for a given story template."""
        rows = await self._fetch_all(
            f"""
            SELECT {self._VISUAL_DETAIL_COLS}
            FROM visual_details
            WHERE story_template_id = %s
            ORDER BY name
            LIMIT %s
            """,
            (template_id, limit),
        )
        return [self._visual_detail_row_to_dict(r) for r in rows]

    async def delete_visual_detail(self, detail_id: str) -> bool:
        """Delete a visual detail by ID."""
        return await self._execute(
            "DELETE FROM visual_details WHERE id = %s", (detail_id,)
        ) > 0


# Global instance
_entity_repository: Optional[EntityRepository] = None


def get_entity_repository() -> EntityRepository:
    """Get or create the global EntityRepository instance."""
    global _entity_repository
    if _entity_repository is None:
        _entity_repository = EntityRepository()
    return _entity_repository
