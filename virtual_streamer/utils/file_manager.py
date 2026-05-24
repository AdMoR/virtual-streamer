"""
Centralized File Management for Virtual Streamer

This module provides a centralized approach to managing temporary and output files
across all services, with automatic cleanup and consistent naming conventions.
"""

import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


class FileCategory(str, Enum):
    """Categories for organizing files"""

    AUDIO = "audio"
    VIDEO = "video"
    SUBTITLE = "subtitle"
    CONCAT = "concat"
    TEMP = "temp"
    OUTPUT = "output"


@dataclass
class FilePathConfig:
    """
    Centralized configuration for all file paths.

    All paths can be configured via environment variables or direct initialization.
    Subdirectories are automatically created within temp_dir if not specified.
    """

    # Base directories
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "/data"))
    temp_dir: str = field(
        default_factory=lambda: os.getenv("TEMP_DIR", "/tmp/virtual_streamer")
    )
    output_dir: str = field(
        default_factory=lambda: os.getenv("OUT_VIDEO_FOLDER", "./out_video_folder")
    )

    # Subdirectories for organization (optional - defaults to {temp_dir}/{category})
    audio_temp_dir: Optional[str] = None
    video_temp_dir: Optional[str] = None
    subtitle_temp_dir: Optional[str] = None
    concat_temp_dir: Optional[str] = None

    # Cleanup settings
    auto_cleanup: bool = field(
        default_factory=lambda: os.getenv("AUTO_CLEANUP_TEMP", "true").lower() == "true"
    )
    cleanup_age_hours: int = field(
        default_factory=lambda: int(os.getenv("CLEANUP_AGE_HOURS", "24"))
    )
    keep_artifacts: bool = field(
        default_factory=lambda: os.getenv("KEEP_ARTIFACTS", "false").lower() == "true"
    )
    max_temp_size_gb: int = field(
        default_factory=lambda: int(os.getenv("MAX_TEMP_SIZE_GB", "10"))
    )

    # File naming conventions
    use_timestamps: bool = field(
        default_factory=lambda: os.getenv("USE_TIMESTAMPS", "true").lower() == "true"
    )
    use_uuids: bool = field(
        default_factory=lambda: os.getenv("USE_UUIDS", "true").lower() == "true"
    )

    def __post_init__(self):
        """Initialize subdirectories with defaults if not specified"""
        if self.audio_temp_dir is None:
            self.audio_temp_dir = os.path.join(self.temp_dir, "audio")
        if self.video_temp_dir is None:
            self.video_temp_dir = os.path.join(self.temp_dir, "video")
        if self.subtitle_temp_dir is None:
            self.subtitle_temp_dir = os.path.join(self.temp_dir, "subtitles")
        if self.concat_temp_dir is None:
            self.concat_temp_dir = os.path.join(self.temp_dir, "concat")

    def get_category_dir(self, category: FileCategory, temp: bool = True) -> str:
        """Get directory path for a specific category"""
        if not temp:
            return self.output_dir

        category_map = {
            FileCategory.AUDIO: self.audio_temp_dir,
            FileCategory.VIDEO: self.video_temp_dir,
            FileCategory.SUBTITLE: self.subtitle_temp_dir,
            FileCategory.CONCAT: self.concat_temp_dir,
            FileCategory.TEMP: self.temp_dir,
            FileCategory.OUTPUT: self.output_dir,
        }
        return category_map.get(category, self.temp_dir)


# ============================================================================
# File Naming Strategy
# ============================================================================


class FileNamingStrategy:
    """
    Provides consistent file naming conventions across the application.

    Supports configurable patterns with timestamps, UUIDs, and descriptive names.
    """

    def __init__(self, config: FilePathConfig):
        self.config = config

    def _add_timestamp(self) -> str:
        """Generate timestamp string for filenames"""
        if self.config.use_timestamps:
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        return ""

    def _add_uuid(self) -> str:
        """Generate UUID for filenames"""
        if self.config.use_uuids:
            return str(uuid.uuid4())[:8]
        return ""

    def generate_filename(
        self,
        prefix: str,
        extension: str,
        suffix: Optional[str] = None,
        include_timestamp: Optional[bool] = None,
        include_uuid: Optional[bool] = None,
    ) -> str:
        """
        Generate a standardized filename.

        Args:
            prefix: Base name (e.g., "tts", "combined")
            extension: File extension without dot (e.g., "wav", "mp4")
            suffix: Optional suffix to add
            include_timestamp: Override config.use_timestamps
            include_uuid: Override config.use_uuids

        Returns:
            Formatted filename

        Example:
            generate_filename("tts", "wav", "char1_entry123")
            -> "tts_char1_entry123_20251112_143022_a1b2c3d4.wav"
        """
        parts = [prefix]

        if suffix:
            parts.append(suffix)

        # Add timestamp if enabled
        use_ts = (
            include_timestamp
            if include_timestamp is not None
            else self.config.use_timestamps
        )
        if use_ts:
            timestamp = self._add_timestamp()
            if timestamp:
                parts.append(timestamp)

        # Add UUID if enabled
        use_id = include_uuid if include_uuid is not None else self.config.use_uuids
        if use_id:
            uuid_str = self._add_uuid()
            if uuid_str:
                parts.append(uuid_str)

        # Ensure extension doesn't have leading dot
        extension = extension.lstrip(".")

        filename = "_".join(parts) + f".{extension}"
        return filename

    def tts_filename(self, character_id: str, entry_id: str) -> str:
        """Generate TTS audio filename"""
        return self.generate_filename("tts", "wav", f"{character_id}_{entry_id}")

    def subtitle_filename(self, source: str, format: str = "srt") -> str:
        """Generate subtitle filename"""
        return self.generate_filename("subtitle", format, source)

    def combined_filename(self, request_id: Optional[str] = None) -> str:
        """Generate combined video filename"""
        suffix = request_id if request_id else None
        return self.generate_filename("combined", "mp4", suffix)

    def final_output_filename(self, title: str) -> str:
        """Generate final output video filename"""
        # Sanitize title for filename
        safe_title = "".join(
            c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title
        )
        safe_title = safe_title.replace(" ", "_").replace(",", "_").lower()[:50]  # Max 50 chars
        return self.generate_filename("video", "mp4", safe_title)

    def temp_dir_name(self, category: str) -> str:
        """Generate temporary directory name"""
        parts = [category]
        if self.config.use_uuids:
            parts.append(self._add_uuid())
        return "_".join(parts)


# ============================================================================
# File Tracker
# ============================================================================


@dataclass
class FileTracker:
    """
    Track created files for lifecycle management.

    Used to monitor file creation, usage, and cleanup.
    """

    filepath: str
    category: FileCategory
    created_at: datetime = field(default_factory=datetime.now)
    purpose: str = ""
    parent_request_id: Optional[str] = None
    should_cleanup: bool = True
    max_age_hours: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def is_expired(self, max_age_hours: Optional[int] = None) -> bool:
        """Check if file has exceeded its maximum age"""
        age_limit = max_age_hours if max_age_hours is not None else self.max_age_hours
        if age_limit is None:
            return False

        age = datetime.now() - self.created_at
        return age > timedelta(hours=age_limit)

    def exists(self) -> bool:
        """Check if the file still exists"""
        return os.path.exists(self.filepath)

    def get_size_mb(self) -> float:
        """Get file size in MB"""
        if self.exists():
            return os.path.getsize(self.filepath) / (1024 * 1024)
        return 0.0


# ============================================================================
# File Manager
# ============================================================================


class FileManager:
    """
    Centralized file management for temporary and output artifacts.

    Provides:
    - Consistent path resolution
    - Automatic directory creation
    - File tracking for cleanup
    - Cleanup scheduling

    Example:
        >>> file_mgr = FileManager()
        >>> audio_path = file_mgr.get_temp_path("output.wav", FileCategory.AUDIO)
        >>> file_mgr.register_file(audio_path, FileCategory.AUDIO, max_age_hours=1)
        >>> file_mgr.cleanup_expired_files()
    """

    def __init__(self, config: Optional[FilePathConfig] = None):
        """
        Initialize FileManager.

        Args:
            config: Optional FilePathConfig. If None, uses defaults from environment.
        """
        self.config = config or FilePathConfig()
        self.naming = FileNamingStrategy(self.config)
        self.tracked_files: List[FileTracker] = []

        # Ensure base directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create all configured directories if they don't exist"""
        directories = [
            self.config.data_dir,
            self.config.temp_dir,
            self.config.output_dir,
            self.config.audio_temp_dir,
            self.config.video_temp_dir,
            self.config.subtitle_temp_dir,
            self.config.concat_temp_dir,
        ]

        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")

    def get_temp_path(
        self,
        filename: str,
        category: FileCategory = FileCategory.TEMP,
        create_dir: bool = False,
    ) -> str:
        """
        Get full path for a temporary file.

        Args:
            filename: Name of the file
            category: File category for organization
            create_dir: If True and filename contains dirs, create them

        Returns:
            Full absolute path to the file
        """
        base_dir = self.config.get_category_dir(category, temp=True)
        full_path = os.path.join(base_dir, filename)

        # Create subdirectories if needed
        if create_dir:
            parent_dir = os.path.dirname(full_path)
            Path(parent_dir).mkdir(parents=True, exist_ok=True)

        return os.path.abspath(full_path)

    def get_output_path(self, filename: str) -> str:
        """
        Get full path for an output file.

        Args:
            filename: Name of the output file

        Returns:
            Full absolute path to the output file
        """
        full_path = os.path.join(self.config.output_dir, filename)
        return os.path.abspath(full_path)

    def register_file(
        self,
        filepath: str,
        category: FileCategory,
        purpose: str = "",
        parent_request_id: Optional[str] = None,
        should_cleanup: bool = True,
        max_age_hours: Optional[int] = None,
        **metadata,
    ) -> FileTracker:
        """
        Register a file for tracking and cleanup.

        Args:
            filepath: Full path to the file
            category: File category
            purpose: Description of file purpose
            parent_request_id: ID of parent request/job
            should_cleanup: Whether file should be auto-cleaned
            max_age_hours: Max age before cleanup (overrides config)
            **metadata: Additional metadata to store

        Returns:
            FileTracker instance
        """
        tracker = FileTracker(
            filepath=filepath,
            category=category,
            purpose=purpose,
            parent_request_id=parent_request_id,
            should_cleanup=should_cleanup,
            max_age_hours=max_age_hours or self.config.cleanup_age_hours,
            metadata=metadata,
        )

        self.tracked_files.append(tracker)
        logger.debug(f"Registered file: {filepath} (category={category})")

        return tracker

    def cleanup_expired_files(self, max_age_hours: Optional[int] = None) -> List[str]:
        """
        Clean up expired temporary files.

        Args:
            max_age_hours: Override default cleanup age

        Returns:
            List of paths that were deleted
        """
        if not self.config.auto_cleanup:
            logger.debug("Auto-cleanup is disabled")
            return []

        deleted_files = []
        age_limit = max_age_hours or self.config.cleanup_age_hours

        for tracker in self.tracked_files[
            :
        ]:  # Copy list to allow removal during iteration
            if not tracker.should_cleanup:
                continue

            if tracker.is_expired(age_limit) and tracker.exists():
                try:
                    os.remove(tracker.filepath)
                    deleted_files.append(tracker.filepath)
                    self.tracked_files.remove(tracker)
                    logger.info(f"Cleaned up expired file: {tracker.filepath}")
                except Exception as e:
                    logger.error(f"Failed to delete {tracker.filepath}: {e}")

        return deleted_files

    def cleanup_directory(
        self, directory: str, max_age_hours: Optional[int] = None
    ) -> List[str]:
        """
        Clean up old files in a specific directory.

        Args:
            directory: Path to directory to clean
            max_age_hours: Files older than this will be deleted

        Returns:
            List of deleted file paths
        """
        if not os.path.exists(directory):
            return []

        age_limit = max_age_hours or self.config.cleanup_age_hours
        cutoff_time = datetime.now() - timedelta(hours=age_limit)
        deleted_files = []

        for root, dirs, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if mtime < cutoff_time:
                        os.remove(filepath)
                        deleted_files.append(filepath)
                        logger.info(f"Cleaned up old file: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to clean {filepath}: {e}")

        # Clean up empty directories
        for root, dirs, files in os.walk(directory, topdown=False):
            for dirname in dirs:
                dirpath = os.path.join(root, dirname)
                try:
                    if not os.listdir(dirpath):  # Empty directory
                        os.rmdir(dirpath)
                        logger.debug(f"Removed empty directory: {dirpath}")
                except Exception as e:
                    logger.error(f"Failed to remove directory {dirpath}: {e}")

        return deleted_files

    def cleanup_all_temp(self) -> Dict[str, List[str]]:
        """
        Clean up all temporary directories.

        Returns:
            Dictionary mapping category to list of deleted files
        """
        results = {}

        categories = [
            ("audio", self.config.audio_temp_dir),
            ("video", self.config.video_temp_dir),
            ("subtitle", self.config.subtitle_temp_dir),
            ("concat", self.config.concat_temp_dir),
        ]

        for category_name, directory in categories:
            if directory and os.path.exists(directory):
                deleted = self.cleanup_directory(directory)
                results[category_name] = deleted
                logger.info(f"Cleaned {len(deleted)} files from {category_name}")

        return results

    def get_temp_size(self) -> Dict[str, float]:
        """
        Get size of temporary directories in MB.

        Returns:
            Dictionary mapping directory name to size in MB
        """

        def get_dir_size(directory: str) -> float:
            if not os.path.exists(directory):
                return 0.0
            total = 0
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    try:
                        total += os.path.getsize(filepath)
                    except:
                        pass
            return total / (1024 * 1024)  # Convert to MB

        return {
            "audio": get_dir_size(self.config.audio_temp_dir),
            "video": get_dir_size(self.config.video_temp_dir),
            "subtitle": get_dir_size(self.config.subtitle_temp_dir),
            "concat": get_dir_size(self.config.concat_temp_dir),
            "total": get_dir_size(self.config.temp_dir),
        }

    def get_stats(self) -> Dict:
        """Get statistics about tracked files and temp directories"""
        sizes = self.get_temp_size()

        return {
            "tracked_files": len(self.tracked_files),
            "temp_sizes_mb": sizes,
            "total_temp_mb": sizes["total"],
            "max_temp_gb": self.config.max_temp_size_gb,
            "auto_cleanup": self.config.auto_cleanup,
            "cleanup_age_hours": self.config.cleanup_age_hours,
        }


# ============================================================================
# Global Instance
# ============================================================================

_file_manager_instance: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """
    Get or create the global FileManager instance.

    Returns:
        FileManager instance
    """
    global _file_manager_instance

    if _file_manager_instance is None:
        _file_manager_instance = FileManager()

    return _file_manager_instance


def reset_file_manager():
    """Reset the global FileManager instance (useful for testing)"""
    global _file_manager_instance
    _file_manager_instance = None


# ============================================================================
# Cleanup Scheduler
# ============================================================================


class CleanupScheduler:
    """
    Background scheduler for automatic cleanup of temporary files.

    Runs cleanup tasks at specified intervals to prevent disk space issues.
    Integrates with FastAPI lifecycle events.

    Example:
        >>> scheduler = CleanupScheduler(file_manager, interval_minutes=60)
        >>> await scheduler.start()
        >>> # ... later ...
        >>> await scheduler.stop()
    """

    def __init__(
        self,
        file_manager: Optional[FileManager] = None,
        interval_minutes: int = 60,
        max_age_hours: Optional[int] = None,
    ):
        """
        Initialize CleanupScheduler.

        Args:
            file_manager: FileManager instance to use (or get global)
            interval_minutes: How often to run cleanup (default: 60 min)
            max_age_hours: Override default max age for cleanup
        """
        self.file_manager = file_manager or get_file_manager()
        self.interval_minutes = interval_minutes
        self.max_age_hours = max_age_hours
        self._task = None
        self._running = False

        logger.info(f"CleanupScheduler initialized (interval={interval_minutes}min)")

    async def cleanup_task(self):
        """Background task that runs cleanup periodically"""
        import asyncio

        while self._running:
            try:
                logger.info("Running scheduled cleanup...")

                # Clean up tracked files
                deleted_tracked = self.file_manager.cleanup_expired_files(
                    self.max_age_hours
                )
                logger.info(f"Cleaned up {len(deleted_tracked)} tracked files")

                # Clean up all temp directories
                deleted_dirs = self.file_manager.cleanup_all_temp()
                total_deleted = sum(len(files) for files in deleted_dirs.values())
                logger.info(f"Cleaned up {total_deleted} files from temp directories")

                # Log statistics
                stats = self.file_manager.get_stats()
                logger.info(f"Temp storage: {stats['total_temp_mb']:.2f} MB")

                # Check if temp storage is approaching limit
                if stats["total_temp_mb"] > (stats["max_temp_gb"] * 1024 * 0.9):
                    logger.warning(
                        f"Temp storage at {stats['total_temp_mb']:.2f} MB "
                        f"(90% of {stats['max_temp_gb']} GB limit)"
                    )

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}", exc_info=True)

            # Wait for next interval
            await asyncio.sleep(self.interval_minutes * 60)

    async def start(self):
        """Start the cleanup scheduler"""
        if self._running:
            logger.warning("CleanupScheduler already running")
            return

        import asyncio

        self._running = True
        self._task = asyncio.create_task(self.cleanup_task())
        logger.info(f"CleanupScheduler started (interval={self.interval_minutes}min)")

    async def stop(self):
        """Stop the cleanup scheduler"""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

        logger.info("CleanupScheduler stopped")

    async def run_now(self):
        """Manually trigger cleanup immediately"""
        logger.info("Manual cleanup triggered")

        deleted_tracked = self.file_manager.cleanup_expired_files(self.max_age_hours)
        deleted_dirs = self.file_manager.cleanup_all_temp()
        total_deleted = len(deleted_tracked) + sum(
            len(files) for files in deleted_dirs.values()
        )

        logger.info(f"Manual cleanup completed: {total_deleted} files deleted")
        return total_deleted


# Global scheduler instance
_cleanup_scheduler_instance: Optional[CleanupScheduler] = None


def get_cleanup_scheduler() -> CleanupScheduler:
    """
    Get or create the global CleanupScheduler instance.

    Returns:
        CleanupScheduler instance
    """
    global _cleanup_scheduler_instance

    if _cleanup_scheduler_instance is None:
        _cleanup_scheduler_instance = CleanupScheduler()

    return _cleanup_scheduler_instance
