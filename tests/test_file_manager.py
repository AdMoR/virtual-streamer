"""
Unit tests for File Manager

Tests file path configuration, naming strategies, file tracking, and cleanup functionality.
"""

import os
import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from virtual_streamer.utils.file_manager import (
    FilePathConfig,
    FileManager,
    FileNamingStrategy,
    FileCategory,
    FileTracker,
    get_file_manager,
    reset_file_manager
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_config(temp_test_dir):
    """Create a test configuration"""
    return FilePathConfig(
        data_dir=os.path.join(temp_test_dir, "data"),
        temp_dir=os.path.join(temp_test_dir, "temp"),
        output_dir=os.path.join(temp_test_dir, "output"),
        auto_cleanup=True,
        cleanup_age_hours=1,
        use_timestamps=True,
        use_uuids=True
    )


@pytest.fixture
def file_manager(test_config):
    """Create a FileManager instance for testing"""
    return FileManager(config=test_config)


@pytest.fixture(autouse=True)
def reset_global_manager():
    """Reset global manager before each test"""
    reset_file_manager()
    yield
    reset_file_manager()


# ============================================================================
# FilePathConfig Tests
# ============================================================================

class TestFilePathConfig:
    """Test FilePathConfig functionality"""
    
    def test_default_initialization(self):
        """Test default config values"""
        config = FilePathConfig()
        
        assert config.data_dir is not None
        assert config.temp_dir is not None
        assert config.output_dir is not None
        assert config.auto_cleanup is True
        assert config.cleanup_age_hours == 24
    
    def test_custom_initialization(self, temp_test_dir):
        """Test custom config values"""
        config = FilePathConfig(
            data_dir="/custom/data",
            temp_dir="/custom/temp",
            output_dir="/custom/output",
            cleanup_age_hours=48
        )
        
        assert config.data_dir == "/custom/data"
        assert config.temp_dir == "/custom/temp"
        assert config.output_dir == "/custom/output"
        assert config.cleanup_age_hours == 48
    
    def test_subdirectory_defaults(self, test_config):
        """Test that subdirectories default correctly"""
        assert test_config.audio_temp_dir == os.path.join(test_config.temp_dir, "audio")
        assert test_config.video_temp_dir == os.path.join(test_config.temp_dir, "video")
        assert test_config.subtitle_temp_dir == os.path.join(test_config.temp_dir, "subtitles")
        assert test_config.concat_temp_dir == os.path.join(test_config.temp_dir, "concat")
    
    def test_custom_subdirectories(self, temp_test_dir):
        """Test custom subdirectory paths"""
        custom_audio = os.path.join(temp_test_dir, "custom_audio")
        config = FilePathConfig(
            temp_dir=temp_test_dir,
            audio_temp_dir=custom_audio
        )
        
        assert config.audio_temp_dir == custom_audio
    
    def test_get_category_dir(self, test_config):
        """Test getting directory for each category"""
        assert test_config.get_category_dir(FileCategory.AUDIO) == test_config.audio_temp_dir
        assert test_config.get_category_dir(FileCategory.VIDEO) == test_config.video_temp_dir
        assert test_config.get_category_dir(FileCategory.SUBTITLE) == test_config.subtitle_temp_dir
        assert test_config.get_category_dir(FileCategory.OUTPUT, temp=False) == test_config.output_dir
    
    @patch.dict(os.environ, {
        "DATA_DIR": "/env/data",
        "TEMP_DIR": "/env/temp",
        "AUTO_CLEANUP_TEMP": "false",
        "CLEANUP_AGE_HOURS": "48"
    })
    def test_environment_variable_loading(self):
        """Test loading config from environment variables"""
        config = FilePathConfig()
        
        assert config.data_dir == "/env/data"
        assert config.temp_dir == "/env/temp"
        assert config.auto_cleanup is False
        assert config.cleanup_age_hours == 48


# ============================================================================
# FileNamingStrategy Tests
# ============================================================================

class TestFileNamingStrategy:
    """Test FileNamingStrategy functionality"""
    
    def test_generate_filename_basic(self, test_config):
        """Test basic filename generation"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.generate_filename("test", "txt", include_timestamp=False, include_uuid=False)
        assert filename == "test.txt"
    
    def test_generate_filename_with_suffix(self, test_config):
        """Test filename generation with suffix"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.generate_filename(
            "test", "txt", suffix="data",
            include_timestamp=False, include_uuid=False
        )
        assert filename == "test_data.txt"
    
    def test_generate_filename_with_timestamp(self, test_config):
        """Test filename generation with timestamp"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.generate_filename("test", "txt", include_timestamp=True, include_uuid=False)
        
        # Check format: test_YYYYMMDD_HHMMSS.txt
        assert filename.startswith("test_")
        assert filename.endswith(".txt")
        assert len(filename) == len("test_20251112_143022.txt")
    
    def test_generate_filename_with_uuid(self, test_config):
        """Test filename generation with UUID"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.generate_filename("test", "txt", include_timestamp=False, include_uuid=True)
        
        # Check format: test_12345678.txt (8 char UUID)
        assert filename.startswith("test_")
        assert filename.endswith(".txt")
        parts = filename.replace(".txt", "").split("_")
        assert len(parts) == 2
        assert len(parts[1]) == 8  # UUID is 8 characters
    
    def test_tts_filename(self, test_config):
        """Test TTS filename generation"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.tts_filename("char1", "entry123")
        
        assert filename.startswith("tts_char1_entry123")
        assert filename.endswith(".wav")
    
    def test_wav2lip_filename(self, test_config):
        """Test Wav2Lip filename generation"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.wav2lip_filename("char1", "session456")
        
        assert filename.startswith("wav2lip_char1_session456")
        assert filename.endswith(".avi")
    
    def test_subtitle_filename(self, test_config):
        """Test subtitle filename generation"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.subtitle_filename("audio1", "srt")
        
        assert filename.startswith("subtitle_audio1")
        assert filename.endswith(".srt")
    
    def test_combined_filename(self, test_config):
        """Test combined video filename generation"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.combined_filename("req123")
        
        assert filename.startswith("combined_req123")
        assert filename.endswith(".mp4")
    
    def test_final_output_filename(self, test_config):
        """Test final output filename generation"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.final_output_filename("My Test Video Title")
        
        assert filename.startswith("video_my_test_video_title")
        assert filename.endswith(".mp4")
    
    def test_final_output_filename_sanitization(self, test_config):
        """Test that special characters are sanitized"""
        naming = FileNamingStrategy(test_config)
        
        filename = naming.final_output_filename("Test/Video:With*Special?Chars")
        
        # Should not contain special characters
        assert "/" not in filename
        assert ":" not in filename
        assert "*" not in filename
        assert "?" not in filename
    
    def test_temp_dir_name(self, test_config):
        """Test temporary directory name generation"""
        naming = FileNamingStrategy(test_config)
        
        dirname = naming.temp_dir_name("wav2lip")
        
        assert dirname.startswith("wav2lip_")
        parts = dirname.split("_")
        assert len(parts) == 2
        assert len(parts[1]) == 8  # UUID


# ============================================================================
# FileTracker Tests
# ============================================================================

class TestFileTracker:
    """Test FileTracker functionality"""
    
    def test_initialization(self, temp_test_dir):
        """Test FileTracker initialization"""
        filepath = os.path.join(temp_test_dir, "test.txt")
        tracker = FileTracker(
            filepath=filepath,
            category=FileCategory.AUDIO,
            purpose="Test file"
        )
        
        assert tracker.filepath == filepath
        assert tracker.category == FileCategory.AUDIO
        assert tracker.purpose == "Test file"
        assert tracker.should_cleanup is True
        assert isinstance(tracker.created_at, datetime)
    
    def test_is_expired_not_expired(self, temp_test_dir):
        """Test that new files are not expired"""
        tracker = FileTracker(
            filepath=os.path.join(temp_test_dir, "test.txt"),
            category=FileCategory.AUDIO,
            max_age_hours=1
        )
        
        assert not tracker.is_expired()
    
    def test_is_expired_old_file(self, temp_test_dir):
        """Test that old files are expired"""
        tracker = FileTracker(
            filepath=os.path.join(temp_test_dir, "test.txt"),
            category=FileCategory.AUDIO,
            max_age_hours=1
        )
        
        # Simulate old file by changing created_at
        tracker.created_at = datetime.now() - timedelta(hours=2)
        
        assert tracker.is_expired()
    
    def test_exists(self, temp_test_dir):
        """Test file existence check"""
        filepath = os.path.join(temp_test_dir, "test.txt")
        
        tracker = FileTracker(filepath=filepath, category=FileCategory.AUDIO)
        
        assert not tracker.exists()  # File doesn't exist yet
        
        # Create the file
        Path(filepath).touch()
        
        assert tracker.exists()  # Now it exists
    
    def test_get_size_mb(self, temp_test_dir):
        """Test getting file size"""
        filepath = os.path.join(temp_test_dir, "test.txt")
        
        tracker = FileTracker(filepath=filepath, category=FileCategory.AUDIO)
        
        assert tracker.get_size_mb() == 0.0  # File doesn't exist
        
        # Create file with known size
        with open(filepath, "w") as f:
            f.write("x" * (1024 * 1024))  # 1MB
        
        size = tracker.get_size_mb()
        assert size > 0.9 and size < 1.1  # Approximately 1MB


# ============================================================================
# FileManager Tests
# ============================================================================

class TestFileManager:
    """Test FileManager functionality"""
    
    def test_initialization(self, file_manager, test_config):
        """Test FileManager initialization"""
        assert file_manager.config == test_config
        assert isinstance(file_manager.naming, FileNamingStrategy)
        assert file_manager.tracked_files == []
    
    def test_directories_created(self, file_manager, test_config):
        """Test that directories are created on initialization"""
        assert os.path.exists(test_config.data_dir)
        assert os.path.exists(test_config.temp_dir)
        assert os.path.exists(test_config.output_dir)
        assert os.path.exists(test_config.audio_temp_dir)
        assert os.path.exists(test_config.video_temp_dir)
    
    def test_get_temp_path(self, file_manager, test_config):
        """Test getting temporary file path"""
        path = file_manager.get_temp_path("test.wav", FileCategory.AUDIO)
        
        assert path.startswith(test_config.audio_temp_dir)
        assert path.endswith("test.wav")
        assert os.path.isabs(path)
    
    def test_get_temp_path_with_subdirs(self, file_manager, test_config):
        """Test getting temporary file path with subdirectories"""
        path = file_manager.get_temp_path("subdir/test.wav", FileCategory.AUDIO, create_dir=True)
        
        assert path.startswith(test_config.audio_temp_dir)
        assert path.endswith("test.wav")
        assert os.path.exists(os.path.dirname(path))
    
    def test_get_output_path(self, file_manager, test_config):
        """Test getting output file path"""
        path = file_manager.get_output_path("final.mp4")
        
        assert path.startswith(test_config.output_dir)
        assert path.endswith("final.mp4")
        assert os.path.isabs(path)
    
    def test_register_file(self, file_manager):
        """Test registering a file for tracking"""
        filepath = "/tmp/test.wav"
        
        tracker = file_manager.register_file(
            filepath,
            FileCategory.AUDIO,
            purpose="Test audio",
            max_age_hours=2
        )
        
        assert tracker in file_manager.tracked_files
        assert tracker.filepath == filepath
        assert tracker.category == FileCategory.AUDIO
        assert tracker.purpose == "Test audio"
        assert tracker.max_age_hours == 2
    
    def test_cleanup_expired_files_disabled(self, test_config, temp_test_dir):
        """Test that cleanup doesn't run when disabled"""
        test_config.auto_cleanup = False
        file_manager = FileManager(config=test_config)
        
        # Create and register an old file
        filepath = os.path.join(temp_test_dir, "old.txt")
        Path(filepath).touch()
        
        tracker = file_manager.register_file(filepath, FileCategory.TEMP, max_age_hours=0)
        tracker.created_at = datetime.now() - timedelta(hours=1)
        
        deleted = file_manager.cleanup_expired_files()
        
        assert len(deleted) == 0
        assert os.path.exists(filepath)  # File not deleted
    
    def test_cleanup_expired_files(self, file_manager, temp_test_dir):
        """Test cleaning up expired files"""
        # Create an old file
        filepath = os.path.join(temp_test_dir, "old.txt")
        Path(filepath).touch()
        
        # Register and make it old
        tracker = file_manager.register_file(filepath, FileCategory.TEMP, max_age_hours=1)
        tracker.created_at = datetime.now() - timedelta(hours=2)
        
        deleted = file_manager.cleanup_expired_files()
        
        assert filepath in deleted
        assert not os.path.exists(filepath)
        assert tracker not in file_manager.tracked_files
    
    def test_cleanup_expired_files_keeps_recent(self, file_manager, temp_test_dir):
        """Test that recent files are not cleaned up"""
        # Create a recent file
        filepath = os.path.join(temp_test_dir, "recent.txt")
        Path(filepath).touch()
        
        file_manager.register_file(filepath, FileCategory.TEMP, max_age_hours=24)
        
        deleted = file_manager.cleanup_expired_files()
        
        assert filepath not in deleted
        assert os.path.exists(filepath)
    
    def test_cleanup_expired_files_respects_should_cleanup(self, file_manager, temp_test_dir):
        """Test that files with should_cleanup=False are not deleted"""
        filepath = os.path.join(temp_test_dir, "keep.txt")
        Path(filepath).touch()
        
        tracker = file_manager.register_file(
            filepath, FileCategory.TEMP,
            max_age_hours=1, should_cleanup=False
        )
        tracker.created_at = datetime.now() - timedelta(hours=2)
        
        deleted = file_manager.cleanup_expired_files()
        
        assert filepath not in deleted
        assert os.path.exists(filepath)
    
    def test_cleanup_directory(self, file_manager, temp_test_dir):
        """Test cleaning up a specific directory"""
        # Create some old files
        for i in range(3):
            filepath = os.path.join(temp_test_dir, f"old_{i}.txt")
            Path(filepath).touch()
            # Make file old
            mtime = (datetime.now() - timedelta(hours=2)).timestamp()
            os.utime(filepath, (mtime, mtime))
        
        # Create a recent file
        recent = os.path.join(temp_test_dir, "recent.txt")
        Path(recent).touch()
        
        deleted = file_manager.cleanup_directory(temp_test_dir, max_age_hours=1)
        
        assert len(deleted) == 3
        assert os.path.exists(recent)
        for i in range(3):
            assert not os.path.exists(os.path.join(temp_test_dir, f"old_{i}.txt"))
    
    def test_cleanup_all_temp(self, file_manager, test_config):
        """Test cleaning up all temp directories"""
        # Create old files in different categories
        categories = [
            ("audio", test_config.audio_temp_dir),
            ("video", test_config.video_temp_dir),
            ("subtitle", test_config.subtitle_temp_dir),
        ]
        
        for cat_name, cat_dir in categories:
            filepath = os.path.join(cat_dir, f"old_{cat_name}.txt")
            Path(filepath).touch()
            # Make file old
            mtime = (datetime.now() - timedelta(hours=2)).timestamp()
            os.utime(filepath, (mtime, mtime))
        
        results = file_manager.cleanup_all_temp()
        
        assert len(results["audio"]) == 1
        assert len(results["video"]) == 1
        assert len(results["subtitle"]) == 1
    
    def test_get_temp_size(self, file_manager, test_config):
        """Test getting temp directory sizes"""
        # Create some files
        audio_file = os.path.join(test_config.audio_temp_dir, "test.wav")
        with open(audio_file, "wb") as f:
            f.write(b"x" * (1024 * 1024))  # 1MB
        
        sizes = file_manager.get_temp_size()
        
        assert "audio" in sizes
        assert "video" in sizes
        assert "total" in sizes
        assert sizes["audio"] > 0.9  # Approximately 1MB
    
    def test_get_stats(self, file_manager, temp_test_dir):
        """Test getting file manager statistics"""
        # Register some files
        for i in range(3):
            filepath = os.path.join(temp_test_dir, f"file_{i}.txt")
            file_manager.register_file(filepath, FileCategory.TEMP)
        
        stats = file_manager.get_stats()
        
        assert stats["tracked_files"] == 3
        assert "temp_sizes_mb" in stats
        assert "total_temp_mb" in stats
        assert stats["auto_cleanup"] is True
        assert stats["cleanup_age_hours"] == 1


# ============================================================================
# Global Instance Tests
# ============================================================================

class TestGlobalInstance:
    """Test global FileManager instance"""
    
    @patch.dict(os.environ, {
        "DATA_DIR": "/tmp/test_data",
        "TEMP_DIR": "/tmp/test_temp",
        "OUT_VIDEO_FOLDER": "/tmp/test_output"
    })
    def test_get_file_manager_singleton(self):
        """Test that get_file_manager returns same instance"""
        reset_file_manager()  # Start fresh
        mgr1 = get_file_manager()
        mgr2 = get_file_manager()
        
        assert mgr1 is mgr2
        reset_file_manager()  # Clean up
    
    @patch.dict(os.environ, {
        "DATA_DIR": "/tmp/test_data",
        "TEMP_DIR": "/tmp/test_temp",
        "OUT_VIDEO_FOLDER": "/tmp/test_output"
    })
    def test_reset_file_manager(self):
        """Test resetting global instance"""
        reset_file_manager()  # Start fresh
        mgr1 = get_file_manager()
        reset_file_manager()
        mgr2 = get_file_manager()
        
        assert mgr1 is not mgr2
        reset_file_manager()  # Clean up


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for FileManager"""
    
    def test_complete_workflow(self, file_manager, temp_test_dir):
        """Test complete file lifecycle workflow"""
        # 1. Generate filename using naming strategy
        filename = file_manager.naming.tts_filename("char1", "entry123")
        
        # 2. Get temp path
        filepath = file_manager.get_temp_path(filename, FileCategory.AUDIO)
        
        # 3. Create the file
        Path(filepath).touch()
        
        # 4. Register for tracking
        tracker = file_manager.register_file(
            filepath,
            FileCategory.AUDIO,
            purpose="TTS output",
            max_age_hours=1
        )
        
        # 5. Verify file exists
        assert tracker.exists()
        
        # 6. Make file old and cleanup
        tracker.created_at = datetime.now() - timedelta(hours=2)
        deleted = file_manager.cleanup_expired_files()
        
        # 7. Verify file was deleted
        assert filepath in deleted
        assert not os.path.exists(filepath)
    
    def test_multiple_categories(self, file_manager, test_config):
        """Test working with multiple file categories"""
        files = []
        
        # Create files in different categories
        categories = [
            (FileCategory.AUDIO, "test.wav"),
            (FileCategory.VIDEO, "test.mp4"),
            (FileCategory.SUBTITLE, "test.srt"),
        ]
        
        for category, filename in categories:
            filepath = file_manager.get_temp_path(filename, category)
            Path(filepath).touch()
            file_manager.register_file(filepath, category)
            files.append(filepath)
        
        # Verify all files are in correct directories
        assert files[0].startswith(test_config.audio_temp_dir)
        assert files[1].startswith(test_config.video_temp_dir)
        assert files[2].startswith(test_config.subtitle_temp_dir)
        
        # Verify all are tracked
        assert len(file_manager.tracked_files) == 3
        
        # Get stats
        stats = file_manager.get_stats()
        assert stats["tracked_files"] == 3

