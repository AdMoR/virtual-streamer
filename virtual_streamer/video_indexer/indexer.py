"""
VideoIndexer orchestrator for video indexing pipeline.

Coordinates video embedding, description, face identification,
and audio transcription with support for partial reprocessing.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from virtual_streamer.video_indexer.interfaces import (
    FaceIdentifier,
    VideoDescriber,
    VideoEmbedder,
    VideoMetadata,
)

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """Get duration of a video file in seconds.
    
    Args:
        video_path: Path to video file.
        
    Returns:
        Duration in seconds.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return float(result.stdout)
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def extract_audio(video_path: str, audio_path: str, bitrate: str = "192K") -> str:
    """Extract audio from video file.
    
    Args:
        video_path: Path to video file.
        audio_path: Output path for audio file.
        bitrate: Audio bitrate (default: 192K).
        
    Returns:
        Path to extracted audio file.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-b:a", bitrate, "-vn", audio_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return audio_path


class VideoIndexer:
    """Orchestrator for video indexing pipeline.
    
    Combines multiple processors (embedder, describer, face identifier,
    transcriber) to extract comprehensive metadata from videos.
    
    Supports partial reprocessing by allowing individual components
    to be skipped or run independently.
    
    Attributes:
        embedder: Video embedding model.
        describer: Video description model.
        face_identifier: Face identification model.
        transcriber: Audio transcription model.
        output_dir: Directory for output files.
    """

    def __init__(
        self,
        embedder: Optional[VideoEmbedder] = None,
        describer: Optional[VideoDescriber] = None,
        face_identifier: Optional[FaceIdentifier] = None,
        transcriber=None,  # STTInterface from video_generation
        output_dir: str = "./video_index",
    ):
        """Initialize video indexer.
        
        Args:
            embedder: Video embedding model (optional).
            describer: Video description model (optional).
            face_identifier: Face identification model (optional).
            transcriber: Audio transcription model (optional).
            output_dir: Directory for output files (default: ./video_index).
        """
        self.embedder = embedder
        self.describer = describer
        self.face_identifier = face_identifier
        self.transcriber = transcriber
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)

    def _get_output_paths(self, video_path: str) -> Tuple[str, str]:
        """Get output paths for a video.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Tuple of (json_path, embedding_path).
        """
        video_name = Path(video_path).stem
        json_path = os.path.join(self.output_dir, f"{video_name}.json")
        embedding_path = os.path.join(
            self.output_dir, "embeddings", f"{video_name}.npy"
        )
        return json_path, embedding_path

    def _load_existing_metadata(self, json_path: str) -> Optional[VideoMetadata]:
        """Load existing metadata if available.
        
        Args:
            json_path: Path to JSON metadata file.
            
        Returns:
            VideoMetadata if file exists, None otherwise.
        """
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                return VideoMetadata.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def index(
        self,
        video_path: str,
        skip_embedding: bool = False,
        skip_description: bool = False,
        skip_faces: bool = False,
        skip_transcription: bool = False,
        force_reprocess: bool = False,
    ) -> VideoMetadata:
        """Index a single video.
        
        Args:
            video_path: Path to video file.
            skip_embedding: Skip video embedding (default: False).
            skip_description: Skip video description (default: False).
            skip_faces: Skip face identification (default: False).
            skip_transcription: Skip audio transcription (default: False).
            force_reprocess: Force reprocessing even if output exists (default: False).
            
        Returns:
            VideoMetadata with extracted information.
        """
        json_path, embedding_path = self._get_output_paths(video_path)
        
        # Load existing metadata for partial reprocessing
        existing = None
        if not force_reprocess:
            existing = self._load_existing_metadata(json_path)
        
        # Initialize metadata
        duration = get_video_duration(video_path)
        metadata = VideoMetadata(
            path=video_path,
            duration=duration,
        )
        
        # Copy existing values if available and not being reprocessed
        if existing:
            if skip_embedding and existing.embedding_path:
                metadata.embedding_path = existing.embedding_path
            if skip_description and existing.description:
                metadata.description = existing.description
            if skip_faces and existing.who:
                metadata.who = existing.who
            if skip_transcription and existing.transcription:
                metadata.transcription = existing.transcription
        
        # Video embedding
        if not skip_embedding and self.embedder is not None:
            logger.info(f"Generating embedding for: {video_path}")
            try:
                embedding = self.embedder.embed(video_path)
                np.save(embedding_path, embedding)
                metadata.embedding_path = embedding_path
            except Exception as e:
                logger.error(f"Embedding failed for {video_path}: {e}")
        
        # Video description
        if not skip_description and self.describer is not None:
            logger.info(f"Generating description for: {video_path}")
            try:
                description = self.describer.describe(video_path)
                metadata.description = description
            except Exception as e:
                logger.error(f"Description failed for {video_path}: {e}")
        
        # Face identification
        if not skip_faces and self.face_identifier is not None:
            logger.info(f"Identifying faces in: {video_path}")
            try:
                faces = self.face_identifier.identify(video_path)
                metadata.who = faces
            except Exception as e:
                logger.error(f"Face identification failed for {video_path}: {e}")
        
        # Audio transcription
        if not skip_transcription and self.transcriber is not None:
            logger.info(f"Transcribing audio from: {video_path}")
            try:
                # Extract audio first
                audio_path = os.path.join(
                    self.output_dir, f"{Path(video_path).stem}.mp3"
                )
                extract_audio(video_path, audio_path)
                
                # Transcribe
                transcription = self.transcriber.transcribe(audio_path)
                if hasattr(transcription, "text"):
                    metadata.transcription = transcription.text
                elif isinstance(transcription, dict):
                    metadata.transcription = transcription.get("text", str(transcription))
                else:
                    metadata.transcription = str(transcription)
                
                # Clean up audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                logger.error(f"Transcription failed for {video_path}: {e}")
        
        # Save metadata
        with open(json_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return metadata

    def index_batch(
        self,
        video_paths: List[str],
        skip_embedding: bool = False,
        skip_description: bool = False,
        skip_faces: bool = False,
        skip_transcription: bool = False,
        force_reprocess: bool = False,
        min_duration: float = 0.0,
    ) -> List[VideoMetadata]:
        """Index multiple videos.
        
        Args:
            video_paths: List of paths to video files.
            skip_embedding: Skip video embedding.
            skip_description: Skip video description.
            skip_faces: Skip face identification.
            skip_transcription: Skip audio transcription.
            force_reprocess: Force reprocessing even if output exists.
            min_duration: Minimum video duration in seconds (default: 0).
            
        Returns:
            List of VideoMetadata for each processed video.
        """
        results = []
        
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i + 1}/{len(video_paths)}: {video_path}")
            
            # Check duration filter
            if min_duration > 0:
                duration = get_video_duration(video_path)
                if duration < min_duration:
                    logger.info(f"Skipping {video_path}: duration {duration}s < {min_duration}s")
                    continue
            
            # Check if already processed
            json_path, _ = self._get_output_paths(video_path)
            if not force_reprocess and os.path.exists(json_path):
                logger.info(f"Skipping {video_path}: already processed")
                existing = self._load_existing_metadata(json_path)
                if existing:
                    results.append(existing)
                continue
            
            try:
                metadata = self.index(
                    video_path,
                    skip_embedding=skip_embedding,
                    skip_description=skip_description,
                    skip_faces=skip_faces,
                    skip_transcription=skip_transcription,
                    force_reprocess=force_reprocess,
                )
                results.append(metadata)
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
        
        return results

    def index_directory(
        self,
        directory: str,
        extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
        **kwargs,
    ) -> List[VideoMetadata]:
        """Index all videos in a directory.
        
        Args:
            directory: Path to directory containing videos.
            extensions: Video file extensions to process.
            **kwargs: Additional arguments passed to index_batch.
            
        Returns:
            List of VideoMetadata for each processed video.
        """
        video_paths = []
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(extensions):
                video_paths.append(os.path.join(directory, filename))
        
        video_paths.sort()
        logger.info(f"Found {len(video_paths)} videos in {directory}")
        
        return self.index_batch(video_paths, **kwargs)

    def load_all_metadata(self) -> List[VideoMetadata]:
        """Load all metadata from the output directory.
        
        Returns:
            List of VideoMetadata from all JSON files.
        """
        metadata_list = []
        
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".json"):
                json_path = os.path.join(self.output_dir, filename)
                metadata = self._load_existing_metadata(json_path)
                if metadata:
                    metadata_list.append(metadata)
        
        return metadata_list

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Load all embeddings from the output directory.
        
        Returns:
            Dictionary mapping video paths to embeddings.
        """
        embeddings = {}
        
        for metadata in self.load_all_metadata():
            if metadata.embedding_path and os.path.exists(metadata.embedding_path):
                embedding = np.load(metadata.embedding_path)
                embeddings[metadata.path] = embedding
        
        return embeddings

