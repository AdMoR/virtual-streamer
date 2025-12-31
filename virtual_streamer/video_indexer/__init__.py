"""
Video Indexer module for processing and indexing video content.

This module provides a modular pipeline for:
- Video embedding (VideoPrism)
- Video description (Florence)
- Face identification
- Audio transcription

Example usage:
    from virtual_streamer.video_indexer import VideoIndexer
    from virtual_streamer.video_indexer.embedders import VideoPrismEmbedder
    from virtual_streamer.video_indexer.describers import FlorenceDescriber
    from virtual_streamer.video_indexer import FaceRecognitionIdentifier
    
    indexer = VideoIndexer(
        embedder=VideoPrismEmbedder(),
        describer=FlorenceDescriber(),
        face_identifier=FaceRecognitionIdentifier(),
    )
    
    metadata = indexer.index("video.mp4")
"""

from virtual_streamer.video_indexer.face_identifier import FaceRecognitionIdentifier
from virtual_streamer.video_indexer.index_builder import (
    HybridVideoRetriever,
    VideoIndexBuilder,
)
from virtual_streamer.video_indexer.indexer import VideoIndexer
from virtual_streamer.video_indexer.interfaces import (
    FaceIdentifier,
    VideoDescriber,
    VideoEmbedder,
    VideoMetadata,
)

__all__ = [
    # Interfaces
    "VideoEmbedder",
    "VideoDescriber",
    "FaceIdentifier",
    "VideoMetadata",
    # Implementations
    "FaceRecognitionIdentifier",
    # Orchestrators
    "VideoIndexer",
    "VideoIndexBuilder",
    "HybridVideoRetriever",
]

