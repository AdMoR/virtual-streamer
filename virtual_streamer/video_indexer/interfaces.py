"""
Abstract interfaces for video indexing components.

This module defines interfaces for video processing to enable:
- Easy swapping of implementations (e.g., VideoPrism vs other embedders)
- Clean separation of concerns
- Testing with mocks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VideoMetadata:
    """Result of video indexing containing all extracted information."""

    path: str
    duration: float
    who: List[Tuple[str, int]] = field(default_factory=list)
    transcription: Optional[str] = None
    description: Optional[str] = None
    embedding_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": self.path,
            "duration": self.duration,
            "who": self.who,
            "transcription": self.transcription,
            "description": self.description,
            "embedding_path": self.embedding_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoMetadata":
        """Create from dictionary (e.g., loaded from JSON)."""
        return cls(
            path=data["path"],
            duration=data["duration"],
            who=data.get("who", []),
            transcription=data.get("transcription"),
            description=data.get("description"),
            embedding_path=data.get("embedding_path"),
        )


class VideoEmbedder(ABC):
    """Abstract interface for video embedding models.
    
    Video embedders convert video content into numerical embeddings
    suitable for similarity search and retrieval.
    """

    @abstractmethod
    def embed(self, video_path: str) -> np.ndarray:
        """Generate embedding for a single video.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Embedding vector as numpy array.
        """
        pass

    @abstractmethod
    def embed_batch(self, video_paths: List[str]) -> np.ndarray:
        """Generate embeddings for multiple videos.
        
        Args:
            video_paths: List of paths to video files.
            
        Returns:
            2D numpy array of shape (num_videos, embedding_dim).
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass


class VideoDescriber(ABC):
    """Abstract interface for video description models.
    
    Video describers generate text descriptions/captions of video content.
    """

    @abstractmethod
    def describe(self, video_path: str) -> str:
        """Generate text description for a video.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Text description of the video content.
        """
        pass

    @abstractmethod
    def describe_frames(self, video_path: str, num_samples: int = 5) -> List[str]:
        """Generate descriptions for sampled frames from a video.
        
        Args:
            video_path: Path to video file.
            num_samples: Number of frames to sample and describe.
            
        Returns:
            List of descriptions, one per sampled frame.
        """
        pass


class FaceIdentifier(ABC):
    """Abstract interface for face identification in videos.
    
    Face identifiers detect and recognize faces in video frames,
    matching them against known face encodings.
    """

    @abstractmethod
    def load_known_faces(self, face_images: Dict[str, List[str]]) -> None:
        """Load known face encodings from image files.
        
        Args:
            face_images: Dictionary mapping person names to lists of image paths.
                        Example: {"fred": ["fred_1.jpg", "fred_2.jpg"], 
                                  "jamy": ["jamy_1.jpg"]}
        """
        pass

    @abstractmethod
    def identify(
        self, video_path: str, speed_up_factor: int = 4
    ) -> List[Tuple[str, int]]:
        """Identify faces in a video.
        
        Args:
            video_path: Path to video file.
            speed_up_factor: Process every Nth second of video (default: 4).
            
        Returns:
            List of (person_name, frame_index) tuples for each identified face.
        """
        pass

    @property
    @abstractmethod
    def known_face_names(self) -> List[str]:
        """Return list of known face names."""
        pass

