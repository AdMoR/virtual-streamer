"""
VideoPrism video embedding implementation.

Uses Google's VideoPrism model for generating video embeddings
via JAX/Flax. Supports both pure video encoding and video-text
encoding for text-to-video retrieval.
"""

from typing import List, Optional, Tuple

import jax
import mediapy
import numpy as np

from virtual_streamer.video_indexer.interfaces import VideoEmbedder


def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: Tuple[int, int]
) -> np.ndarray:
    """Read and preprocess a video for VideoPrism.
    
    Args:
        filename: Path to video file.
        target_num_frames: Number of frames to sample.
        target_frame_size: Target (height, width) for frames.
        
    Returns:
        Preprocessed frames as numpy array of shape 
        (num_frames, height, width, 3) with values in [0.0, 1.0].
    """
    frames = mediapy.read_video(filename)
    
    # Sample to target number of frames
    frame_indices = np.linspace(
        0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])
    
    # Resize to target size
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    
    # Handle aspect ratio mismatch by center cropping
    if original_height * target_width != original_width * target_height:
        # Calculate crop dimensions to match target aspect ratio
        target_ratio = target_width / target_height
        current_ratio = original_width / original_height
        
        if current_ratio > target_ratio:
            # Video is wider, crop width
            new_width = int(original_height * target_ratio)
            start_x = (original_width - new_width) // 2
            frames = frames[:, :, start_x:start_x + new_width, :]
        else:
            # Video is taller, crop height
            new_height = int(original_width / target_ratio)
            start_y = (original_height - new_height) // 2
            frames = frames[:, start_y:start_y + new_height, :, :]
    
    frames = mediapy.resize_video(frames, shape=target_frame_size)
    
    # Normalize pixel values to [0.0, 1.0]
    frames = mediapy.to_float01(frames)
    
    return frames


class VideoPrismEmbedder(VideoEmbedder):
    """Video embedder using Google's VideoPrism model.
    
    This implementation uses the video-text encoder variant (LVT)
    which allows for text-to-video retrieval.
    
    Attributes:
        model_name: Name of the VideoPrism model configuration.
        num_frames: Number of frames to sample from each video.
        frame_size: Target frame size as (height, width).
    """

    def __init__(
        self,
        model_name: str = "videoprism_lvt_public_v1_base",
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize VideoPrism embedder.
        
        Args:
            model_name: VideoPrism model configuration name.
                       Use "videoprism_public_v1_base" for pure video encoder,
                       or "videoprism_lvt_public_v1_base" for video-text encoder.
            num_frames: Number of frames to sample from videos (default: 16).
            frame_size: Target frame size as (height, width) (default: 224x224).
        """
        from videoprism import models as vp
        
        self.model_name = model_name
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Load model and weights
        self._flax_model = vp.get_model(model_name)
        self._loaded_state = vp.load_pretrained_weights(model_name)
        
        # Check if this is a video-text model (LVT)
        self._is_lvt = "lvt" in model_name.lower()
        
        if self._is_lvt:
            self._text_tokenizer = vp.load_text_tokenizer("c4_en")
            self._forward_fn = self._create_lvt_forward_fn()
        else:
            self._forward_fn = self._create_video_forward_fn()
        
        # Determine embedding dimension by running a dummy forward pass
        self._embedding_dim: Optional[int] = None

    def _create_video_forward_fn(self):
        """Create JIT-compiled forward function for pure video encoder."""
        @jax.jit
        def forward_fn(inputs):
            return self._flax_model.apply(self._loaded_state, inputs, train=False)
        return forward_fn

    def _create_lvt_forward_fn(self):
        """Create JIT-compiled forward function for video-text encoder."""
        @jax.jit
        def forward_fn(inputs, text_token_ids, text_token_paddings):
            return self._flax_model.apply(
                self._loaded_state,
                inputs,
                text_token_ids,
                text_token_paddings,
                train=False,
            )
        return forward_fn

    def _preprocess_video(self, video_path: str) -> np.ndarray:
        """Preprocess a video for the model.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Preprocessed video tensor of shape (1, num_frames, H, W, 3).
        """
        frames = read_and_preprocess_video(
            video_path, self.num_frames, self.frame_size
        )
        # Add batch dimension
        return np.expand_dims(frames, axis=0)

    def embed(self, video_path: str) -> np.ndarray:
        """Generate embedding for a single video.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Embedding vector as numpy array of shape (embedding_dim,).
        """
        video_inputs = self._preprocess_video(video_path)
        
        if self._is_lvt:
            from videoprism import models as vp
            # Use empty text query to get pure video embedding
            text_ids, text_paddings = vp.tokenize_texts(self._text_tokenizer, [""])
            video_embeddings, _, _ = self._forward_fn(
                video_inputs, text_ids, text_paddings
            )
            embedding = np.array(video_embeddings[0])
        else:
            outputs, _ = self._forward_fn(video_inputs)
            # Pool over tokens (mean pooling)
            embedding = np.mean(np.array(outputs[0]), axis=0)
        
        # Cache embedding dimension
        if self._embedding_dim is None:
            self._embedding_dim = embedding.shape[0]
        
        return embedding

    def embed_batch(self, video_paths: List[str]) -> np.ndarray:
        """Generate embeddings for multiple videos.
        
        Args:
            video_paths: List of paths to video files.
            
        Returns:
            2D numpy array of shape (num_videos, embedding_dim).
        """
        if not video_paths:
            return np.array([])
        
        # Preprocess all videos
        video_batch = np.concatenate(
            [self._preprocess_video(path) for path in video_paths],
            axis=0
        )
        
        if self._is_lvt:
            from videoprism import models as vp
            # Use empty text queries
            text_ids, text_paddings = vp.tokenize_texts(
                self._text_tokenizer, [""] * len(video_paths)
            )
            video_embeddings, _, _ = self._forward_fn(
                video_batch, text_ids, text_paddings
            )
            embeddings = np.array(video_embeddings)
        else:
            outputs, _ = self._forward_fn(video_batch)
            # Pool over tokens (mean pooling)
            embeddings = np.mean(np.array(outputs), axis=1)
        
        # Cache embedding dimension
        if self._embedding_dim is None and len(embeddings) > 0:
            self._embedding_dim = embeddings.shape[1]
        
        return embeddings

    def embed_with_text_query(
        self, video_path: str, text_query: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate video and text embeddings for similarity matching.
        
        This is useful for text-to-video retrieval scenarios.
        
        Args:
            video_path: Path to video file.
            text_query: Text query to embed alongside video.
            
        Returns:
            Tuple of (video_embedding, text_embedding).
            
        Raises:
            ValueError: If model is not an LVT (video-text) model.
        """
        if not self._is_lvt:
            raise ValueError(
                "embed_with_text_query requires an LVT model. "
                f"Current model: {self.model_name}"
            )
        
        from videoprism import models as vp
        
        video_inputs = self._preprocess_video(video_path)
        text_ids, text_paddings = vp.tokenize_texts(
            self._text_tokenizer, [text_query]
        )
        
        video_embeddings, text_embeddings, _ = self._forward_fn(
            video_inputs, text_ids, text_paddings
        )
        
        return np.array(video_embeddings[0]), np.array(text_embeddings[0])

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of the embedding vectors.
        
        Note: This may require a forward pass if not yet determined.
        """
        if self._embedding_dim is None:
            # Typical dimensions for VideoPrism models
            if "base" in self.model_name:
                return 768
            elif "large" in self.model_name:
                return 1024
            else:
                return 768  # Default assumption
        return self._embedding_dim

