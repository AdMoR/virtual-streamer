"""
Vector index builder for video embeddings.

Builds searchable vector indices from video embeddings for
efficient similarity search and retrieval.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from virtual_streamer.video_indexer.interfaces import VideoMetadata

logger = logging.getLogger(__name__)


class VideoIndexBuilder:
    """Builder for vector indices from video embeddings.
    
    Creates and manages vector indices for efficient video retrieval
    using embeddings from VideoPrism or similar models.
    
    Supports multiple backends:
    - numpy: Simple brute-force search (good for small datasets)
    - faiss: Facebook AI Similarity Search (efficient for large datasets)
    - llama_index: Integration with LlamaIndex for RAG applications
    
    Attributes:
        index_path: Path to store/load the index.
        embedding_dim: Dimension of embeddings.
    """

    def __init__(
        self,
        index_path: str,
        embedding_dim: int = 768,
    ):
        """Initialize index builder.
        
        Args:
            index_path: Path to store/load the index.
            embedding_dim: Dimension of embeddings (default: 768 for VideoPrism base).
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        
        # Storage for embeddings and metadata
        self._embeddings: Optional[np.ndarray] = None
        self._video_paths: List[str] = []
        self._metadata: List[VideoMetadata] = []
        
        # Create index directory
        os.makedirs(index_path, exist_ok=True)

    def add_from_directory(
        self,
        metadata_dir: str,
        character_filter: Optional[str] = None,
    ) -> int:
        """Add videos from a metadata directory.
        
        Args:
            metadata_dir: Directory containing JSON metadata files.
            character_filter: Optional filter by character name (from "who" field).
            
        Returns:
            Number of videos added.
        """
        embeddings_list = []
        count = 0
        
        embeddings_dir = os.path.join(metadata_dir, "embeddings")
        
        for filename in os.listdir(metadata_dir):
            if not filename.endswith(".json"):
                continue
            
            json_path = os.path.join(metadata_dir, filename)
            
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                metadata = VideoMetadata.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load {json_path}: {e}")
                continue
            
            # Apply character filter
            if character_filter:
                # Extract unique character names from "who" field
                characters = set(name for name, _ in metadata.who)
                if character_filter not in characters:
                    continue
            
            # Load embedding
            if metadata.embedding_path and os.path.exists(metadata.embedding_path):
                embedding = np.load(metadata.embedding_path)
            else:
                # Try default location
                default_path = os.path.join(
                    embeddings_dir, f"{Path(metadata.path).stem}.npy"
                )
                if os.path.exists(default_path):
                    embedding = np.load(default_path)
                    metadata.embedding_path = default_path
                else:
                    logger.warning(f"No embedding found for {metadata.path}")
                    continue
            
            embeddings_list.append(embedding)
            self._video_paths.append(metadata.path)
            self._metadata.append(metadata)
            count += 1
        
        if embeddings_list:
            new_embeddings = np.stack(embeddings_list)
            if self._embeddings is None:
                self._embeddings = new_embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, new_embeddings])
        
        logger.info(f"Added {count} videos to index")
        return count

    def add_embedding(
        self,
        video_path: str,
        embedding: np.ndarray,
        metadata: Optional[VideoMetadata] = None,
    ) -> None:
        """Add a single embedding to the index.
        
        Args:
            video_path: Path to the video file.
            embedding: Embedding vector.
            metadata: Optional VideoMetadata.
        """
        embedding = embedding.reshape(1, -1)
        
        if self._embeddings is None:
            self._embeddings = embedding
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])
        
        self._video_paths.append(video_path)
        
        if metadata:
            self._metadata.append(metadata)
        else:
            self._metadata.append(VideoMetadata(path=video_path, duration=0.0))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float, VideoMetadata]]:
        """Search for similar videos.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of (video_path, similarity_score, metadata) tuples,
            sorted by descending similarity.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize stored embeddings
        embeddings_norm = self._embeddings / np.linalg.norm(
            self._embeddings, axis=1, keepdims=True
        )
        
        # Compute cosine similarities
        similarities = np.dot(embeddings_norm, query_norm.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self._video_paths[idx],
                float(similarities[idx]),
                self._metadata[idx],
            ))
        
        return results

    def search_by_text(
        self,
        text_query: str,
        embedder,  # VideoEmbedder with embed_with_text_query method
        top_k: int = 10,
    ) -> List[Tuple[str, float, VideoMetadata]]:
        """Search for videos using a text query.
        
        Requires a video-text embedder (like VideoPrism LVT) that can
        generate text embeddings compatible with video embeddings.
        
        Args:
            text_query: Text query string.
            embedder: VideoEmbedder with embed_with_text_query capability.
            top_k: Number of results to return.
            
        Returns:
            List of (video_path, similarity_score, metadata) tuples.
        """
        if not hasattr(embedder, "embed_with_text_query"):
            raise ValueError(
                "Embedder must support embed_with_text_query for text search"
            )
        
        # Get text embedding (using a dummy video path)
        # For LVT models, we need to pass a video even for text-only embedding
        # This is a workaround - in practice you might want to store text embeddings
        
        from videoprism import models as vp
        
        # Tokenize text
        text_tokenizer = vp.load_text_tokenizer("c4_en")
        text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, [text_query])
        
        # For pure text embedding, we'd need access to the model's text encoder
        # This is a limitation of the current VideoPrism API
        # For now, raise an informative error
        raise NotImplementedError(
            "Pure text search requires access to text encoder. "
            "Use search() with a video query embedding instead, "
            "or use hybrid search with BM25 for text queries."
        )

    def save(self) -> None:
        """Save the index to disk."""
        if self._embeddings is None:
            logger.warning("No embeddings to save")
            return
        
        # Save embeddings
        np.save(
            os.path.join(self.index_path, "embeddings.npy"),
            self._embeddings
        )
        
        # Save video paths
        with open(os.path.join(self.index_path, "video_paths.json"), "w") as f:
            json.dump(self._video_paths, f)
        
        # Save metadata
        metadata_dicts = [m.to_dict() for m in self._metadata]
        with open(os.path.join(self.index_path, "metadata.json"), "w") as f:
            json.dump(metadata_dicts, f, indent=2)
        
        logger.info(f"Saved index with {len(self._video_paths)} videos to {self.index_path}")

    def load(self) -> bool:
        """Load the index from disk.
        
        Returns:
            True if index was loaded successfully, False otherwise.
        """
        embeddings_path = os.path.join(self.index_path, "embeddings.npy")
        paths_path = os.path.join(self.index_path, "video_paths.json")
        metadata_path = os.path.join(self.index_path, "metadata.json")
        
        if not all(os.path.exists(p) for p in [embeddings_path, paths_path, metadata_path]):
            return False
        
        try:
            self._embeddings = np.load(embeddings_path)
            
            with open(paths_path, "r") as f:
                self._video_paths = json.load(f)
            
            with open(metadata_path, "r") as f:
                metadata_dicts = json.load(f)
            self._metadata = [VideoMetadata.from_dict(d) for d in metadata_dicts]
            
            logger.info(f"Loaded index with {len(self._video_paths)} videos")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    @property
    def size(self) -> int:
        """Return the number of videos in the index."""
        return len(self._video_paths)

    def get_metadata(self, video_path: str) -> Optional[VideoMetadata]:
        """Get metadata for a specific video.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            VideoMetadata if found, None otherwise.
        """
        try:
            idx = self._video_paths.index(video_path)
            return self._metadata[idx]
        except ValueError:
            return None


class HybridVideoRetriever:
    """Hybrid retriever combining video embeddings with text search.
    
    Uses video embeddings for semantic similarity and BM25/text
    embeddings for keyword matching.
    
    Attributes:
        embedding_index: VideoIndexBuilder for embedding-based search.
        text_retriever: Text-based retriever (BM25 or vector).
    """

    def __init__(
        self,
        embedding_index: VideoIndexBuilder,
        text_retriever=None,  # BM25Retriever or VectorStoreIndex
        embedding_weight: float = 0.6,
    ):
        """Initialize hybrid retriever.
        
        Args:
            embedding_index: VideoIndexBuilder for embedding search.
            text_retriever: Optional text-based retriever.
            embedding_weight: Weight for embedding scores (default: 0.6).
        """
        self.embedding_index = embedding_index
        self.text_retriever = text_retriever
        self.embedding_weight = embedding_weight

    def search(
        self,
        query_embedding: Optional[np.ndarray] = None,
        text_query: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float, VideoMetadata]]:
        """Search using both embedding and text queries.
        
        Args:
            query_embedding: Optional video/query embedding.
            text_query: Optional text query string.
            top_k: Number of results to return.
            
        Returns:
            List of (video_path, combined_score, metadata) tuples.
        """
        results_dict: Dict[str, Tuple[float, VideoMetadata]] = {}
        
        # Embedding-based search
        if query_embedding is not None:
            embedding_results = self.embedding_index.search(query_embedding, top_k=top_k * 2)
            for path, score, metadata in embedding_results:
                results_dict[path] = (score * self.embedding_weight, metadata)
        
        # Text-based search
        if text_query and self.text_retriever:
            text_results = self.text_retriever.retrieve(text_query)
            text_weight = 1.0 - self.embedding_weight
            
            for result in text_results[:top_k * 2]:
                path = result.node.metadata.get("path", "")
                score = result.score if hasattr(result, "score") else 0.5
                
                if path in results_dict:
                    # Combine scores
                    existing_score, metadata = results_dict[path]
                    results_dict[path] = (existing_score + score * text_weight, metadata)
                else:
                    # Create metadata from result
                    metadata = VideoMetadata(
                        path=path,
                        duration=result.node.metadata.get("duration", 0.0),
                        who=[(result.node.metadata.get("who", ""), 0)],
                        transcription=result.node.text,
                    )
                    results_dict[path] = (score * text_weight, metadata)
        
        # Sort by combined score
        sorted_results = sorted(
            [(path, score, meta) for path, (score, meta) in results_dict.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_results[:top_k]

