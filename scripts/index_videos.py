#!/usr/bin/env python
"""
Video indexing CLI script.

Processes videos to extract embeddings, descriptions, face identifications,
and transcriptions, then builds a searchable vector index.

Usage:
    # Index all videos in a directory
    python scripts/index_videos.py /path/to/videos --output ./video_index
    
    # Index with specific components
    python scripts/index_videos.py /path/to/videos --skip-description --skip-transcription
    
    # Build index from existing metadata
    python scripts/index_videos.py --build-index-only --metadata-dir ./video_index
    
    # Index with face identification
    python scripts/index_videos.py /path/to/videos --faces-config faces.json

Example faces.json:
    {
        "fred": ["assets/fred_1.jpg", "assets/fred_2.jpg", "assets/fred_3.jpg"],
        "jamy": ["assets/jamy_1.jpg", "assets/jamy_2.jpg", "assets/jamy_3.jpg"]
    }
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_faces_config(config_path: str) -> Dict[str, List[str]]:
    """Load face configuration from JSON file.
    
    Args:
        config_path: Path to JSON file with face image paths.
        
    Returns:
        Dictionary mapping names to lists of image paths.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def create_embedder(args):
    """Create video embedder based on arguments."""
    if args.skip_embedding:
        return None
    
    from virtual_streamer.video_indexer.embedders import VideoPrismEmbedder
    
    return VideoPrismEmbedder(
        model_name=args.embedder_model,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
    )


def create_describer(args):
    """Create video describer based on arguments."""
    if args.skip_description:
        return None
    
    from virtual_streamer.video_indexer.describers import FlorenceDescriber
    
    return FlorenceDescriber(
        model_id=args.describer_model,
        task_prompt=args.description_task,
    )


def create_face_identifier(args):
    """Create face identifier based on arguments."""
    if args.skip_faces or not args.faces_config:
        return None
    
    from virtual_streamer.video_indexer.face_identifier import FaceRecognitionIdentifier
    
    identifier = FaceRecognitionIdentifier(
        tolerance=args.face_tolerance,
    )
    
    # Load known faces
    faces_config = load_faces_config(args.faces_config)
    identifier.load_known_faces(faces_config)
    
    return identifier


def create_transcriber(args):
    """Create audio transcriber based on arguments."""
    if args.skip_transcription:
        return None
    
    try:
        import stable_whisper
        
        model = stable_whisper.load_faster_whisper(args.whisper_model)
        
        # Wrap in a simple interface
        class WhisperTranscriber:
            def __init__(self, model):
                self._model = model
            
            def transcribe(self, audio_path: str):
                result = self._model.transcribe(audio_path, language="fr")
                return result
        
        return WhisperTranscriber(model)
    except ImportError:
        logging.warning("stable_whisper not available, skipping transcription")
        return None


def index_videos(args) -> None:
    """Main indexing function."""
    from virtual_streamer.video_indexer.indexer import VideoIndexer
    
    logger = logging.getLogger(__name__)
    
    # Create components
    logger.info("Initializing components...")
    
    embedder = create_embedder(args)
    describer = create_describer(args)
    face_identifier = create_face_identifier(args)
    transcriber = create_transcriber(args)
    
    # Create indexer
    indexer = VideoIndexer(
        embedder=embedder,
        describer=describer,
        face_identifier=face_identifier,
        transcriber=transcriber,
        output_dir=args.output,
    )
    
    # Process videos
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single video
            logger.info(f"Indexing single video: {input_path}")
            metadata = indexer.index(
                str(input_path),
                force_reprocess=args.force,
            )
            logger.info(f"Indexed: {metadata.path}")
        else:
            # Directory
            logger.info(f"Indexing directory: {input_path}")
            results = indexer.index_directory(
                str(input_path),
                min_duration=args.min_duration,
                force_reprocess=args.force,
            )
            logger.info(f"Indexed {len(results)} videos")


def build_index(args) -> None:
    """Build vector index from existing metadata."""
    from virtual_streamer.video_indexer.index_builder import VideoIndexBuilder
    
    logger = logging.getLogger(__name__)
    
    metadata_dir = args.metadata_dir or args.output
    
    logger.info(f"Building index from: {metadata_dir}")
    
    # Create index builder
    builder = VideoIndexBuilder(
        index_path=args.index_output or os.path.join(args.output, "vector_index"),
        embedding_dim=args.embedding_dim,
    )
    
    # Add embeddings from metadata directory
    count = builder.add_from_directory(
        metadata_dir,
        character_filter=args.character_filter,
    )
    
    if count > 0:
        builder.save()
        logger.info(f"Built index with {count} videos")
    else:
        logger.warning("No videos found to index")


def main():
    parser = argparse.ArgumentParser(
        description="Index videos for retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/Output
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to video file or directory",
    )
    parser.add_argument(
        "-o", "--output",
        default="./video_index",
        help="Output directory for metadata and embeddings (default: ./video_index)",
    )
    
    # Processing options
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip video embedding generation",
    )
    parser.add_argument(
        "--skip-description",
        action="store_true",
        help="Skip video description generation",
    )
    parser.add_argument(
        "--skip-faces",
        action="store_true",
        help="Skip face identification",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip audio transcription",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=6.0,
        help="Minimum video duration in seconds (default: 6.0)",
    )
    
    # Model options
    parser.add_argument(
        "--embedder-model",
        default="videoprism_lvt_public_v1_base",
        help="VideoPrism model name (default: videoprism_lvt_public_v1_base)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample for embedding (default: 16)",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=224,
        help="Frame size for embedding (default: 224)",
    )
    parser.add_argument(
        "--describer-model",
        default="microsoft/Florence-2-large",
        help="Florence model ID (default: microsoft/Florence-2-large)",
    )
    parser.add_argument(
        "--description-task",
        default="<MORE_DETAILED_CAPTION>",
        help="Florence task prompt (default: <MORE_DETAILED_CAPTION>)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size (default: base)",
    )
    
    # Face identification
    parser.add_argument(
        "--faces-config",
        help="Path to JSON file with known face images",
    )
    parser.add_argument(
        "--face-tolerance",
        type=float,
        default=0.6,
        help="Face matching tolerance (default: 0.6)",
    )
    
    # Index building
    parser.add_argument(
        "--build-index-only",
        action="store_true",
        help="Only build vector index from existing metadata",
    )
    parser.add_argument(
        "--metadata-dir",
        help="Directory containing existing metadata (for --build-index-only)",
    )
    parser.add_argument(
        "--index-output",
        help="Output directory for vector index (default: <output>/vector_index)",
    )
    parser.add_argument(
        "--character-filter",
        help="Filter videos by character name",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)",
    )
    
    # General
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if not args.build_index_only and not args.input:
        parser.error("Input path required unless using --build-index-only")
    
    # Run
    if args.build_index_only:
        build_index(args)
    else:
        index_videos(args)
        
        # Optionally build index after indexing
        if not args.skip_embedding:
            build_index(args)


if __name__ == "__main__":
    main()

