#!/usr/bin/env python3
"""
Character Registration Script with Auto-Transcription.

Registers a new character with voice samples, automatically transcribing
audio files using Whisper large-v3, and uploads via the API.

Usage:
    python scripts/register_character.py \
        --name "fred" \
        --description "Host of C'est pas Sorcier" \
        --audio-dir ./samples/fred_voice/ \
        --video ./videos/fred_talking.mp4

    python scripts/register_character.py \
        --name "jamy" \
        --audio-files sample1.wav sample2.wav sample3.wav \
        --video ./videos/jamy.mp4 \
        --whisper-model large-v3
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import requests

from virtual_streamer.utils.transcription import (
    get_whisper_model,
    transcribe_audio as _transcribe_audio,
    get_audio_files,
)


def load_whisper_model(model_name: str = "large-v3"):
    """
    Load stable-whisper with faster-whisper backend for high-quality transcription.
    
    Args:
        model_name: Whisper model size (tiny, base, small, medium, large, large-v3)
        
    Returns:
        Loaded whisper model
    """
    print(f"Loading Whisper model '{model_name}'...")
    model = get_whisper_model(model_name, use_faster=True)
    print(f"✓ Model loaded successfully")
    return model


def transcribe_audio(model, audio_path: str) -> str:
    """
    Transcribe a single audio file to text.
    
    Note: model parameter kept for backward compatibility but is ignored.
    The shared transcription module handles model caching internally.
    
    Args:
        model: Loaded whisper model (ignored, kept for compatibility)
        audio_path: Path to audio file
        
    Returns:
        Transcribed text
    """
    # Use the shared transcription utility with the same model
    return _transcribe_audio(audio_path, model_name="large-v3", use_faster=True)


def register_via_api(
    api_url: str,
    name: str,
    description: Optional[str],
    audio_paths: list[Path],
    transcripts: list[str],
    video_path: Path,
    identity_image_paths: Optional[list[Path]] = None,
    video_search_tag: Optional[str] = None,
) -> dict:
    """
    Register character via POST /characters API.
    
    Args:
        api_url: Base API URL
        name: Character name
        description: Character description
        audio_paths: List of paths to audio files
        transcripts: List of transcripts (one per audio file)
        video_path: Path to representative video file
        identity_image_paths: Optional list of paths to identity images
        video_search_tag: Optional tag for video search filtering (e.g., 'person:fred')
        
    Returns:
        API response as dict
    """
    url = f"{api_url.rstrip('/')}/api/v1/characters"
    
    # Prepare form data
    data = {
        "name": name,
    }
    if description:
        data["description"] = description
    if video_search_tag:
        data["video_search_tag"] = video_search_tag
    
    # Prepare files
    files = []
    
    # Add voice files
    for audio_path in audio_paths:
        files.append(
            ("voice_files", (audio_path.name, open(audio_path, "rb"), "audio/wav"))
        )
    
    # Add transcripts
    for transcript in transcripts:
        files.append(("transcripts", (None, transcript)))
    
    # Add video file
    files.append(
        ("video_file", (video_path.name, open(video_path, "rb"), "video/mp4"))
    )
    
    # Add identity images
    if identity_image_paths:
        for img_path in identity_image_paths:
            content_type = "image/jpeg"
            if img_path.suffix.lower() == ".png":
                content_type = "image/png"
            elif img_path.suffix.lower() == ".webp":
                content_type = "image/webp"
            files.append(
                ("identity_files", (img_path.name, open(img_path, "rb"), content_type))
            )
    
    print(f"\nRegistering character '{name}' via API...")
    print(f"  URL: {url}")
    print(f"  Voice samples: {len(audio_paths)}")
    print(f"  Video: {video_path.name}")
    if identity_image_paths:
        print(f"  Identity images: {len(identity_image_paths)}")
    if video_search_tag:
        print(f"  Video search tag: {video_search_tag}")
    
    response = requests.post(url, data=data, files=files)
    
    # Close file handles
    for _, file_tuple in files:
        if hasattr(file_tuple[1], "close"):
            file_tuple[1].close()
    
    if response.ok:
        print(f"✓ Character '{name}' registered successfully!")
        return response.json()
    else:
        print(f"✗ Failed to register character: {response.status_code}")
        print(f"  Response: {response.text}")
        response.raise_for_status()


def main():
    parser = argparse.ArgumentParser(
        description="Register a character with voice samples (auto-transcribed with Whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register with audio directory
  python scripts/register_character.py \\
      --name "fred" \\
      --audio-dir ./samples/fred_voice/ \\
      --video ./videos/fred_talking.mp4

  # Register with specific audio files
  python scripts/register_character.py \\
      --name "jamy" \\
      --audio-files sample1.wav sample2.wav \\
      --video ./videos/jamy.mp4 \\
      --whisper-model large-v3
        """,
    )
    
    # Identity
    parser.add_argument(
        "--name",
        required=True,
        help="Character name (will also be used as character_id)",
    )
    parser.add_argument(
        "--description",
        help="Character description",
    )
    
    # Audio sources (mutually exclusive)
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument(
        "--audio-dir",
        help="Directory containing voice sample audio files (*.wav, *.mp3)",
    )
    audio_group.add_argument(
        "--audio-files",
        nargs="+",
        help="Specific audio file paths",
    )
    
    # Video
    parser.add_argument(
        "--video",
        required=True,
        help="Path to representative video file for Wav2Lip",
    )
    
    # Identity images
    parser.add_argument(
        "--identity-images",
        nargs="+",
        help="Paths to identity/reference images for the character",
    )
    
    # Video search tag
    parser.add_argument(
        "--video-search-tag",
        help="Tag for video search filtering (e.g., 'person:fred')",
    )
    
    # Transcription settings
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model to use for transcription (default: large-v3)",
    )
    
    # API settings
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    
    args = parser.parse_args()
    
    # Validate video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Get audio files
    try:
        audio_paths = get_audio_files(args.audio_dir, args.audio_files)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Character Registration: {args.name}")
    print(f"{'='*60}")
    print(f"\nFound {len(audio_paths)} audio file(s):")
    for p in audio_paths:
        print(f"  - {p}")
    
    # Load Whisper model
    print()
    model = load_whisper_model(args.whisper_model)
    
    # Transcribe all audio files
    print(f"\nTranscribing {len(audio_paths)} audio file(s)...")
    transcripts = []
    for i, audio_path in enumerate(audio_paths, 1):
        print(f"\n[{i}/{len(audio_paths)}] Transcribing: {audio_path.name}")
        transcript = transcribe_audio(model, str(audio_path))
        transcripts.append(transcript)
        print(f"  → \"{transcript[:80]}{'...' if len(transcript) > 80 else ''}\"")
    
    print(f"\n✓ All {len(audio_paths)} file(s) transcribed")
    
    # Get identity image paths if provided
    identity_image_paths = None
    if args.identity_images:
        identity_image_paths = [Path(f) for f in args.identity_images]
        for p in identity_image_paths:
            if not p.exists():
                print(f"Error: Identity image not found: {p}")
                sys.exit(1)
        print(f"\nFound {len(identity_image_paths)} identity image(s):")
        for p in identity_image_paths:
            print(f"  - {p}")
    
    # Register via API
    try:
        result = register_via_api(
            api_url=args.api_url,
            name=args.name,
            description=args.description,
            audio_paths=audio_paths,
            transcripts=transcripts,
            video_path=video_path,
            identity_image_paths=identity_image_paths,
            video_search_tag=args.video_search_tag,
        )
        
        print(f"\n{'='*60}")
        print("Registration Complete!")
        print(f"{'='*60}")
        print(f"  Character ID: {result.get('character_id')}")
        print(f"  Name: {result.get('name')}")
        print(f"  Voice samples: {len(result.get('voice_samples', []))}")
        print(f"  Video: {result.get('video_clip_path')}")
        if result.get('video_search_tag'):
            print(f"  Video search tag: {result.get('video_search_tag')}")
        if result.get('identity_images'):
            print(f"  Identity images: {len(result.get('identity_images', []))}")
        
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to API at {args.api_url}")
        print("  Make sure the API server is running.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ Error: API request failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

