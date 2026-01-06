#!/usr/bin/env python3
"""
Character Update Script.

Updates an existing character's metadata, voice samples, video, or identity images.

Usage:
    # Update description only
    python scripts/update_character.py fred --description "Updated description"

    # Update voice samples (replaces all existing)
    python scripts/update_character.py fred \
        --audio-dir ./new_samples/ \
        --whisper-model large-v3

    # Update video file
    python scripts/update_character.py fred --video ./new_video.mp4

    # Update multiple fields
    python scripts/update_character.py fred \
        --name "Fred Updated" \
        --description "New description" \
        --video-search-tag "person:fred_new"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import requests


def load_whisper_model(model_name: str = "large-v3"):
    """Load stable-whisper with faster-whisper backend for transcription."""
    import stable_whisper
    
    print(f"Loading Whisper model '{model_name}'...")
    model = stable_whisper.load_faster_whisper(model_name)
    print(f"✓ Model loaded successfully")
    return model


def transcribe_audio(model, audio_path: str) -> str:
    """Transcribe a single audio file to text."""
    result = model.transcribe(audio_path)
    return result.text.strip()


def get_audio_files(audio_dir: Optional[str], audio_files: Optional[list[str]]) -> list[Path]:
    """Get list of audio files from directory or explicit file list."""
    if audio_dir:
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        files = list(audio_path.glob("*.wav")) + list(audio_path.glob("*.mp3"))
        if not files:
            raise ValueError(f"No audio files (*.wav, *.mp3) found in {audio_dir}")
        
        return sorted(files)
    
    elif audio_files:
        paths = [Path(f) for f in audio_files]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Audio file not found: {p}")
        return paths
    
    return []


def get_character(api_url: str, character_id: str) -> Optional[dict]:
    """Get current character data from API."""
    url = f"{api_url.rstrip('/')}/api/v1/characters/{character_id}"
    response = requests.get(url)
    if response.ok:
        return response.json()
    return None


def update_via_api(
    api_url: str,
    character_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    video_search_tag: Optional[str] = None,
    audio_paths: Optional[list[Path]] = None,
    transcripts: Optional[list[str]] = None,
    video_path: Optional[Path] = None,
    identity_image_paths: Optional[list[Path]] = None,
) -> dict:
    """
    Update character via PUT /characters/{id} API.
    """
    url = f"{api_url.rstrip('/')}/api/v1/characters/{character_id}"
    
    # Prepare form data (only include non-None values)
    data = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    if video_search_tag is not None:
        data["video_search_tag"] = video_search_tag
    
    # Prepare files
    files = []
    
    # Add voice files if provided
    if audio_paths and transcripts:
        for audio_path in audio_paths:
            files.append(
                ("voice_files", (audio_path.name, open(audio_path, "rb"), "audio/wav"))
            )
        for transcript in transcripts:
            files.append(("transcripts", (None, transcript)))
    
    # Add video file if provided
    if video_path:
        files.append(
            ("video_file", (video_path.name, open(video_path, "rb"), "video/mp4"))
        )
    
    # Add identity images if provided
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
    
    print(f"\nUpdating character '{character_id}' via API...")
    print(f"  URL: {url}")
    if data:
        print(f"  Fields to update: {', '.join(data.keys())}")
    if audio_paths:
        print(f"  New voice samples: {len(audio_paths)}")
    if video_path:
        print(f"  New video: {video_path.name}")
    if identity_image_paths:
        print(f"  New identity images: {len(identity_image_paths)}")
    
    response = requests.put(url, data=data if data else None, files=files if files else None)
    
    # Close file handles
    for _, file_tuple in files:
        if hasattr(file_tuple[1], "close"):
            file_tuple[1].close()
    
    if response.ok:
        print(f"✓ Character '{character_id}' updated successfully!")
        return response.json()
    else:
        print(f"✗ Failed to update character: {response.status_code}")
        print(f"  Response: {response.text}")
        response.raise_for_status()


def main():
    parser = argparse.ArgumentParser(
        description="Update an existing character",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update description
  python scripts/update_character.py fred --description "New description"

  # Update voice samples with auto-transcription
  python scripts/update_character.py fred \\
      --audio-dir ./new_samples/ \\
      --whisper-model large-v3

  # Update video file
  python scripts/update_character.py fred --video ./new_video.mp4

  # Update multiple fields
  python scripts/update_character.py fred \\
      --name "Fred Updated" \\
      --description "New description" \\
      --video-search-tag "person:fred_new"
        """,
    )
    
    # Required: character ID
    parser.add_argument(
        "character_id",
        help="ID of the character to update",
    )
    
    # Optional updates
    parser.add_argument(
        "--name",
        help="New display name",
    )
    parser.add_argument(
        "--description",
        help="New description",
    )
    parser.add_argument(
        "--video-search-tag",
        help="New video search tag (e.g., 'person:fred')",
    )
    
    # Audio sources (mutually exclusive)
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument(
        "--audio-dir",
        help="Directory containing new voice sample files (replaces all existing)",
    )
    audio_group.add_argument(
        "--audio-files",
        nargs="+",
        help="Specific new audio file paths (replaces all existing)",
    )
    
    # Video
    parser.add_argument(
        "--video",
        help="Path to new video file",
    )
    
    # Identity images
    parser.add_argument(
        "--identity-images",
        nargs="+",
        help="Paths to new identity images (replaces all existing)",
    )
    
    # Transcription settings
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model for transcription (default: large-v3)",
    )
    
    # API settings
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    
    args = parser.parse_args()
    
    # Check if at least one update is specified
    has_update = any([
        args.name,
        args.description,
        args.video_search_tag,
        args.audio_dir,
        args.audio_files,
        args.video,
        args.identity_images,
    ])
    
    if not has_update:
        print("Error: No updates specified. Use --help to see available options.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Character Update: {args.character_id}")
    print(f"{'='*60}")
    
    # Check if character exists
    existing = get_character(args.api_url, args.character_id)
    if existing is None:
        print(f"\n✗ Character '{args.character_id}' not found")
        sys.exit(1)
    
    print(f"\nCurrent character:")
    print(f"  Name: {existing.get('name')}")
    print(f"  Description: {existing.get('description', 'N/A')[:50]}...")
    print(f"  Voice samples: {len(existing.get('voice_samples', []))}")
    
    # Process audio files if provided
    audio_paths = None
    transcripts = None
    if args.audio_dir or args.audio_files:
        try:
            audio_paths = get_audio_files(args.audio_dir, args.audio_files)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        print(f"\nFound {len(audio_paths)} audio file(s) to transcribe:")
        for p in audio_paths:
            print(f"  - {p}")
        
        # Load Whisper and transcribe
        model = load_whisper_model(args.whisper_model)
        
        print(f"\nTranscribing {len(audio_paths)} audio file(s)...")
        transcripts = []
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}] Transcribing: {audio_path.name}")
            transcript = transcribe_audio(model, str(audio_path))
            transcripts.append(transcript)
            print(f"  → \"{transcript[:80]}{'...' if len(transcript) > 80 else ''}\"")
        
        print(f"\n✓ All {len(audio_paths)} file(s) transcribed")
    
    # Validate video file
    video_path = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
    
    # Validate identity images
    identity_image_paths = None
    if args.identity_images:
        identity_image_paths = [Path(f) for f in args.identity_images]
        for p in identity_image_paths:
            if not p.exists():
                print(f"Error: Identity image not found: {p}")
                sys.exit(1)
    
    # Perform update
    try:
        result = update_via_api(
            api_url=args.api_url,
            character_id=args.character_id,
            name=args.name,
            description=args.description,
            video_search_tag=args.video_search_tag,
            audio_paths=audio_paths,
            transcripts=transcripts,
            video_path=video_path,
            identity_image_paths=identity_image_paths,
        )
        
        print(f"\n{'='*60}")
        print("Update Complete!")
        print(f"{'='*60}")
        print(f"  Character ID: {result.get('character_id')}")
        print(f"  Name: {result.get('name')}")
        print(f"  Description: {result.get('description', 'N/A')[:50]}...")
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

