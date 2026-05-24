#!/usr/bin/env python3
"""
Rebuild Video from Blueprint Dump

This script downloads debug artifacts from MinIO and recombines them into a final video
without regenerating any content. It uses the existing TTS audio and subtitles stored
during a previous generation run.

Usage:
    python scripts/rebuild_from_blueprint.py --debug-path debug/video-generation/template_id/job_id
    python scripts/rebuild_from_blueprint.py --debug-path debug/video-generation/template_id/job_id --output ./my_video.mp4

The script expects the following structure in MinIO:
    {debug_path}/
    ├── blueprint.json
    ├── tts/
    │   ├── segment_0.wav
    │   └── ...
    └── subtitles/
        ├── segment_0.srt
        └── ...
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from virtual_streamer.utils.minio_client import MinIOClient
from virtual_streamer.utils.utils import (
    combine_video_and_short_audio,
    add_subtitle_from_srt,
    combine_part_in_concat_file,
)


async def download_artifacts(
    client: MinIOClient,
    debug_path: str,
    temp_dir: str,
) -> dict:
    """
    Download all artifacts from MinIO.
    
    Args:
        client: MinIO client instance
        debug_path: Path prefix in MinIO (e.g., debug/video-generation/template/job_id)
        temp_dir: Local directory to download artifacts to
        
    Returns:
        Dictionary with paths to downloaded files organized by type
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create subdirectories
    tts_dir = os.path.join(temp_dir, "tts")
    subtitles_dir = os.path.join(temp_dir, "subtitles")

    for d in [tts_dir, subtitles_dir]:
        os.makedirs(d, exist_ok=True)

    artifacts = {
        "blueprint": None,
        "tts": {},
        "subtitles": {},
    }
    
    # Download blueprint.json
    blueprint_key = f"{debug_path}/blueprint.json"
    blueprint_local = os.path.join(temp_dir, "blueprint.json")
    print(f"Downloading blueprint: {blueprint_key}")
    try:
        await client.download_file(blueprint_key, blueprint_local)
        artifacts["blueprint"] = blueprint_local
    except Exception as e:
        print(f"Error downloading blueprint: {e}")
        raise RuntimeError(f"Blueprint not found at {blueprint_key}")
    
    # Load blueprint to know how many segments to expect
    with open(blueprint_local, "r") as f:
        blueprint = json.load(f)
    
    num_segments = len(blueprint.get("planned_tts", []))
    print(f"Blueprint indicates {num_segments} segments")
    
    # Download artifacts for each segment
    for i in range(num_segments):
        # TTS audio
        tts_key = f"{debug_path}/tts/segment_{i}.wav"
        tts_local = os.path.join(tts_dir, f"segment_{i}.wav")
        try:
            print(f"  Downloading TTS segment {i}...")
            await client.download_file(tts_key, tts_local)
            artifacts["tts"][i] = tts_local
        except Exception as e:
            print(f"  Warning: Could not download TTS segment {i}: {e}")

        # Subtitles
        srt_key = f"{debug_path}/subtitles/segment_{i}.srt"
        srt_local = os.path.join(subtitles_dir, f"segment_{i}.srt")
        try:
            print(f"  Downloading subtitles segment {i}...")
            await client.download_file(srt_key, srt_local)
            artifacts["subtitles"][i] = srt_local
        except Exception as e:
            print(f"  Warning: Could not download subtitles segment {i}: {e}")
    
    return artifacts


def rebuild_video(
    artifacts: dict,
    output_path: str,
    temp_dir: str,
    fontsize: int = 14,
) -> str:
    """
    Rebuild video from downloaded artifacts.
    
    Args:
        artifacts: Dictionary with paths to downloaded files
        output_path: Path for the final output video
        temp_dir: Temp directory for intermediate files
        fontsize: Font size for subtitles
        
    Returns:
        Path to the final video
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Create directory for combined segments
    combined_dir = os.path.join(temp_dir, "combined_rebuild")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Get segment indices (sorted from TTS artifacts)
    segment_indices = sorted(artifacts["tts"].keys())

    if not segment_indices:
        raise RuntimeError("No TTS segments found in artifacts")

    print(f"\nRebuilding {len(segment_indices)} segments...")

    video_segments = []

    for i in segment_indices:
        print(f"\n  Processing segment {i}...")

        tts_path = artifacts["tts"].get(i)
        srt_path = artifacts["subtitles"].get(i)

        if not tts_path or not os.path.exists(tts_path):
            print(f"    Skipping segment {i}: TTS audio not found")
            continue

        # NOTE: Without lip-sync video, we cannot combine video + audio here.
        # The blueprint must include video clip paths for a full rebuild.
        # For now, copy TTS audio path as a placeholder so subtitles can be applied.
        combined_path = tts_path  # placeholder — no video to combine with
        print(f"    (No video clip available for segment {i}, using audio only)")
        
        # Step 2: Add subtitles if available
        if srt_path and os.path.exists(srt_path):
            segment_path = os.path.join(combined_dir, f"segment_{i}.mp4")
            print(f"    Adding subtitles -> {segment_path}")
            add_subtitle_from_srt(
                combined_path,
                srt_path,
                segment_path,
                fontsize=fontsize,
            )
            video_segments.append(segment_path)
        else:
            print(f"    No subtitles found, using combined video")
            video_segments.append(combined_path)
        
        print(f"    Segment {i} complete")
    
    if not video_segments:
        raise RuntimeError("No segments were successfully processed")
    
    # Concatenate all segments
    print(f"\nConcatenating {len(video_segments)} segments...")
    concat_file = os.path.join(temp_dir, "concat_rebuild.txt")
    combine_part_in_concat_file(video_segments, concat_file, output_path)
    
    print(f"\nFinal video created: {output_path}")
    return output_path


async def main():
    parser = argparse.ArgumentParser(
        description="Rebuild video from blueprint dump stored in MinIO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --debug-path debug/video-generation/cest_pas_sorcier/abc123
  %(prog)s --debug-path debug/video-generation/template/job --output ./rebuilt.mp4
  %(prog)s --debug-path debug/video-generation/template/job --fontsize 18
        """,
    )
    
    parser.add_argument(
        "--debug-path",
        required=True,
        help="MinIO path prefix containing the blueprint and artifacts "
             "(e.g., debug/video-generation/template_id/job_id)",
    )
    parser.add_argument(
        "--output",
        default="./output/rebuilt_video.mp4",
        help="Output video path (default: ./output/rebuilt_video.mp4)",
    )
    parser.add_argument(
        "--temp-dir",
        default="./temp/rebuild",
        help="Temp directory for downloads and intermediate files (default: ./temp/rebuild)",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=14,
        help="Font size for subtitles (default: 14)",
    )
    parser.add_argument(
        "--minio-endpoint",
        default="http://localhost:9000",
        help="MinIO endpoint URL (default: from MINIO_ENDPOINT env var)",
    )
    parser.add_argument(
        "--minio-bucket",
        default=None,
        help="MinIO bucket name (default: from MINIO_BUCKET env var)",
    )
    
    args = parser.parse_args()
    
    # Initialize MinIO client
    client = MinIOClient(
        endpoint=args.minio_endpoint,
        bucket=args.minio_bucket,
    )
    
    print(f"MinIO endpoint: {client.endpoint}")
    print(f"MinIO bucket: {client.bucket}")
    print(f"Debug path: {args.debug_path}")
    print(f"Output: {args.output}")
    print()
    
    # Download artifacts
    print("=" * 60)
    print("Downloading artifacts from MinIO...")
    print("=" * 60)
    artifacts = await download_artifacts(
        client=client,
        debug_path=args.debug_path,
        temp_dir=args.temp_dir,
    )
    
    # Print blueprint summary
    if artifacts["blueprint"]:
        with open(artifacts["blueprint"], "r") as f:
            blueprint = json.load(f)
        print(f"\nBlueprint Summary:")
        print(f"  Job ID: {blueprint.get('job_id', 'N/A')}")
        print(f"  Template: {blueprint.get('story_template_id', 'N/A')}")
        print(f"  Timestamp: {blueprint.get('timestamp', 'N/A')}")
        if "story_output" in blueprint:
            print(f"  Story Title: {blueprint['story_output'].get('title', 'N/A')}")
    
    print(f"\nDownloaded artifacts:")
    print(f"  TTS segments: {len(artifacts['tts'])}")
    print(f"  Subtitle segments: {len(artifacts['subtitles'])}")
    
    # Rebuild video
    print()
    print("=" * 60)
    print("Rebuilding video from artifacts...")
    print("=" * 60)
    
    final_path = rebuild_video(
        artifacts=artifacts,
        output_path=args.output,
        temp_dir=args.temp_dir,
        fontsize=args.fontsize,
    )
    
    print()
    print("=" * 60)
    print(f"SUCCESS! Video rebuilt at: {final_path}")
    print("=" * 60)
    
    return final_path


if __name__ == "__main__":
    asyncio.run(main())
