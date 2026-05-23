#!/usr/bin/env python3
"""
Add subtitles to an existing video using Whisper large-v3 for transcription.

Usage:
    python scripts/add_subtitles.py input.mp4
    python scripts/add_subtitles.py input.mp4 --output output.mp4
    python scripts/add_subtitles.py input.mp4 --fontsize 18 --word-level
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from virtual_streamer.utils.transcription import transcribe_to_srt
from virtual_streamer.utils.utils import add_subtitle_from_srt


def extract_audio(video_path: str, audio_path: str) -> None:
    args = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr.decode()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add subtitles to a video via Whisper transcription")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--output", help="Path to output video (default: input_subtitled.mp4)")
    parser.add_argument("--fontsize", type=int, default=14, help="Subtitle font size (default: 14)")
    parser.add_argument("--word-level", action="store_true", help="Word-level subtitle timing")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).resolve() if args.output else input_path.with_stem(input_path.stem + "_subtitled")

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = str(Path(tmp) / "audio.wav")
        srt_path = str(Path(tmp) / "subtitles.srt")

        print("Extracting audio...")
        extract_audio(str(input_path), audio_path)

        print("Transcribing with Whisper large-v3 (this may take a while)...")
        transcribe_to_srt(
            audio_path=audio_path,
            srt_path=srt_path,
            model_name="large-v3",
            use_faster=True,
            word_level=args.word_level,
        )

        print("Burning subtitles into video...")
        add_subtitle_from_srt(
            video_path=str(input_path),
            srt_path=srt_path,
            output_path=str(output_path),
            fontsize=args.fontsize,
        )

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
