"""
Shared transcription utilities.

Provides cached Whisper model loading and transcription functions
to avoid duplicate implementations across the codebase.
"""

import functools
import os
from pathlib import Path
from typing import List, Optional, Union

import stable_whisper


@functools.lru_cache(maxsize=4)
def get_whisper_model(model_name: str = "large-v3", use_faster: bool = True):
    """
    Get or create a cached Whisper model.
    
    Models are cached to avoid reloading on every transcription call.
    The cache holds up to 4 different model configurations.
    
    Args:
        model_name: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
        use_faster: If True, use faster-whisper backend; otherwise use standard whisper
        
    Returns:
        Loaded whisper model
    """
    if use_faster:
        return stable_whisper.load_faster_whisper(model_name)
    return stable_whisper.load_model(model_name)


def transcribe_audio(
    audio_path: str,
    model_name: str = "large-v3",
    use_faster: bool = True,
) -> str:
    """
    Transcribe audio file to text.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
        use_faster: If True, use faster-whisper backend
        
    Returns:
        Transcribed text
    """
    model = get_whisper_model(model_name, use_faster=use_faster)
    result = model.transcribe(audio_path)
    return result.text.strip()


def transcribe_to_srt(
    audio_path: str,
    srt_path: str,
    model_name: str = "base",
    use_faster: bool = False,
    word_level: bool = False,
) -> str:
    """
    Transcribe audio and save as SRT subtitle file.
    
    Args:
        audio_path: Path to audio file
        srt_path: Path to save SRT file
        model_name: Whisper model size
        use_faster: If True, use faster-whisper backend
        word_level: If True, generate word-level timing
        
    Returns:
        Path to the generated SRT file
    """
    model = get_whisper_model(model_name, use_faster=use_faster)
    result = model.transcribe(audio_path)
    result.to_srt_vtt(srt_path, word_level=word_level)
    return srt_path


def get_audio_files(
    audio_dir: Optional[str] = None,
    audio_files: Optional[List[str]] = None,
) -> List[Path]:
    """
    Get list of audio files from directory or explicit file list.
    
    Args:
        audio_dir: Directory containing audio files (*.wav, *.mp3)
        audio_files: List of specific audio file paths
        
    Returns:
        List of Path objects to audio files
        
    Raises:
        FileNotFoundError: If audio directory or files not found
        ValueError: If neither audio_dir nor audio_files is provided
    """
    if audio_dir:
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        # Get all wav/mp3 files
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
    
    else:
        raise ValueError("Either audio_dir or audio_files must be provided")


def clear_model_cache():
    """Clear the cached Whisper models to free memory."""
    get_whisper_model.cache_clear()
