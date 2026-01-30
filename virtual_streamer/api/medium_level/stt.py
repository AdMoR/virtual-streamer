"""
Medium-level API: Speech-to-Text Service

Provides STT transcription using Whisper models.
"""
import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from virtual_streamer.utils.transcription import get_whisper_model

# Router setup
router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])

logger = logging.getLogger(__name__)


class STTResponse(BaseModel):
    """Response model for STT transcription."""

    text: str
    srt_path: str = None


@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Transcribes audio file to text using Whisper.

    Args:
        audio_file: Audio file to transcribe

    Returns:
        STTResponse with transcribed text
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Use cached Whisper model
        model = get_whisper_model("base", use_faster=False)

        # Transcribe
        result = model.transcribe(tmp_path)
        text = result.text

        return STTResponse(text=text)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"STT transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/transcribe-to-srt", response_model=STTResponse)
async def transcribe_to_srt(audio_file: UploadFile = File(...)):
    """
    Transcribes audio file to SRT subtitle format.

    Args:
        audio_file: Audio file to transcribe

    Returns:
        STTResponse with SRT file path
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Use cached Whisper model
        model = get_whisper_model("base", use_faster=False)

        # Transcribe
        result = model.transcribe(tmp_path)

        # Generate SRT file
        temp_dir = os.environ.get("TEMP_DIR", "./temp")
        os.makedirs(temp_dir, exist_ok=True)

        srt_path = os.path.join(temp_dir, f"subtitle_{os.path.basename(tmp_path)}.srt")
        result.to_srt_vtt(srt_path, word_level=False)

        text = result.text

        return STTResponse(text=text, srt_path=srt_path)

    except Exception as e:
        logger.error(f"STT transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"STT transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temp audio file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
