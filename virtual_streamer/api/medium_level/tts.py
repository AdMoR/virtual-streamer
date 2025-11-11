"""
Medium-level API: Text-to-Speech Service

Provides TTS generation using various backends (Fish-Speech, etc.)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import os

from virtual_streamer.video_server.models import DialogueEntry, Character
from virtual_streamer.utils.utils import txt_to_speech_call_fish
from virtual_streamer.video_server.utils import get_character_data
from virtual_streamer.api.dependencies import get_path_resolver

# Router setup
router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])


class TTSResponse(BaseModel):
    """Response model for TTS generation."""
    entry_id: str
    audio_path: str  # Path accessible by subsequent services
    duration: float = 0.0


@router.post("/generate", response_model=TTSResponse)
async def generate_tts(payload: DialogueEntry):
    """
    Generates Text-to-Speech audio for a given dialogue entry.
    
    Args:
        payload: DialogueEntry with character_id and text
        
    Returns:
        TTSResponse with audio file path
    """
    print(f"Received TTS generation request for entry: {payload.entry_id}")
    
    # Fetch character data to get voice configuration
    try:
        character: Character = await get_character_data(payload.character_id)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{payload.character_id}' not found: {e}"
        )
    
    # Generate unique filename in temp directory
    temp_dir = os.environ.get("TEMP_DIR", "./temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_filename = f"tts_{payload.entry_id}_{uuid.uuid4()}.wav"
    audio_outpath = os.path.join(temp_dir, audio_filename)
    
    print(f"Generating TTS for entry {payload.entry_id} with character {character.character_id}...")
    
    # Prepare TTS parameters
    tts_params = {}
    if character.voice_samples and len(character.voice_samples) > 0:
        # Use first voice sample for voice cloning
        first_sample = character.voice_samples[0]
        
        # Resolve the reference audio path using path resolver
        path_resolver = get_path_resolver()
        reference_audio_path = path_resolver.resolve_audio(first_sample.sample_storage_path)
        
        if not path_resolver.exists(first_sample.sample_storage_path):
            print(f"Warning: Reference audio not found at {reference_audio_path}")
        
        tts_params["reference_audio"] = reference_audio_path
        tts_params["reference_text"] = first_sample.transcript
        print(f"Using voice cloning with: {reference_audio_path}")
    
    # Generate TTS
    try:
        txt_to_speech_call_fish(
            speech_lines=payload.text,
            outpath=audio_outpath,
            format="wav",
            **tts_params
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {str(e)}"
        )
    
    if not os.path.exists(audio_outpath):
        raise HTTPException(
            status_code=500,
            detail="TTS call failed to produce audio file"
        )
    
    print(f"TTS audio generated successfully at: {audio_outpath}")
    
    # Get audio duration
    from virtual_streamer.utils.utils import get_length
    duration = get_length(audio_outpath)
    
    return TTSResponse(
        entry_id=payload.entry_id,
        audio_path=os.path.abspath(audio_outpath),
        duration=duration
    )

