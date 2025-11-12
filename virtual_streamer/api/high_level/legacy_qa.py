"""
Legacy API: Q&A Video Generation (from webservice.py)

Provides backward compatibility for the /process endpoint.
This endpoint generates a video response from a question and GPT response.

DEPRECATED: This is maintained for backward compatibility.
New code should use the /video-generation endpoints instead.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
import os
import uuid
import shutil
import datetime

from virtual_streamer.video_server.models import DialogueEntry, VideoClipBase, VideoOptions
from virtual_streamer.api.dependencies import get_character_data
from virtual_streamer.utils.utils import sanitize_str, combine_video_and_audio, add_subtitle, s3_upload, SubtitleMode
from virtual_streamer.api.medium_level.tts import generate_tts
from virtual_streamer.api.medium_level.wav2lip import generate_wav2lip, Wav2LipRequest


# Router setup
router = APIRouter(tags=["Legacy Q&A"])


# --- Pydantic Models ---

class QuestionData(BaseModel):
    """Question data for legacy Q&A endpoint."""
    question: str = ""
    character_name: str = ""
    subtitle_mode: str = "NONE"
    name: str = "User"


class ProcessRequest(BaseModel):
    """Request model for legacy /process endpoint."""
    question: QuestionData
    gpt_response: str


class ProcessResponse(BaseModel):
    """Response model for legacy /process endpoint."""
    video_path: str
    s3_path: Optional[str] = None
    response_text: str


async def qa_process_video(question_data: QuestionData, gpt_response: str) -> Dict[str, Any]:
    """
    Process a Q&A video generation request.
    
    Args:
        question_data: Question information
        gpt_response: GPT response text to convert to video
        
    Returns:
        ProcessResponse with video paths and metadata
    """
    dirname = os.environ.get("OUT_VIDEO_FOLDER", "./out_video_folder")
    os.makedirs(dirname, exist_ok=True)
    temp_dir = os.environ.get("TEMP_DIR", "./temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract data from question model
    question_text = question_data.question
    character_name = question_data.character_name
    subtitle_mode = question_data.subtitle_mode
    name = question_data.name
    
    # Get character data from internal storage
    try:
        character = await get_character_data(character_name)
    except HTTPException as e:
        # Re-raise the HTTPException with additional context
        raise HTTPException(
            status_code=e.status_code,
            detail=f"Failed to fetch character '{character_name}': {e.detail}"
        )
    
    # --- Step 1: Generate TTS audio ---
    try:
        tts_response = await generate_tts(DialogueEntry(
            entry_id=str(uuid.uuid4()),
            character_id=character.character_id,
            text=gpt_response,
            timestamp=0
        ))
        audio_path = tts_response.audio_path
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="TTS call failed to produce audio file.")
    except Exception as e:
        print(f"Error during TTS call in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech generation failed: {e}")
    
    # --- Step 2: Call Wav2Lip endpoint ---
    wav2lip_request_payload = Wav2LipRequest(
        audio_path=os.path.abspath(audio_path),
        video=VideoClipBase(
            storage_path=character.video_clip_path,
            collection_ids=list(),
        ),
        character_id=character.character_id,
        output_dir=None,
        options=VideoOptions(subtitles_enabled=True, subtitle_style=None)
    )
    wav2lip_response = await generate_wav2lip(wav2lip_request_payload)
    raw_video_path = wav2lip_response.raw_video_path
    
    # --- Step 3: Combine video and audio, add subtitles ---
    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(question_text[:30])
    outfile_combined_path = os.path.join(temp_dir, f'result_combined_{tag}.mp4')
    
    try:
        combine_video_and_audio(raw_video_path, audio_path, outfile_combined_path)
    except Exception as e:
        print(f"Error combining video and audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to combine video/audio: {e}")
    
    # Add subtitles if needed
    outfile_titled_path = outfile_combined_path  # Default to combined path
    try:
        if subtitle_mode == "QUESTION":
            subtitle = f"Question de {name} : {question_text}"
            outfile_titled_path = os.path.join(temp_dir, f'result_titled_{tag}.mp4')
            add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
        elif subtitle_mode == "VOICE_SUBTITLE":
            subtitle = gpt_response
            outfile_titled_path = os.path.join(temp_dir, f'result_titled_{tag}.mp4')
            add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
    except Exception as e:
        print(f"Error adding subtitles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add subtitles: {e}")
    
    # --- Step 4: Move the file to final location ---
    final_outfile_path = os.path.join(dirname, f"result_{tag}.mp4")
    try:
        shutil.move(outfile_titled_path, final_outfile_path)
    except Exception as e:
        print(f"Error moving final video file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save final video: {e}")
    
    # --- Step 5: Upload to S3 if configured ---
    s3_path = None
    upload_bucket = os.environ.get("S3_BUCKET_URL", "default-bucket")
    if upload_bucket != "default-bucket":
        try:
            s3_path = s3_upload(final_outfile_path, upload_bucket)
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            # Non-critical, just log and continue
            pass
    
    # --- Step 6: Cleanup temporary files ---
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        # Clean up wav2lip temp directory if it was created
        if raw_video_path:
            raw_video_dir = os.path.dirname(raw_video_path)
            if raw_video_dir.startswith(os.path.abspath(temp_dir)):
                shutil.rmtree(raw_video_dir, ignore_errors=True)
        # Clean up combined video if different from titled
        if outfile_combined_path != outfile_titled_path and os.path.exists(outfile_combined_path):
            os.remove(outfile_combined_path)
    except OSError as e:
        print(f"Warning: Error during temporary file cleanup: {e}")
    
    # --- Step 7: Return response ---
    return ProcessResponse(
        video_path=final_outfile_path,
        s3_path=s3_path,
        response_text=gpt_response
    )


@router.post("/process", response_model=ProcessResponse)
async def single_qa_video_process(payload: ProcessRequest):
    """
    Legacy endpoint for Q&A video generation.
    
    DEPRECATED: Use /api/v1/video-generation/submit instead for new applications.
    
    Generates a video response to a question using the specified character.
    This endpoint processes the request synchronously and returns the video path.
    
    Args:
        payload: ProcessRequest with question and GPT response
        
    Returns:
        ProcessResponse with video path and metadata
    """
    print(f"[LEGACY] Received /process request: {payload}")
    
    # Extract data from request model
    question_data = payload.question
    gpt_response = payload.gpt_response
    
    # Process the video
    result = await qa_process_video(question_data, gpt_response)
    
    return result

