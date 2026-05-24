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

from virtual_streamer.video_server.models import DialogueEntry
from virtual_streamer.utils.character_loader import load_character
from virtual_streamer.api.medium_level.tts import generate_tts


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


async def qa_process_video(
    question_data: QuestionData, gpt_response: str
) -> Dict[str, Any]:
    """
    Process a Q&A video generation request.

    Args:
        question_data: Question information
        gpt_response: GPT response text to convert to video

    Returns:
        ProcessResponse with video paths and metadata
    """
    # Extract data from question model
    character_name = question_data.character_name

    # Get character data from internal storage
    try:
        character = await load_character(character_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching character '{character_name}': {e}",
        )

    # --- Step 1: Generate TTS audio ---
    try:
        tts_response = await generate_tts(
            DialogueEntry(
                entry_id=str(uuid.uuid4()),
                character_id=character.character_id,
                text=gpt_response,
                timestamp=0,
            )
        )
        audio_path = tts_response.audio_path
        if not os.path.exists(audio_path):
            raise HTTPException(
                status_code=500, detail="TTS call failed to produce audio file."
            )
    except Exception as e:
        print(f"Error during TTS call in /process: {e}")
        raise HTTPException(
            status_code=500, detail=f"Text-to-speech generation failed: {e}"
        )

    # Lip-sync (wav2lip) has been removed from the codebase.
    raise HTTPException(
        status_code=501,
        detail="This legacy endpoint requires lip-sync video generation which has been removed. "
               "Use the story pipeline endpoints instead.",
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
