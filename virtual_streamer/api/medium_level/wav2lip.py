"""
Medium-level API: Wav2Lip Service

Provides lip-sync video generation using Wav2Lip model.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uuid
import os
import shutil
import time

from virtual_streamer.video_server.models import VideoClipBase, VideoOptions
from virtual_streamer.video_server.utils import get_character_data
from virtual_streamer.api.dependencies import get_path_resolver

# Router setup
router = APIRouter(prefix="/wav2lip", tags=["Wav2Lip"])

# Global state (initialized on first use)
_model = None
_detector = None
_detector_model = None
_face_detection_groups: Dict = {}


class Wav2LipRequest(BaseModel):
    """Request model for Wav2Lip generation."""
    audio_path: str  # Path accessible by the server
    video: VideoClipBase
    options: VideoOptions
    character_id: str
    output_dir: Optional[str] = None


class Wav2LipResponse(BaseModel):
    """Response model for Wav2Lip generation."""
    raw_video_path: str  # Path to generated video (no audio)
    processing_time: float


def _init_wav2lip():
    """Initialize Wav2Lip model (lazy loading)."""
    global _model, _detector, _detector_model
    
    if _model is not None:
        return
    
    import torch
    from virtual_streamer.wav2lip.main_logic import Config, do_load
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Initializing Wav2Lip model on {device}...')
    
    args = Config()
    args.checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoints/Wav2Lip.pth")
    
    _model, _detector, _detector_model = do_load(args.checkpoint_path, device)
    print('Wav2Lip model initialized.')


@router.post("/generate", response_model=Wav2LipResponse)
async def generate_wav2lip(payload: Wav2LipRequest):
    """
    Generates lip-synced video using Wav2Lip.
    
    Args:
        payload: Wav2LipRequest with audio path and character info
        
    Returns:
        Wav2LipResponse with generated video path
    """
    # Initialize model on first use
    _init_wav2lip()
    
    print(f"Received Wav2Lip request: {payload}")
    
    character_id = payload.character_id
    audio_path = payload.audio_path
    video_path = payload.video.storage_path
    
    # Validate audio file exists
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"Audio file not found at path: {audio_path}"
        )
    
    # Retrieve character data
    try:
        character = await get_character_data(character_id)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{character_id}' not found: {e}"
        )
    
    # Determine output directory
    if payload.output_dir:
        run_dirname = payload.output_dir
        os.makedirs(run_dirname, exist_ok=True)
    else:
        run_dirname = f"./temp/wav2lip_run_{uuid.uuid4()}"
        os.makedirs(run_dirname, exist_ok=True)
    
    # Resolve video path using path resolver
    path_resolver = get_path_resolver()
    video_path = path_resolver.resolve_video(video_path)
    
    if not os.path.exists(video_path):
        shutil.rmtree(run_dirname, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Video clip not found at path: {video_path}"
        )
    
    # Get or preprocess face detection data
    from virtual_streamer.wav2lip.main_logic import Config, preprocess
    
    args = Config()
    
    if character.name not in _face_detection_groups:
        print(f"Preprocessing face detection for character '{character.name}'...")
        try:
            preprocess(args, video_path, character.name, _detector, _face_detection_groups)
        except Exception as e:
            shutil.rmtree(run_dirname, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"Face preprocessing failed: {e}"
            )
    
    face_det_group = _face_detection_groups[character.name]
    
    # Run Wav2Lip generation
    start_time = time.time()
    
    try:
        from webservice import wav2lip_exec  # Import the existing function
        raw_video_path = wav2lip_exec(run_dirname, audio_path, face_det_group)
    except Exception as e:
        shutil.rmtree(run_dirname, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Wav2Lip generation failed: {e}"
        )
    
    processing_time = time.time() - start_time
    print(f"Wav2Lip generation completed in {processing_time:.2f}s")
    
    return Wav2LipResponse(
        raw_video_path=raw_video_path,
        processing_time=processing_time
    )


@router.get("/health")
async def wav2lip_health():
    """Check Wav2Lip service health."""
    import torch
    
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cached_characters": len(_face_detection_groups)
    }

