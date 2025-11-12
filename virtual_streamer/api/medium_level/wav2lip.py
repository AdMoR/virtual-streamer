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
import datetime
import subprocess
import numpy as np
import cv2
import torch
from tqdm import tqdm

from virtual_streamer.video_server.models import VideoClipBase, VideoOptions
from virtual_streamer.api.dependencies import get_path_resolver, get_character_data
from virtual_streamer.wav2lip import audio
from virtual_streamer.wav2lip.main_logic import Config, do_load, preprocess, datagen, FaceDetectionGroup
from virtual_streamer.utils.utils import sanitize_str

# Router setup
router = APIRouter(prefix="/wav2lip", tags=["Wav2Lip"])

# Global state (initialized on first use)
_model = None
_detector = None
_detector_model = None
_device = None
_args = None
_mel_step_size = 16
_face_detection_groups: Dict[str, FaceDetectionGroup] = {}


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
    global _model, _detector, _detector_model, _device, _args
    
    if _model is not None:
        return
    
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Initializing Wav2Lip model on {_device}...')
    
    _args = Config()
    _args.checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoints/Wav2Lip.pth")
    
    _model, _detector, _detector_model = do_load(_args.checkpoint_path, _device)
    print('Wav2Lip model initialized.')


def wav2lip_exec(dirname: str, audio_path: str, det_results: FaceDetectionGroup) -> str:
    """
    Execute Wav2Lip generation on preprocessed face detection data.
    
    Args:
        dirname: Output directory for the generated video
        audio_path: Path to the audio file
        det_results: Face detection group containing frames and detection results
        
    Returns:
        Path to the generated video file
    """
    global _model, _device, _args, _mel_step_size
    
    # Ensure model is initialized
    if _model is None:
        raise RuntimeError("Wav2Lip model not initialized. Call _init_wav2lip() first.")
    
    fps = 24
    
    full_frames = det_results.full_frames
    face_det_results = det_results.face_det_results
    
    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(os.path.basename(audio_path))
    out_path = f'{dirname}/result.avi'
    batch_size = _args.wav2lip_batch_size
    
    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        subprocess.check_call([
            "ffmpeg", "-y",
            "-i", audio_path,
            f"{dirname}/temp.wav",
        ])
        audio_path = f'{dirname}/temp.wav'
    
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    
    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + _mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - _mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + _mel_step_size])
        i += 1
    
    print(f"Length of mel chunks: {len(mel_chunks)}, length of frames {len(full_frames)}")
    
    full_frames = full_frames[:len(mel_chunks)]
    face_det_results = face_det_results[:len(mel_chunks)]
    gen = datagen(_args, full_frames.copy(), mel_chunks, face_det_results.copy())
    
    for i, rez in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        print(i)
        (face_img_batch, mel_batch, frames, coords) = rez
        
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(f'{dirname}/result.avi',
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        print("-")
        face_img_batch = torch.FloatTensor(np.transpose(face_img_batch, (0, 3, 1, 2))).to(_device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(_device)
        print("--")
        with torch.no_grad():
            pred = _model(mel_batch, face_img_batch)
        print("---")
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)
    
    out.release()
    print("wav2lip_exec Done")
    return out_path


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
    if character.name not in _face_detection_groups:
        print(f"Preprocessing face detection for character '{character.name}'...")
        try:
            preprocess(_args, video_path, character.name, _detector, _face_detection_groups)
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
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": _device if _device else ("cuda" if torch.cuda.is_available() else "cpu"),
        "cached_characters": len(_face_detection_groups)
    }

