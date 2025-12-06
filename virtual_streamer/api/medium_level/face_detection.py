"""
Medium-level API: Face Detection Service

Provides face detection and preprocessing for video files.
Preprocessing results are cached to S3/local storage for efficiency.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import time
import hashlib
import pickle

from virtual_streamer.wav2lip.main_logic import preprocess, FaceDetectionGroup, Config
from virtual_streamer.api.dependencies import get_path_resolver, get_storage_client

# Router setup
router = APIRouter(prefix="/face-detection", tags=["Face Detection"])

# Global state
_detector = None
_device = None

# Cache configuration
FACE_CACHE_PREFIX = "cache/face_detection/"


class BoundingBox(BaseModel):
    """Face bounding box coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class DetectedFace(BaseModel):
    """A detected face with metadata."""
    frame_index: int
    bounding_box: BoundingBox


class FaceDetectionRequest(BaseModel):
    """Request to detect faces in video/image."""
    source_path: str = Field(..., description="Path to video or image file")
    sample_rate: int = Field(1, description="Process every N-th frame (for video)")
    min_confidence: float = Field(0.5, description="Minimum detection confidence")


class FaceDetectionResponse(BaseModel):
    """Response with detected faces."""
    source_path: str
    total_frames: int
    faces: List[DetectedFace]
    processing_time_seconds: float


class PreprocessRequest(BaseModel):
    """Request to preprocess faces for Wav2Lip caching."""
    video_path: str = Field(..., description="Path to video file")
    character_id: str = Field(..., description="Character identifier for cache key")


class PreprocessResponse(BaseModel):
    """Response with preprocessed face data info."""
    character_id: str
    video_path: str
    cache_path: str
    frame_count: int
    status: str


class FaceDetectionHealthResponse(BaseModel):
    """Health check response."""
    status: str
    device: str
    model_loaded: bool
    cached_videos: int


def _init_detector():
    """Initialize face detector (lazy loading)."""
    global _detector, _device
    
    if _detector is not None:
        return
    
    import torch
    from virtual_streamer.wav2lip.main_logic import do_load, Config
    
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Initializing face detector on {_device}...')
    
    args = Config()
    args.checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoints/Wav2Lip.pth")
    
    # Load just the detector from wav2lip
    _, _detector, _ = do_load(args.checkpoint_path, _device)
    print('Face detector initialized.')


def _get_cache_path(video_path: str) -> str:
    """Get cache path for a video."""
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    return f"{FACE_CACHE_PREFIX}{video_hash}.pkl"


def _get_full_cache_path(cache_key: str) -> str:
    """Get full filesystem path for cache key."""
    data_dir = os.environ.get("DATA_DIR", "/data")
    return os.path.join(data_dir, cache_key)


async def _load_from_cache(video_path: str) -> Optional[dict]:
    """Load preprocessed data from cache."""
    cache_key = _get_cache_path(video_path)
    cache_path = _get_full_cache_path(cache_key)
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
    
    return None


async def _save_to_cache(video_path: str, data: dict) -> str:
    """Save preprocessed data to cache. Returns cache path."""
    cache_key = _get_cache_path(video_path)
    cache_path = _get_full_cache_path(cache_key)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    return cache_key


@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(request: FaceDetectionRequest):
    """
    Detect faces in a video or image file.
    
    Returns list of detected faces with bounding boxes.
    """
    _init_detector()
    
    # Resolve path
    path_resolver = get_path_resolver()
    source_path = path_resolver.resolve(request.source_path)
    
    if not os.path.exists(source_path):
        raise HTTPException(
            status_code=400,
            detail=f"Source file not found: {request.source_path}"
        )
    
    import cv2
    
    start_time = time.time()
    faces: List[DetectedFace] = []
    total_frames = 0
    
    try:
        # Open video
        cap = cv2.VideoCapture(source_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            
            # Sample frames
            if frame_idx % request.sample_rate != 0:
                frame_idx += 1
                continue
            
            # Detect faces using the detector
            # Note: Using batch_face or similar detector

            import batch_face
            face_detector = batch_face.RetinaFace(gpu_id=0 if _device == 'cuda' else -1)
            detections = face_detector(frame)

            for det in detections:
                if len(det) >= 5:
                    x1, y1, x2, y2, conf = det[:5]
                    if conf >= request.min_confidence:
                        faces.append(DetectedFace(
                            frame_index=frame_idx,
                            bounding_box=BoundingBox(
                                x1=int(x1),
                                y1=int(y1),
                                x2=int(x2),
                                y2=int(y2),
                                confidence=float(conf)
                            )
                        ))

        
        cap.release()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Face detection failed: {str(e)}"
        )
    
    processing_time = time.time() - start_time
    
    return FaceDetectionResponse(
        source_path=request.source_path,
        total_frames=total_frames,
        faces=faces,
        processing_time_seconds=processing_time
    )


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_for_wav2lip(request: PreprocessRequest):
    """
    Preprocess and cache face detection for Wav2Lip.
    
    This extracts faces from all frames and caches the results to S3/local storage,
    so Wav2Lip doesn't need to re-detect faces each time.
    
    The cache is stored in a NoSQL-like format (pickled dict) on the storage backend.
    """
    _init_detector()
    
    # Resolve path
    path_resolver = get_path_resolver()
    video_path = path_resolver.resolve(request.video_path)
    
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=400,
            detail=f"Video file not found: {request.video_path}"
        )
    
    # Check if already cached
    cached_data = await _load_from_cache(video_path)
    if cached_data is not None:
        return PreprocessResponse(
            character_id=request.character_id,
            video_path=request.video_path,
            cache_path=_get_cache_path(video_path),
            frame_count=len(cached_data.get('full_frames', [])),
            status="already_cached"
        )
    
    try:

        # Preprocess using wav2lip logic
        args = Config()
        temp_groups = {}
        preprocess(args, video_path, request.character_id, _detector, temp_groups)
        
        face_det_group = temp_groups[request.character_id]
        
        # Prepare data for caching
        cache_data = {
            'full_frames': face_det_group.full_frames,
            'face_det_results': face_det_group.face_det_results,
            'character_id': request.character_id,
            'video_path': request.video_path
        }
        
        # Save to cache
        cache_path = await _save_to_cache(video_path, cache_data)
        
        return PreprocessResponse(
            character_id=request.character_id,
            video_path=request.video_path,
            cache_path=cache_path,
            frame_count=len(face_det_group.full_frames),
            status="cached"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing failed: {str(e)}"
        )


@router.delete("/cache/{video_hash}")
async def clear_cache(video_hash: str):
    """Clear cached face detection for a video."""
    cache_path = _get_full_cache_path(f"{FACE_CACHE_PREFIX}{video_hash}.pkl")
    
    if os.path.exists(cache_path):
        os.remove(cache_path)
        return {"status": "deleted", "video_hash": video_hash}
    
    raise HTTPException(status_code=404, detail="Cache entry not found")


@router.get("/health", response_model=FaceDetectionHealthResponse)
async def face_detection_health():
    """Check Face Detection service health."""
    import torch
    
    # Count cached videos
    cache_dir = _get_full_cache_path(FACE_CACHE_PREFIX)
    cached_count = 0
    if os.path.exists(cache_dir):
        cached_count = len([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])
    
    return FaceDetectionHealthResponse(
        status="healthy",
        device=_device if _device else ("cuda" if torch.cuda.is_available() else "cpu"),
        model_loaded=_detector is not None,
        cached_videos=cached_count
    )

