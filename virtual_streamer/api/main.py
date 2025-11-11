"""
Virtual Streamer Unified API Server

This is the main FastAPI application that combines all service layers:
- Low-level: Entity management (characters, clips)
- Medium-level: Core services (TTS, STT, Wav2lip)
- High-level: Application workflows (video generation)

The server provides a complete API for the Virtual Streamer system with
proper separation of concerns and dependency injection.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Import routers from all layers
from virtual_streamer.api.low_level.characters import router as characters_router
from virtual_streamer.api.low_level.clips import router as clips_router
from virtual_streamer.api.medium_level.tts import router as tts_router
from virtual_streamer.api.medium_level.stt import router as stt_router
from virtual_streamer.api.medium_level.wav2lip import router as wav2lip_router
from virtual_streamer.api.high_level.video_generation import router as video_generation_router

# Create FastAPI app
app = FastAPI(
    title="Virtual Streamer API",
    description="""
    Unified API for Virtual Streamer system with layered architecture:
    
    **Low-level (Entities)**:
    - Characters: Voice samples and video clips
    - Clips: Video clip metadata management
    
    **Medium-level (Services)**:
    - TTS: Text-to-speech generation
    - STT: Speech-to-text transcription
    - Wav2Lip: Lip-sync video generation
    
    **High-level (Applications)**:
    - Video Generation: Complete story-to-video workflow
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
# Low-level: Entity management
app.include_router(characters_router, prefix="/api/v1")
app.include_router(clips_router, prefix="/api/v1")

# Medium-level: Core services
app.include_router(tts_router, prefix="/api/v1")
app.include_router(stt_router, prefix="/api/v1")
app.include_router(wav2lip_router, prefix="/api/v1")

# High-level: Applications
app.include_router(video_generation_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Virtual Streamer API",
        "version": "1.0.0",
        "docs": "/docs",
        "layers": {
            "low_level": ["characters", "clips"],
            "medium_level": ["tts", "stt", "wav2lip"],
            "high_level": ["video_generation"]
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Global health check endpoint.
    
    Returns the health status of the entire system.
    """
    return {
        "status": "healthy",
        "service": "virtual-streamer-api",
        "data_dir": os.environ.get("DATA_DIR", "/data"),
        "temp_dir": os.environ.get("TEMP_DIR", "./temp")
    }


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("=" * 70)
    print("Virtual Streamer API Starting")
    print("=" * 70)
    print(f"Data directory: {os.environ.get('DATA_DIR', '/data')}")
    print(f"Temp directory: {os.environ.get('TEMP_DIR', './temp')}")
    print(f"Entity service: {os.environ.get('ENTITY_SERVICE_HOST', '0.0.0.0')}:8002")
    print("=" * 70)
    
    # Ensure temp directory exists
    temp_dir = os.environ.get("TEMP_DIR", "./temp")
    os.makedirs(temp_dir, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Virtual Streamer API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    # Configuration from environment
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    print(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "virtual_streamer.api.main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

