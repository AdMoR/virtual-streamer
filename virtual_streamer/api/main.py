"""
Virtual Streamer Unified API Server

This is the main FastAPI application that combines all service layers:
- Low-level: Entity management (characters, clips)
- Medium-level: Core services (TTS, STT)
- High-level: Application workflows (video generation)
- ADK Agents: Google ADK agents mounted at /adk

The server provides a complete API for the Virtual Streamer system with
proper separation of concerns and dependency injection.
"""

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers from all layers
from virtual_streamer.api.low_level.characters import router as characters_router
from virtual_streamer.api.low_level.clips import router as clips_router
from virtual_streamer.api.low_level.story_templates import router as story_templates_router
from virtual_streamer.api.low_level.stories import router as stories_router
from virtual_streamer.api.low_level.streams import router as streams_router
from virtual_streamer.api.low_level.programmations import router as programmations_router
from virtual_streamer.api.low_level.playlist import router as playlist_router
from virtual_streamer.api.low_level.articles import router as articles_router
from virtual_streamer.api.low_level.db_browser import router as db_browser_router
from virtual_streamer.api.low_level.storage import router as storage_router
from virtual_streamer.api.low_level.candidates import router as candidates_router
from virtual_streamer.api.low_level.eval_bench import router as eval_bench_router
from virtual_streamer.api.medium_level.review import router as review_router
from virtual_streamer.api.medium_level.tts import router as tts_router
from virtual_streamer.api.medium_level.stt import router as stt_router
from virtual_streamer.api.high_level.video_generation import (
    router as video_generation_router,
)
from virtual_streamer.api.high_level.broadcast_generation import (
    router as broadcast_generation_router,
)
from virtual_streamer.api.low_level.jobs import router as jobs_router
from virtual_streamer.api.high_level.legacy_qa import router as legacy_qa_router
from virtual_streamer.api.high_level.jesus_agents import router as jesus_agents_router
from virtual_streamer.api.high_level.story_template_generation import (
    router as story_template_generation_router,
)
from virtual_streamer.api.low_level.locations import router as locations_router
from virtual_streamer.api.low_level.visual_details import router as visual_details_router
from virtual_streamer.api.high_level.location_generation import (
    router as location_generation_router,
)
from virtual_streamer.api.high_level.story_pipeline import (
    router as story_pipeline_router,
)

# Import ADK app factory and mounting utilities
from virtual_streamer.api.adk_app import create_adk_app
from virtual_streamer.api.utils.mount_app import MountableApp, mount_app

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for the main app.
    Mounted apps' lifespans are merged into this via mount_app().
    """
    # Startup
    logger.info("Virtual Streamer API Starting (Fully Unified)")
    logger.info("Data directory: %s", os.environ.get("DATA_DIR", "/data"))
    logger.info("Temp directory: %s", os.environ.get("TEMP_DIR", "./temp"))
    logger.info(
        "TTS service: %s:%s",
        os.environ.get("FISH_TTS_HOST", "localhost"),
        os.environ.get("FISH_TTS_PORT", "8003"),
    )

    # Ensure temp directory exists
    temp_dir = os.environ.get("TEMP_DIR", "./temp")
    os.makedirs(temp_dir, exist_ok=True)

    yield  # App is running

    # Shutdown
    logger.info("Virtual Streamer API shutting down...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Virtual Streamer API",
    description="""
    Fully Unified API for Virtual Streamer system with layered architecture.
    
    All entity management, services, and ML processing are integrated into this single API,
    ensuring efficient resource usage with only one instance of ML models loaded.
    
    **Low-level (Entities)**:
    - Characters: Voice samples and video clips (local storage)
    - Clips: Video clip metadata management (local storage)
    
    **Medium-level (Services)**:
    - TTS: Text-to-speech generation
    - STT: Speech-to-text transcription

    **High-level (Applications)**:
    - Video Generation: Complete story-to-video workflow
    
    **ADK Agents** (mounted at /adk):
    - story_generator: Generate stories from titles
    - video_matcher: Match videos to dialogue
    - orchestrator: Full video generation pipeline
    
    **Legacy Endpoints**:
    - /process: Backward-compatible Q&A video generation (deprecated)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
app.include_router(story_templates_router, prefix="/api/v1")
app.include_router(stories_router, prefix="/api/v1")
app.include_router(candidates_router, prefix="/api/v1")
app.include_router(eval_bench_router, prefix="/api/v1")
app.include_router(review_router, prefix="/api/v1")
app.include_router(locations_router, prefix="/api/v1")
app.include_router(visual_details_router, prefix="/api/v1")
app.include_router(articles_router, prefix="/api/v1")
# Low-level: Streaming infrastructure
app.include_router(streams_router, prefix="/api/v1")
app.include_router(programmations_router, prefix="/api/v1")
app.include_router(playlist_router, prefix="/api/v1")
# Low-level: Database browser (admin/debug)
app.include_router(db_browser_router, prefix="/api/v1")
app.include_router(storage_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")

# Medium-level: Core services
app.include_router(tts_router, prefix="/api/v1")
app.include_router(stt_router, prefix="/api/v1")

# High-level: Applications
app.include_router(video_generation_router, prefix="/api/v1")
app.include_router(broadcast_generation_router, prefix="/api/v1")
app.include_router(jesus_agents_router, prefix="/api/v1")
app.include_router(story_template_generation_router, prefix="/api/v1")
app.include_router(location_generation_router, prefix="/api/v1")
app.include_router(story_pipeline_router, prefix="/api/v1")

# Legacy: Backward compatibility
app.include_router(legacy_qa_router)  # No prefix for backward compatibility


# Mount ADK agents app
try:
    adk_app = create_adk_app()
    adk_mountable = MountableApp(
        name="adk_agents",
        app=adk_app,
        path="/adk",
        merge_lifespan=True,
        merge_docs=True,
        protected=False,
    )
    mount_app(app, adk_mountable)
    logger.info("ADK agents mounted at /adk")
except Exception as e:
    logger.warning(f"Failed to mount ADK agents: {e}")
    logger.warning("ADK agents will not be available. Continuing without them.")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Virtual Streamer API",
        "version": "2.0.0",
        "docs": "/docs",
        "layers": {
            "low_level": ["characters", "clips", "story-templates", "articles", "streams", "programmations", "playlist"],
            "medium_level": ["tts", "stt"],
            "high_level": ["video_generation"],
            "adk_agents": "/adk",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Global health check endpoint.

    Returns the health status of the entire system.
    """
    import torch

    return {
        "status": "healthy",
        "service": "virtual-streamer-api",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": os.environ.get("DATA_DIR", "/data"),
        "temp_dir": os.environ.get("TEMP_DIR", "./temp"),
    }


if __name__ == "__main__":
    import uvicorn

    # Configuration from environment
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting server on %s:%s", host, port)

    uvicorn.run(
        "virtual_streamer.api.main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info",
    )
