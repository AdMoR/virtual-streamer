"""
Video Server - Pure API Proxy

This is a minimal FastAPI application that proxies requests to the main
Virtual Streamer API. It contains zero business logic - all decisions
about what video to play next are made by the main API.

The server:
1. Exposes /api/next to get the next video URL
2. Exposes /api/played/{entry_id} to mark a video as played
3. Serves the static HTML player at /

Environment variables:
- API_URL: URL of the main Virtual Streamer API (default: http://virtual_streamer_api:8000)
- STREAM_ID: ID of the stream to play from (default: default)
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import httpx

from virtual_streamer.streaming.video_server.atari.routes import router as atari_router

# Configuration from environment
API_URL = os.environ.get("API_URL", "http://virtual_streamer_api:8000")
STREAM_ID = os.environ.get("STREAM_ID", "default")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Video Server Proxy",
    description="Pure API proxy for OBS video streaming",
    version="1.0.0",
)

app.include_router(atari_router)


@app.get("/api/next")
async def get_next():
    """
    Proxy: Get next video to play.
    
    Forwards request to the main API's /streams/{stream_id}/next-video endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{API_URL}/api/v1/streams/{STREAM_ID}/next-video"
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"API unavailable: {str(e)}")


@app.post("/api/played/{entry_id}")
async def mark_played(entry_id: str):
    """
    Proxy: Mark video as played.
    
    Forwards request to the main API's /playlist/{entry_id}/played endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{API_URL}/api/v1/playlist/{entry_id}/played"
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"API unavailable: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "stream_id": STREAM_ID,
        "api_url": API_URL,
    }


@app.get("/config")
async def config():
    """Get current configuration (useful for debugging)."""
    return {
        "stream_id": STREAM_ID,
        "api_url": API_URL,
    }


# Serve index.html for root path
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the video player HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    else:
        return HTMLResponse(
            content="""
            <html>
            <body>
                <h1>Video Server</h1>
                <p>Static files not found. Please ensure index.html is in the static directory.</p>
                <p>API URL: {api_url}</p>
                <p>Stream ID: {stream_id}</p>
            </body>
            </html>
            """.format(api_url=API_URL, stream_id=STREAM_ID)
        )


# Mount static files (if directory exists)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
