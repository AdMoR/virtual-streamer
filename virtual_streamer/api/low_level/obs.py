"""
Low-level API: OBS control.

Remote control of the OBS instance over its websocket (port 4455).
This is the API surface agents use to edit scenes, switch the streamed
scene, and manage sources — including the two standardized broadcast
modes:
- "playlist" browser source → video_server playlist player at /
- "live_channel" browser source → video_server /live?channel={id}
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from virtual_streamer.streaming.obs import OBSError, get_obs_controller

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/obs", tags=["OBS Control"])

# Public URL of the video_server as seen from the OBS container
VIDEO_SERVER_PUBLIC_URL = os.environ.get(
    "VIDEO_SERVER_PUBLIC_URL", "http://video_server:5000"
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SceneCreateRequest(BaseModel):
    name: str


class SourceCreateRequest(BaseModel):
    type: str = Field(description="browser | media | live_channel | playlist")
    name: str
    url: Optional[str] = None
    file: Optional[str] = None
    channel_id: Optional[str] = None
    width: int = 1920
    height: int = 1080
    loop: bool = False
    transform: Optional[Dict[str, Any]] = None


class ItemUpdateRequest(BaseModel):
    settings: Optional[Dict[str, Any]] = None
    transform: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    input_name: Optional[str] = Field(
        default=None, description="Required when updating settings"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_http_error(e: OBSError) -> HTTPException:
    msg = str(e)
    if "already exists" in msg.lower():
        return HTTPException(status_code=409, detail=msg)
    if "not found" in msg.lower() or "no source" in msg.lower():
        return HTTPException(status_code=404, detail=msg)
    return HTTPException(status_code=502, detail=msg)


def _resolve_source_url(req: SourceCreateRequest) -> str:
    if req.type == "live_channel":
        if not req.channel_id:
            raise HTTPException(status_code=422, detail="channel_id required for live_channel")
        return f"{VIDEO_SERVER_PUBLIC_URL}/live?channel={req.channel_id}"
    if req.type == "playlist":
        return f"{VIDEO_SERVER_PUBLIC_URL}/"
    if not req.url:
        raise HTTPException(status_code=422, detail="url required for browser source")
    return req.url


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
async def obs_status():
    """OBS connection, current scene, and streaming status."""
    try:
        return await get_obs_controller().get_status()
    except OBSError as e:
        return {"connected": False, "error": str(e)}


@router.get("/scenes")
async def list_scenes():
    try:
        obs = get_obs_controller()
        scenes = await obs.list_scenes()
        current = await obs.get_current_scene()
        return {"scenes": scenes, "current_scene": current}
    except OBSError as e:
        raise _obs_http_error(e)


@router.post("/scenes", status_code=201)
async def create_scene(req: SceneCreateRequest):
    try:
        await get_obs_controller().create_scene(req.name)
        return {"name": req.name, "created": True}
    except OBSError as e:
        raise _obs_http_error(e)


@router.delete("/scenes/{name}")
async def remove_scene(name: str):
    try:
        await get_obs_controller().remove_scene(name)
        return {"name": name, "deleted": True}
    except OBSError as e:
        raise _obs_http_error(e)


@router.post("/scenes/{name}/activate")
async def activate_scene(name: str):
    """Switch the streamed (program) scene."""
    try:
        await get_obs_controller().set_current_scene(name)
        return {"current_scene": name}
    except OBSError as e:
        raise _obs_http_error(e)


@router.get("/scenes/{name}/items")
async def list_scene_items(name: str):
    try:
        return {"scene": name, "items": await get_obs_controller().list_scene_items(name)}
    except OBSError as e:
        raise _obs_http_error(e)


@router.post("/scenes/{name}/sources", status_code=201)
async def add_source(name: str, req: SourceCreateRequest):
    """
    Add a source to a scene.

    type=browser: generic browser source (url required)
    type=live_channel: browser source watching a LiveChannel (channel_id required)
    type=playlist: browser source watching the playlist player
    type=media: media source (file or url required)
    """
    obs = get_obs_controller()
    try:
        if req.type == "media":
            target = req.file or req.url
            if not target:
                raise HTTPException(status_code=422, detail="file or url required for media")
            item_id = await obs.add_media_source(name, req.name, target, loop=req.loop)
        elif req.type in ("browser", "live_channel", "playlist"):
            url = _resolve_source_url(req)
            item_id = await obs.add_browser_source(
                name, req.name, url, width=req.width, height=req.height
            )
        else:
            raise HTTPException(status_code=422, detail=f"Unknown source type: {req.type}")

        if req.transform:
            await obs.set_item_transform(name, item_id, req.transform)
        return {"scene": name, "source": req.name, "item_id": item_id}
    except OBSError as e:
        raise _obs_http_error(e)


@router.patch("/scenes/{name}/items/{item_id}")
async def update_scene_item(name: str, item_id: int, req: ItemUpdateRequest):
    obs = get_obs_controller()
    try:
        if req.settings is not None:
            if not req.input_name:
                raise HTTPException(
                    status_code=422, detail="input_name required to update settings"
                )
            await obs.update_input_settings(req.input_name, req.settings)
        if req.transform is not None:
            await obs.set_item_transform(name, item_id, req.transform)
        if req.enabled is not None:
            await obs.set_item_enabled(name, item_id, req.enabled)
        return {"scene": name, "item_id": item_id, "updated": True}
    except OBSError as e:
        raise _obs_http_error(e)


@router.delete("/scenes/{name}/items/{item_id}")
async def remove_scene_item(name: str, item_id: int):
    try:
        await get_obs_controller().remove_scene_item(name, item_id)
        return {"scene": name, "item_id": item_id, "deleted": True}
    except OBSError as e:
        raise _obs_http_error(e)


@router.post("/stream/start")
async def start_stream():
    try:
        await get_obs_controller().start_stream()
        return {"streaming": True}
    except OBSError as e:
        raise _obs_http_error(e)


@router.post("/stream/stop")
async def stop_stream():
    try:
        await get_obs_controller().stop_stream()
        return {"streaming": False}
    except OBSError as e:
        raise _obs_http_error(e)
