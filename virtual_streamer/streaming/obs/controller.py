"""
OBS WebSocket controller.

Async facade over the synchronous obsws-python client (OBS WebSocket v5,
port 4455 on the streaming_obs container). All blocking calls run in a
thread via asyncio.to_thread. The client connects lazily and reconnects
once on failure before surfacing an OBSError.

Environment variables:
- OBS_WS_HOST: OBS websocket host (default: streaming_obs)
- OBS_WS_PORT: OBS websocket port (default: 4455)
- OBS_WS_PASSWORD: OBS websocket password (default: empty)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OBSConfig:
    host: str = field(default_factory=lambda: os.environ.get("OBS_WS_HOST", "streaming_obs"))
    port: int = field(default_factory=lambda: int(os.environ.get("OBS_WS_PORT", "4455")))
    password: str = field(default_factory=lambda: os.environ.get("OBS_WS_PASSWORD", ""))
    timeout: float = 5.0


class OBSError(Exception):
    """Raised when an OBS websocket request fails."""


# Input kinds understood by OBS for the sources we create
BROWSER_INPUT_KIND = "browser_source"
MEDIA_INPUT_KIND = "ffmpeg_source"


class OBSController:
    """Thin async wrapper around obsws_python.ReqClient."""

    def __init__(self, config: Optional[OBSConfig] = None):
        self.config = config or OBSConfig()
        self._client = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        import obsws_python as obs

        return obs.ReqClient(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            timeout=self.config.timeout,
        )

    async def _call(self, method: str, *args, **kwargs) -> Any:
        """Run a ReqClient method in a thread, reconnecting once on failure."""
        async with self._lock:
            for attempt in (1, 2):
                try:
                    if self._client is None:
                        self._client = await asyncio.to_thread(self._connect)
                    fn = getattr(self._client, method)
                    return await asyncio.to_thread(fn, *args, **kwargs)
                except Exception as e:
                    self._client = None
                    if attempt == 2:
                        raise OBSError(f"OBS request '{method}' failed: {e}") from e
                    logger.warning("OBS request '%s' failed (%s), reconnecting", method, e)

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                try:
                    await asyncio.to_thread(self._client.disconnect)
                except Exception:
                    pass
                self._client = None

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------

    async def list_scenes(self) -> List[Dict[str, Any]]:
        resp = await self._call("get_scene_list")
        return list(reversed(resp.scenes))  # OBS returns them bottom-up

    async def create_scene(self, name: str) -> None:
        await self._call("create_scene", name)

    async def remove_scene(self, name: str) -> None:
        await self._call("remove_scene", name)

    async def get_current_scene(self) -> str:
        resp = await self._call("get_current_program_scene")
        return resp.current_program_scene_name

    async def set_current_scene(self, name: str) -> None:
        await self._call("set_current_program_scene", name)

    # ------------------------------------------------------------------
    # Scene items / inputs
    # ------------------------------------------------------------------

    async def list_scene_items(self, scene: str) -> List[Dict[str, Any]]:
        resp = await self._call("get_scene_item_list", scene)
        return resp.scene_items

    async def add_browser_source(
        self,
        scene: str,
        name: str,
        url: str,
        width: int = 1920,
        height: int = 1080,
        control_audio: bool = True,
    ) -> int:
        settings = {
            "url": url,
            "width": width,
            "height": height,
            # Route page audio through the OBS mixer instead of desktop audio
            "reroute_audio": control_audio,
            "restart_when_active": True,
            "shutdown": False,
        }
        resp = await self._call(
            "create_input", scene, name, BROWSER_INPUT_KIND, settings, True
        )
        return resp.scene_item_id

    async def add_media_source(
        self, scene: str, name: str, file_or_url: str, loop: bool = False
    ) -> int:
        if file_or_url.startswith(("http://", "https://", "rtmp://", "rtsp://")):
            settings = {"is_local_file": False, "input": file_or_url, "looping": loop}
        else:
            settings = {"is_local_file": True, "local_file": file_or_url, "looping": loop}
        resp = await self._call(
            "create_input", scene, name, MEDIA_INPUT_KIND, settings, True
        )
        return resp.scene_item_id

    async def update_input_settings(self, name: str, settings: Dict[str, Any]) -> None:
        await self._call("set_input_settings", name, settings, True)

    async def set_item_transform(
        self, scene: str, item_id: int, transform: Dict[str, Any]
    ) -> None:
        await self._call("set_scene_item_transform", scene, item_id, transform)

    async def set_item_enabled(self, scene: str, item_id: int, enabled: bool) -> None:
        await self._call("set_scene_item_enabled", scene, item_id, enabled)

    async def remove_scene_item(self, scene: str, item_id: int) -> None:
        await self._call("remove_scene_item", scene, item_id)

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    async def get_status(self) -> Dict[str, Any]:
        stream = await self._call("get_stream_status")
        current_scene = await self.get_current_scene()
        return {
            "connected": True,
            "current_scene": current_scene,
            "streaming": stream.output_active,
            "stream_timecode": getattr(stream, "output_timecode", None),
            "reconnecting": getattr(stream, "output_reconnecting", False),
        }

    async def start_stream(self) -> None:
        await self._call("start_stream")

    async def stop_stream(self) -> None:
        await self._call("stop_stream")


# Global controller instance (lazy initialized)
_obs_controller: Optional[OBSController] = None


def get_obs_controller() -> OBSController:
    """Get or create the global OBS controller instance."""
    global _obs_controller
    if _obs_controller is None:
        _obs_controller = OBSController()
    return _obs_controller


def reset_obs_controller() -> None:
    """Reset the global OBS controller instance (useful for testing)."""
    global _obs_controller
    _obs_controller = None
