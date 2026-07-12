"""
LiveChannel registry — the standard contract for real-time video sources.

A producer (atari game loop, any future live generator):
  1. register_channel(channel_id, kind, metadata)
  2. publish_frame(channel_id, jpeg_bytes) at its native rate
  3. optionally publish_event(channel_id, {...}) for audio/overlay events
  4. close_channel(channel_id) when done

Consumers subscribe via the /api/live/{id}/ws websocket (see routes.py),
which fans frames out through per-subscriber bounded queues. Video frames
drop-oldest under backpressure; JSON events are never dropped.

The registry is in-memory and process-local, like the atari session store:
frames are raw bytes and cannot cheaply cross processes. The main API
proxies the control plane here.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Message kinds carried on subscriber queues
KIND_FRAME = "frame"
KIND_EVENT = "event"
KIND_CLOSE = "close"


@dataclass
class LiveChannel:
    channel_id: str
    kind: str
    metadata: Dict[str, Any]
    created_at: datetime
    status: str = "starting"  # starting | live | stopped
    current_frame: Optional[bytes] = None
    subscribers: List[asyncio.Queue] = field(default_factory=list)


_channels: Dict[str, LiveChannel] = {}


def register_channel(
    channel_id: str, kind: str, metadata: Optional[Dict[str, Any]] = None
) -> LiveChannel:
    channel = LiveChannel(
        channel_id=channel_id,
        kind=kind,
        metadata=metadata or {},
        created_at=datetime.utcnow(),
    )
    _channels[channel_id] = channel
    logger.info("Live channel registered: %s (kind=%s)", channel_id, kind)
    return channel


def get_channel(channel_id: str) -> Optional[LiveChannel]:
    return _channels.get(channel_id)


def list_channels() -> List[LiveChannel]:
    return list(_channels.values())


def subscribe(channel_id: str, maxsize: int = 4) -> asyncio.Queue:
    """Create a subscriber queue for a channel. Caller must unsubscribe."""
    channel = _channels[channel_id]
    queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    channel.subscribers.append(queue)
    return queue


def unsubscribe(channel_id: str, queue: asyncio.Queue) -> None:
    channel = _channels.get(channel_id)
    if channel is not None and queue in channel.subscribers:
        channel.subscribers.remove(queue)


def _offer(queue: asyncio.Queue, item: Tuple[str, Any], droppable: bool) -> None:
    """Put an item on a subscriber queue.

    Droppable items (video frames) evict the oldest droppable entry under
    backpressure; non-droppable items (events, close) always get through.
    """
    while True:
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            try:
                dropped = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if dropped[0] != KIND_FRAME:
                # Never lose an event: put it back and drop the new frame instead
                queue.put_nowait(dropped)
                if droppable:
                    return


def publish_frame(channel_id: str, jpeg: bytes) -> None:
    """Set the channel's current frame and fan it out to all subscribers."""
    channel = _channels.get(channel_id)
    if channel is None:
        return
    channel.current_frame = jpeg
    if channel.status == "starting":
        channel.status = "live"
    for queue in channel.subscribers:
        _offer(queue, (KIND_FRAME, jpeg), droppable=True)


def publish_event(channel_id: str, event: Dict[str, Any]) -> None:
    """Fan a JSON event (audio, music, overlay...) out to all subscribers."""
    channel = _channels.get(channel_id)
    if channel is None:
        return
    for queue in channel.subscribers:
        _offer(queue, (KIND_EVENT, event), droppable=False)


def close_channel(channel_id: str) -> None:
    channel = _channels.pop(channel_id, None)
    if channel is None:
        return
    channel.status = "stopped"
    for queue in channel.subscribers:
        _offer(queue, (KIND_CLOSE, None), droppable=False)
    channel.subscribers.clear()
    logger.info("Live channel closed: %s", channel_id)
