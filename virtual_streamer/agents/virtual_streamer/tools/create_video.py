"""
Video creation tool for the Virtual Streamer Agent.

This tool allows the agent to create new videos that will be added
to the streaming queue once generated.
"""

import logging
import os
from typing import Optional

import httpx

from virtual_streamer.agents.virtual_streamer.tools.base import (
    API_URL,
    STREAM_ID,
    register_tool,
)

logger = logging.getLogger(__name__)


@register_tool("create_video")
async def create_video(
    title: str,
    story_template_id: str = "cest_pas_sorcier",
    stream_id: Optional[str] = None,
) -> dict:
    """
    Create a new video and add it to the streaming queue.
    
    This tool submits a video generation job to the API. The video will be
    generated asynchronously and automatically added to the playlist when complete.
    
    Args:
        title: The topic/title for the video (e.g., "Pourquoi les chats retombent toujours sur leurs pattes")
        story_template_id: Which story template to use for generation (default: cest_pas_sorcier)
        stream_id: Stream ID to add the video to (uses env STREAM_ID if not provided)
        
    Returns:
        dict with:
            - job_id: ID to track the generation job
            - status: Current job status (usually "pending")
            - success: Whether the submission was successful
            - error: Error message if submission failed
    """
    # Use environment stream_id if not provided
    effective_stream_id = stream_id or STREAM_ID
    effective_api_url = os.environ.get("API_URL", API_URL)
    
    logger.info(
        f"Creating video: title='{title}', template='{story_template_id}', "
        f"stream='{effective_stream_id}'"
    )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{effective_api_url}/api/v1/video-generation/submit",
                json={
                    "title": title,
                    "story_template_id": story_template_id,
                    "stream_id": effective_stream_id,
                    "metadata": {
                        "source": "virtual_streamer_agent",
                        "requested_title": title,
                    }
                }
            )
            
            if response.status_code in (200, 202):
                data = response.json()
                logger.info(f"Video creation submitted: job_id={data.get('job_id')}")
                return {
                    "success": True,
                    "job_id": data.get("job_id"),
                    "status": data.get("status", "pending"),
                    "message": f"Vidéo '{title}' en cours de création !",
                }
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "message": "Erreur lors de la création de la vidéo",
                }
                
    except httpx.TimeoutException:
        logger.error("API request timed out")
        return {
            "success": False,
            "error": "Request timed out",
            "message": "Le serveur met trop de temps à répondre",
        }
    except httpx.RequestError as e:
        logger.error(f"API request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Impossible de contacter le serveur",
        }
