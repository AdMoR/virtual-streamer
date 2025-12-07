"""
Video Matcher Agent.

Judges if a video clip matches a dialogue using vision LLM.
This is a standard BaseLlmAgent created via factory function with:
- InstructionProvider that reads sentence from namespaced state
- BeforeModelCallback that injects the vision frame
- AfterModelCallback that stores the judgement in namespaced state
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import LlmAgent
from google.adk.models import LlmResponse
from google.genai import types

from virtual_streamer.lib.agents import (
    BaseLlmAgent,
)
from virtual_streamer.agents.video_matcher.schema import VideoJudgementOutput, VideoSentenceInput
from virtual_streamer.agents.video_matcher.prompt import JUDGE_PROMPT
from virtual_streamer.agents.video_matcher.callback import InjectVisionFrameCallback, StoreJudgementCallback

logger = logging.getLogger(__name__)



def get_video_matcher(run_id: str) -> BaseLlmAgent:
    """
    Factory function to create a VideoMatcherAgent for a specific run.
    
    This creates a new agent instance for each video being matched,
    with callbacks configured for the specific run_id and worker_name.
    
    Args:
        run_id: Unique ID for this processing run (e.g., "s0" for sentence 0)
        worker_name: Name for this worker (e.g., "m0", "m1", etc.)
    
    Returns:
        Configured BaseLlmAgent for video matching
    
    Example of input :

    WORKING
    {"sentence": "La Jamy, je suis dans un datacenter", "video_path": "/home/amor/Downloads/FRED ET JAMY FONT TOUT POUR ÊTRE DANS LES TENDANCES YOUTUBE !! [DCf-EI5WgEw]-Scene-017.mp4"}

    FAILING
    {"sentence": "La Jamy, je suis dans un datacenter", "video_path": "/home/amor/Documents/video.mp4"}
    {"sentence": "Coucou Jamy, la je me balade en forêt de Fontainebleau", "video_path": "/home/amor/Documents/result_2025-08-10-19:34:50.249412Test.mp4"}

    """
    return BaseLlmAgent(
        name="video_matcher",
        instruction=JUDGE_PROMPT,
        input_schema=VideoSentenceInput,
        output_schema=VideoJudgementOutput,
        before_model_callback=[InjectVisionFrameCallback(run_id)],
        after_model_callback=[StoreJudgementCallback(run_id)],
    )

root_agent = get_video_matcher("xxxxxx")