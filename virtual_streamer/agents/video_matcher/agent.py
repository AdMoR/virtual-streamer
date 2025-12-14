"""
Video Matcher Agent.

Judges if a video clip matches a dialogue using vision LLM.
This is a StatefulLlmAgent created via factory function with:
- InjectVisionFrameCallback that reads sentence/video from namespaced state
- StoreJudgementCallback that stores the judgement in namespaced state

The agent exposes get_input_key() and get_output_key() methods for
the DynamicParallelAgent to discover where to write inputs and read outputs.

Example:
    # Create a matcher for a specific worker
    matcher = get_video_matcher(run_id="s0_w1")
    
    # Get keys for state management
    input_key = matcher.get_input_key()   # "task:s0_w1:video_sentence"
    output_key = matcher.get_output_key() # "result:s0_w1:judgement"
    
    # Input format (write to input_key):
    {"sentence": "La Jamy, je suis dans un datacenter", 
     "video_path": "/path/to/video.mp4"}
    
    # Output format (read from output_key):
    {"rating": "CONTEXTUAL", "grade": 8, "reasoning": "..."}
"""

import logging
from typing import Optional

from virtual_streamer.lib.agents import StatefulLlmAgent
from virtual_streamer.agents.video_matcher.prompt import JUDGE_PROMPT
from virtual_streamer.agents.video_matcher.callback import (
    InjectVisionFrameCallback,
    StoreJudgementCallback,
)

logger = logging.getLogger(__name__)


def get_video_matcher(run_id: Optional[str] = None) -> StatefulLlmAgent:
    """
    Factory function to create a VideoMatcherAgent for a specific run.
    
    This creates a new agent instance for each video being matched,
    with callbacks configured for the specific run_id.
    
    Args:
        run_id: Unique ID for this processing run (e.g., "s0_w1").
                If None, keys will not be namespaced.
    
    Returns:
        Configured StatefulLlmAgent for video matching with:
        - get_input_key(): returns the state key for input
        - get_output_key(): returns the state key for output
    
    Example:
        # Without run_id (for standalone use)
        matcher = get_video_matcher()
        print(matcher.get_input_key())  # "video_sentence"
        
        # With run_id (for parallel processing)
        matcher = get_video_matcher(run_id="s0_w1")
        print(matcher.get_input_key())   # "task:s0_w1:video_sentence"
        print(matcher.get_output_key())  # "result:s0_w1:judgement"
    """
    input_callback = InjectVisionFrameCallback(run_id)
    output_callback = StoreJudgementCallback(run_id)
    
    return StatefulLlmAgent(
        name="video_matcher",
        instruction=JUDGE_PROMPT,
        input_callback=input_callback,
        output_callback=output_callback,
    )


# Default agent instance for ADK CLI usage
root_agent = get_video_matcher()
