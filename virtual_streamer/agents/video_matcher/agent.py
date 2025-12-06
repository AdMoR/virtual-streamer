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
from google.adk.models import LlmResponse
from google.genai import types

from virtual_streamer.lib.agents import (
    BaseLlmAgent,
    BeforeModelCallback,
    AfterModelCallback,
    extract_llm_response_json,
)
from virtual_streamer.lib.providers.instruction import InstructionProvider
from virtual_streamer.agents.common.state_keys import task_key, result_key
from virtual_streamer.agents.video_matcher.schema import VideoJudgementOutput
from virtual_streamer.agents.video_matcher.prompt import format_judge_prompt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Instruction Provider
# ═══════════════════════════════════════════════════════════════════════════════


class VideoMatcherInstructionProvider(InstructionProvider):
    """
    Dynamic instruction provider that reads the sentence from namespaced state
    and formats the judge prompt.
    """
    
    def __init__(self, run_id: str, worker_name: str):
        """
        Initialize the instruction provider.
        
        Args:
            run_id: Unique ID for this processing run (e.g., "s0" for sentence 0)
            worker_name: Name of this worker (e.g., "m0" for matcher 0)
        """
        self.run_id = run_id
        self.worker_name = worker_name
    
    async def __call__(self, ctx: ReadonlyContext) -> str:
        """
        Generate the instruction by reading sentence from namespaced state.
        
        Args:
            ctx: Readonly context with access to state
        
        Returns:
            Formatted prompt string
        """
        sentence = ctx.state.get(task_key(self.run_id, "sentence"), "")
        if not sentence:
            logger.warning(f"No sentence found for run_id={self.run_id}")
        
        return format_judge_prompt(sentence)


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════════


class InjectVisionFrameCallback(BeforeModelCallback):
    """
    Callback that injects the base64 video frame into the LLM request.
    
    This enables the vision LLM to analyze the video frame when judging
    the video-dialogue match.
    """
    
    def __init__(self, run_id: str, worker_name: str):
        """
        Initialize the callback.
        
        Args:
            run_id: Unique ID for this processing run
            worker_name: Name of this worker
        """
        self.run_id = run_id
        self.worker_name = worker_name
    
    async def __call__(
        self,
        callback_context: CallbackContext,
    ) -> Optional[types.Content]:
        """
        Inject the vision frame into state for the LLM to use.
        
        The frame is stored in a special key that the LLM agent
        will use for vision input.
        
        Args:
            callback_context: Context with access to mutable state
        
        Returns:
            None to continue with LLM call
        """
        frame_key = task_key(self.run_id, f"{self.worker_name}:frame")
        frame = callback_context.state.get(frame_key)
        
        if not frame:
            logger.warning(f"No frame found for {self.worker_name}")
        else:
            # Store in a known location for vision processing
            callback_context.state["_vision_frame_b64"] = frame
            logger.debug(f"Injected vision frame for {self.worker_name}")
        
        return None


class StoreJudgementCallback(AfterModelCallback):
    """
    Callback that parses the LLM response and stores the judgement
    in namespaced state.
    """
    
    def __init__(self, run_id: str, worker_name: str):
        """
        Initialize the callback.
        
        Args:
            run_id: Unique ID for this processing run
            worker_name: Name of this worker
        """
        self.run_id = run_id
        self.worker_name = worker_name
    
    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        """
        Parse judgement and store in namespaced state.
        
        Args:
            callback_context: Context with access to mutable state
            llm_response: Response from the vision LLM
        """
        # Get the video path for this worker
        video_key = task_key(self.run_id, f"{self.worker_name}:video")
        video_path = callback_context.state.get(video_key, "")
        
        # Parse the structured output
        parsed = extract_llm_response_json(llm_response)
        
        if parsed:
            judgement = {
                "video_path": video_path,
                "rating": parsed.get("rating", "NOT_CONTEXTUAL"),
                "grade": parsed.get("grade", 0),
                "reasoning": parsed.get("reasoning", ""),
            }
        else:
            # Fallback if parsing fails
            logger.warning(f"Failed to parse judgement for {self.worker_name}")
            judgement = {
                "video_path": video_path,
                "rating": "NOT_CONTEXTUAL",
                "grade": 0,
                "reasoning": "Failed to parse LLM response",
            }
        
        # Store in namespaced result key
        callback_context.state[result_key(self.run_id, self.worker_name)] = judgement
        
        logger.debug(
            f"Stored judgement for {self.worker_name}: "
            f"rating={judgement['rating']}, grade={judgement['grade']}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════════════


def get_video_matcher(run_id: str, worker_name: str) -> BaseLlmAgent:
    """
    Factory function to create a VideoMatcherAgent for a specific run.
    
    This creates a new agent instance for each video being matched,
    with callbacks configured for the specific run_id and worker_name.
    
    Args:
        run_id: Unique ID for this processing run (e.g., "s0" for sentence 0)
        worker_name: Name for this worker (e.g., "m0", "m1", etc.)
    
    Returns:
        Configured BaseLlmAgent for video matching
    
    Example:
        # Create matcher for sentence 0, video 2
        matcher = get_video_matcher("s0", "m2")
    """
    return BaseLlmAgent(
        name=worker_name,
        instruction=VideoMatcherInstructionProvider(run_id, worker_name),
        output_schema=VideoJudgementOutput,
        before_model_callback=[InjectVisionFrameCallback(run_id, worker_name)],
        after_model_callback=[StoreJudgementCallback(run_id, worker_name)],
    )

