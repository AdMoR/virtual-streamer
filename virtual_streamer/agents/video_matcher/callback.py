"""
Callbacks for VideoMatcher agent.

These callbacks handle:
- InjectVisionFrameCallback: Reads video/sentence from state, extracts frame, injects into LLM request
- StoreJudgementCallback: Parses LLM response and stores judgement in state
"""

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types

from virtual_streamer.lib.agents import (
    StateInputCallback,
    StateOutputCallback,
    extract_llm_response_json,
)
from virtual_streamer.agents.video_matcher.schema import (
    VideoJudgementOutput,
    VideoMatchResult,
    VideoSentenceInput,
)
from virtual_streamer.agents.common.utils import extract_middle_frame

logger = logging.getLogger(__name__)


class InjectVisionFrameCallback(StateInputCallback):
    """
    Callback that injects the base64 video frame into the LLM request.

    This enables the vision LLM to analyze the video frame when judging
    the video-dialogue match. It reads the video path and sentence from
    the namespaced state key, extracts the middle frame, and injects it
    into the LLM request.
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize the callback.

        Args:
            run_id: Unique ID for this processing run (e.g., "s0_w1")
        """
        super().__init__(
            input_key="video_sentence",
            input_schema=VideoSentenceInput,
            run_id=run_id,
        )

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[types.Content]:
        """
        Inject the vision frame into the LLM request.

        Reads video_path and sentence from state, extracts the middle frame
        from the video, and appends it to the LLM request for vision analysis.

        Args:
            callback_context: Context with access to mutable state
            llm_request: The LLM request to modify

        Returns:
            None to continue with LLM call

        Raises:
            Exception: If sentence/video_path are missing or frame extraction fails
        """
        input_key = self.get_input_key()
        
        # Try to read from state first
        state_data = callback_context.state.get(input_key)
        
        if state_data:
            # Parse from state
            video_sentence = self.input_schema.model_validate_json(state_data)
            sentence = video_sentence.sentence
            video_path = video_sentence.video_path
        else:
            # Fallback: try to parse from request content
            try:
                to_parse = llm_request.contents[0].parts[0].text
                video_sentence = self.input_schema.model_validate_json(to_parse)
                logger.info(f"Parsed video sentence from request: {video_sentence}")
                sentence = video_sentence.sentence
                video_path = video_sentence.video_path
            except Exception as e:
                raise Exception(
                    f"Could not find input at key '{input_key}' in state "
                    f"or parse from request: {e}"
                )
        
        if not (sentence and video_path):
            raise Exception(
                f"Sentence and video path are empty: {sentence}, {video_path} "
                f"for key '{input_key}'"
            )

        # Extract middle frame from video
        frame = extract_middle_frame(video_path)
        if frame is None:
            raise Exception(f"Failed to extract frame from {video_path}")
        
        # Append sentence and frame to user content
        callback_context.user_content.parts.append(
            types.Part.from_text(text=f"Sentence : {sentence}")
        )
        callback_context.user_content.parts.append(
            types.Part.from_bytes(data=frame, mime_type="image/jpeg")
        )
        
        # Also append to request contents
        llm_request.contents[0].parts.append(
            types.Part.from_bytes(data=frame, mime_type="image/jpeg")
        )
        
        logger.info(f"Injected vision frame for: {sentence[:50]}...")
        
        # Return None to continue with LLM call
        return None


class StoreJudgementCallback(StateOutputCallback):
    """
    Callback that parses the LLM response and stores the full match result
    (including video_path) in namespaced state.
    
    The output schema is VideoMatchResult which combines:
    - Input data (sentence, video_path)
    - LLM output (rating, grade, reasoning)
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize the callback.

        Args:
            run_id: Unique ID for this processing run (e.g., "s0_w1")
        """
        super().__init__(
            output_key="judgement",
            output_schema=VideoMatchResult,
            run_id=run_id,
        )

    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> Optional[LlmResponse]:
        """
        Parse judgement, combine with input, and store full result in state.

        Args:
            callback_context: Context with access to mutable state
            llm_response: Response from the vision LLM

        Returns:
            Modified LlmResponse with video frame appended for display,
            or None if video data not found.
        """
        # Parse the LLM output into VideoJudgementOutput
        llm_output: VideoJudgementOutput = extract_llm_response_json(
            llm_response, VideoJudgementOutput
        )
        
        if llm_output is None:
            logger.warning("Failed to parse judgement from LLM response")
            return None
        
        # Read the input data to get sentence and video_path
        input_key = f"task:{self.run_id}:video_sentence" if self.run_id else "video_sentence"
        state_data = callback_context.state.get(input_key)
        
        if not state_data:
            logger.warning(f"Could not find input data at key '{input_key}'")
            # Store just the LLM output without video_path
            output_key = self.get_output_key()
            callback_context.state[output_key] = llm_output.model_dump_json()
            return None
        
        # Parse input data
        input_data = VideoSentenceInput.model_validate_json(state_data)
        
        # Combine input and output into VideoMatchResult
        result = VideoMatchResult.from_input_and_output(input_data, llm_output)
        
        # Store the full result in state
        output_key = self.get_output_key()
        callback_context.state[output_key] = result.model_dump_json()
        
        logger.info(
            f"Stored match result at '{output_key}': "
            f"video={result.video_path}, rating={result.rating}, grade={result.grade}"
        )
        
        # Optional: Append the video frame for better display
        try:
            frame = extract_middle_frame(input_data.video_path)
            if frame:
                llm_response.content.parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            data=frame,
                            display_name="video_frame",
                            mime_type="image/jpeg"
                        )
                    )
                )
                return llm_response
        except Exception as e:
            logger.debug(f"Could not append frame to response: {e}")
        
        return None
