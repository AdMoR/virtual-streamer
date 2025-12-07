import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types

from virtual_streamer.agents.common.state_keys import task_key, result_key
from virtual_streamer.agents.video_matcher.schema import VideoJudgementOutput, VideoSentenceInput
from virtual_streamer.lib.agents.callbacks import (
    BeforeModelCallback,
    AfterModelCallback,
    extract_llm_response_json,
    extract_llm_content_json
)
from virtual_streamer.agents.common.utils import (
    extract_middle_frame,
)

VIDEO_PATH_KEY = "video_path"
SENTENCE_KEY = "sentence"


class InjectVisionFrameCallback(BeforeModelCallback):
    """
    Callback that injects the base64 video frame into the LLM request.

    This enables the vision LLM to analyze the video frame when judging
    the video-dialogue match.
    """


    def __init__(self, run_id: str):
        """
        Initialize the callback.

        Args:
            run_id: Unique ID for this processing run
            worker_name: Name of this worker
        """
        self.run_id = run_id

    async def __call__(
            self,
            callback_context: CallbackContext,
            llm_request: LlmRequest,
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
        frame_key = task_key(self.run_id, VIDEO_PATH_KEY)
        video_path = callback_context.state.get(frame_key)
        sentence_key = task_key(self.run_id, SENTENCE_KEY)
        sentence = callback_context.state.get(sentence_key)

        if not (sentence and video_path):
            # Try to parse the input
            to_parse = llm_request.contents[0].parts[0].text
            video_sentence = VideoSentenceInput.model_validate_json(to_parse)
            logging.info(f"Video sentence: {video_sentence}")
            sentence = video_sentence.sentence
            video_path = video_sentence.video_path
        if not (sentence and video_path):
            raise Exception(f"Sentence and video path are empty : {sentence}, {video_path} for run {self.run_id}")

        frame = extract_middle_frame(video_path)
        if frame is None:
            raise Exception("Failed to extract frame")
        callback_context.user_content.parts.append(
            types.Part.from_text(text=f"Sentence : {sentence}")
        )
        callback_context.user_content.parts.append(
            types.Part.from_bytes(data=frame, mime_type="image/jpeg")
        )
        llm_request.contents[0].parts.append(
            types.Part.from_bytes(data=frame, mime_type="image/jpeg")
        )
        logging.info(f"Injected vision frame : {callback_context.user_content.parts}")
        # Must return None to avoid shortcut
        return None


class StoreJudgementCallback(AfterModelCallback):
    """
    Callback that parses the LLM response and stores the judgement
    in namespaced state.
    """
    RESULT_KEY = "result"

    def __init__(self, run_id: str):
        """
        Initialize the callback.

        Args:
            run_id: Unique ID for this processing run
            worker_name: Name of this worker
        """
        self.run_id = run_id

    async def __call__(
            self,
            callback_context: CallbackContext,
            llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """
        Parse judgement and store in namespaced state.

        Args:
            callback_context: Context with access to mutable state
            llm_response: Response from the vision LLM
        """
        # Parse the structured output into VideoJudgementOutput model
        parsed: VideoJudgementOutput = extract_llm_response_json(llm_response, VideoJudgementOutput)
        # Store in namespaced result key
        callback_context.state[result_key(self.run_id, self.RESULT_KEY)] = parsed.rating

        # Optional part : display the judgement better with the image
        frame_key = task_key(self.run_id, VIDEO_PATH_KEY)
        video_path = callback_context.state.get(frame_key)
        sentence_key = task_key(self.run_id, SENTENCE_KEY)
        sentence = callback_context.state.get(sentence_key)
        if not (sentence and video_path):
            logging.warning("Did not find the video data or sentence")
            return None
        frame = extract_middle_frame(video_path)
        llm_response.content.parts.append(
            types.Part(inline_data=types.Blob(data=frame, display_name="video_frame", mime_type="image/jpeg")))
        return llm_response
