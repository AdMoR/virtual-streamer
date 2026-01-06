"""
Core logic for video generation utilities.

This module contains utility functions for:
- Concurrency control (semaphore-based)
- Text processing (sentence splitting)
- Story generation from title
- Video frame extraction
- Video-dialogue matching judgement
- Search keyword generation

Note: The main video generation flow is now handled by ADK agents.
See virtual_streamer.api.high_level.video_generation for the supported API.
"""

import asyncio
import cv2
import base64
from typing import List, Optional

from virtual_streamer.video_generation.interfaces import (
    LLMInterface,
    PromptProviderInterface,
    VideoJudgementResult,
    ProgressCallback,
)
from virtual_streamer.video_generation.config import (
    VideoGenerationConfig,
    StoryOutput,
)


# ============================================================================
# Concurrency Control
# ============================================================================


async def with_semaphore(semaphore: asyncio.Semaphore, coro):
    """
    Execute a coroutine with semaphore-based concurrency control.

    This ensures that only a limited number of LLM API calls run concurrently,
    preventing rate limit issues and API overload.

    Args:
        semaphore: Asyncio semaphore for concurrency control
        coro: Coroutine to execute

    Returns:
        Result of the coroutine
    """
    async with semaphore:
        return await coro


# ============================================================================
# Text Processing
# ============================================================================


def separation_fn(raw_text: str, max_length: int = 35) -> List[str]:
    """
    Split text into manageable sentences for video generation.

    Args:
        raw_text: Raw story text
        max_length: Maximum length per sentence segment

    Returns:
        List of sentence segments
    """

    def split(txt: str, separator: str) -> List[str]:
        return [x for x in txt.split(separator) if len(x.replace(" ", "")) > 0]

    parts = []
    for p in split(raw_text, "\n"):
        if len(p) > max_length:
            broken_down = False
            for sep in [".", "!", "?"]:
                sub_parts = split(p, sep)
                if len(sub_parts) > 1:
                    broken_down = True
                    parts.extend(sub_parts)
                    break
            if not broken_down:
                parts.append(p)
        else:
            parts.append(p)

    return parts


# ============================================================================
# Story Generation
# ============================================================================


async def generate_story(
    title: str,
    llm: LLMInterface,
    prompt_provider: PromptProviderInterface,
    config: VideoGenerationConfig,
    progress: Optional[ProgressCallback] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> StoryOutput:
    """
    Generate a story based on a title using LLM with structured output.

    Args:
        title: The title/topic for the story
        llm: LLM interface implementation
        prompt_provider: Prompt provider interface
        config: Configuration object
        progress: Optional progress callback

    Returns:
        StoryOutput with title, story_plan, and dialog
    """
    if progress:
        progress.update("Generating story from title...")

    # Get prompt template
    prompt_template = prompt_provider.get_raw_prompt("story_generation")

    # Format with title - add instructions for structured output
    base_prompt = (
        prompt_template.replace("{title}", title)
        if "{title}" in prompt_template
        else f"{prompt_template}\n\nScenario: {title}"
    )

    # Add structured output instructions
    full_prompt = f"""{base_prompt}

IMPORTANT: Your response must be structured with three parts:

1. **title**: Create a refined, more complete title for the story (based on the user's input: "{title}")
2. **story_plan**: Describe your overall plan and reasoning for creating this dialog (like a thinking process - what makes this scenario funny, what progression you're following, key elements you're including)
3. **dialog**: The actual dialog lines produced by Fred (and potentially other characters), following all the rules above

Focus on:
- Making the refined title catchy and descriptive
- In story_plan, explain your creative choices and the comedic arc
- In dialog, provide only the spoken lines (no stage directions or descriptions)"""

    # Generate structured story (with concurrency control if semaphore provided)
    if semaphore:
        story_output = await with_semaphore(
            semaphore, llm.complete_structured(full_prompt, StoryOutput)
        )
    else:
        story_output = await llm.complete_structured(full_prompt, StoryOutput)

    if progress:
        progress.update(f"Story generated: {story_output.title}")

    return story_output


# ============================================================================
# Video Matching and Judgement
# ============================================================================


def extract_middle_frame(video_path: str) -> Optional[str]:
    """Extract the middle frame from a video and return as base64 encoded string."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2

        # Set position to middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        return base64_image
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return None


JUDGE_PROMPT = """You are a contextual image rater. You grade if an image where a character is located and speaks a line of dialogue is contextual or not.

Examples:
- The character is in space. The character's line of dialogue speaks about nutrition => NOT_CONTEXTUAL
- The character is with a horse, the character speaks about horse races => CONTEXTUAL
- The character talks about gambling while facing the camera in a bar => NEUTRAL (vaguely related)

Please provide:
Rating: CONTEXTUAL/NEUTRAL/NOT_CONTEXTUAL
Grade: count the factors supporting one rating (for ranking)
Reasoning: brief explanation

Dialogue line: {dialogue}
"""


async def judge_video_match(
    video_path: str,
    dialogue: str,
    llm: LLMInterface,
    config: VideoGenerationConfig,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Optional[VideoJudgementResult]:
    """
    Judge if a video matches a dialogue using vision LLM.

    Args:
        video_path: Path to video file
        dialogue: Dialogue text
        llm: LLM interface with vision support
        config: Configuration

    Returns:
        VideoJudgementResult or None if failed
    """
    # Extract middle frame
    base64_image = extract_middle_frame(video_path)
    if not base64_image:
        return None

    # Construct prompt
    prompt = JUDGE_PROMPT.format(dialogue=dialogue)

    try:
        # Call vision API (with concurrency control if semaphore provided)
        if semaphore:
            response = await with_semaphore(
                semaphore, llm.complete_with_vision(prompt, base64_image)
            )
        else:
            response = await llm.complete_with_vision(prompt, base64_image)

        # Parse response
        rating = "NOT_CONTEXTUAL"
        grade = 0
        reasoning = response

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("Rating"):
                rating_text = line.split(":")[-1].strip()
                if "CONTEXTUAL" in rating_text and "NOT" not in rating_text:
                    rating = "CONTEXTUAL"
                elif "NEUTRAL" in rating_text:
                    rating = "NEUTRAL"
                else:
                    rating = "NOT_CONTEXTUAL"
            elif line.startswith("Grade"):
                grade_text = line.split(":")[-1].strip()
                try:
                    grade = int(grade_text)
                except:
                    grade = 0

        return VideoJudgementResult(
            video_path=video_path, rating=rating, grade=grade, reasoning=response
        )

    except Exception as e:
        print(f"Error judging video {video_path}: {e}")
        return None


KEYWORD_GENERATION_PROMPT = """You are a search keyword generator for finding video clips from the French educational TV show "C'est pas Sorcier".

Given a dialogue line spoken by Fred (the presenter), generate a concise search keyword or phrase that would help find a relevant video clip where Fred could be saying this line.

The keyword should focus on:
- The main topic or subject matter of the dialogue
- Visual elements that would match the context
- Locations or settings mentioned
- Actions or activities described

Keep the keyword short (1-5 words) and in French.

Previous search attempts that did not yield satisfactory results:
{previous_keywords}

Dialogue line: {dialogue}

Generate a NEW search keyword that is different from the previous attempts and might find a better matching video clip.

Return ONLY the keyword/phrase, nothing else.
"""


async def generate_search_keyword(
    dialogue: str,
    previous_keywords: List[str],
    llm: LLMInterface,
    config: VideoGenerationConfig,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> str:
    """
    Generate a search keyword for finding relevant video clips.

    Args:
        dialogue: The dialogue line
        previous_keywords: Previously tried keywords
        llm: LLM interface
        config: Configuration

    Returns:
        Generated keyword
    """
    previous_str = (
        "\n".join([f"- {kw}" for kw in previous_keywords])
        if previous_keywords
        else "None"
    )

    prompt = KEYWORD_GENERATION_PROMPT.format(
        previous_keywords=previous_str, dialogue=dialogue
    )

    # Call LLM (with concurrency control if semaphore provided)
    if semaphore:
        keyword = await with_semaphore(semaphore, llm.complete(prompt))
    else:
        keyword = await llm.complete(prompt)
    return keyword.strip()


