"""
Core logic for async video generation.

This module contains the main workflow functions:
- Story generation from title
- Video generation from story (with async optimization)
- Sentence processing (parallel LLM, serial TTS/STT)
- Config dumping for reproducibility
"""

import os
import asyncio
import time
import cv2
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import json

from virtual_streamer.video_generation.interfaces import (
    LLMInterface,
    TTSInterface,
    STTInterface,
    VideoRetrieverInterface,
    PromptProviderInterface,
    VideoJudgementResult,
    VideoMatchResult,
    ProgressCallback,
)
from virtual_streamer.video_generation.config import (
    VideoGenerationConfig,
    ConfigDump,
    GenerationResult,
    StoryOutput,
)
from virtual_streamer.utils.utils import (
    combine_video_and_short_audio,
    add_subtitle_from_srt,
    combine_part_in_concat_file,
    get_length,
)
from virtual_streamer.utils.minio_client import get_storage_client


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


async def find_best_video_for_sentence(
    sentence: str,
    llm: LLMInterface,
    video_retriever: VideoRetrieverInterface,
    config: VideoGenerationConfig,
    progress: Optional[ProgressCallback] = None,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> VideoMatchResult:
    """
    Find the best matching video for a sentence using parallel LLM calls.

    Args:
        sentence: Sentence text
        llm: LLM interface
        video_retriever: Video retriever
        config: Configuration
        progress: Optional progress callback

    Returns:
        VideoMatchResult with best match
    """
    if progress:
        progress.update(f"Searching videos for: {sentence[:50]}...")

    # Initial search with sentence as query
    videos = video_retriever.search(sentence, config.video_retrieval.top_k)

    if not videos:
        # No videos found, return empty result
        return VideoMatchResult(
            sentence=sentence,
            selected_video="",
            rating="NOT_CONTEXTUAL",
            grade=0,
            reasoning="No videos found",
            alternatives_tried=[],
        )

    # Judge top videos in parallel (with semaphore controlling concurrency)
    judgement_tasks = [
        judge_video_match(video, sentence, llm, config, semaphore)
        for video in videos[: config.max_video_judgement_attempts]
    ]
    judgements = await asyncio.gather(*judgement_tasks)

    # Filter out None results
    valid_judgements = [j for j in judgements if j is not None]

    # Find best match
    best_judgement = None
    for j in valid_judgements:
        if j.rating == "CONTEXTUAL":
            best_judgement = j
            break
        elif j.rating == "NEUTRAL" and (
            not best_judgement or best_judgement.rating == "NOT_CONTEXTUAL"
        ):
            best_judgement = j
        elif not best_judgement or j.grade > best_judgement.grade:
            best_judgement = j

    alternatives_tried = []

    # If no good match, try alternative keywords (parallel)
    if best_judgement and best_judgement.rating == "NOT_CONTEXTUAL":
        if progress:
            progress.update("Trying alternative search keywords...")

        # Generate alternative keywords in parallel (with semaphore controlling concurrency)
        keyword_tasks = [
            generate_search_keyword(
                sentence, alternatives_tried, llm, config, semaphore
            )
            for _ in range(config.max_search_attempts)
        ]
        keywords = await asyncio.gather(*keyword_tasks)

        # Search and judge with each keyword
        for keyword in keywords:
            alternatives_tried.append(keyword)

            alt_videos = video_retriever.search(keyword, config.video_retrieval.top_k)
            if alt_videos:
                alt_judgement_tasks = [
                    judge_video_match(video, sentence, llm, config, semaphore)
                    for video in alt_videos[: config.max_video_judgement_attempts]
                ]
                alt_judgements = await asyncio.gather(*alt_judgement_tasks)
                alt_valid = [j for j in alt_judgements if j is not None]

                for j in alt_valid:
                    if j.rating in ["CONTEXTUAL", "NEUTRAL"]:
                        best_judgement = j
                        break

                if best_judgement and best_judgement.rating in [
                    "CONTEXTUAL",
                    "NEUTRAL",
                ]:
                    break

    # Return best match found
    if best_judgement:
        return VideoMatchResult(
            sentence=sentence,
            selected_video=best_judgement.video_path,
            rating=best_judgement.rating,
            grade=best_judgement.grade,
            reasoning=best_judgement.reasoning,
            alternatives_tried=alternatives_tried,
        )
    else:
        # Fallback to first video
        return VideoMatchResult(
            sentence=sentence,
            selected_video=videos[0] if videos else "",
            rating="NOT_CONTEXTUAL",
            grade=0,
            reasoning="No judgement available, using fallback",
            alternatives_tried=alternatives_tried,
        )


# ============================================================================
# Video Generation
# ============================================================================


async def generate_video_from_story(
    story: str,
    llm: LLMInterface,
    tts: TTSInterface,
    stt: STTInterface,
    video_retriever: VideoRetrieverInterface,
    config: VideoGenerationConfig,
    progress: Optional[ProgressCallback] = None,
    story_output: Optional[StoryOutput] = None,
) -> GenerationResult:
    """
    Generate a complete video from a story with audio and subtitles.

    This function orchestrates the entire video generation pipeline:
    1. Split story into sentences
    2. Find matching videos (PARALLEL LLM calls)
    3. Generate audio for each sentence (SERIAL, local)
    4. Generate subtitles (SERIAL, local)
    5. Combine segments
    6. Create final video
    7. Generate comprehensive config dump

    Args:
        story: The story text
        llm: LLM interface
        tts: TTS interface
        stt: STT interface
        video_retriever: Video retriever interface
        config: Configuration
        progress: Optional progress callback

    Returns:
        GenerationResult with video path and metadata
    """
    start_time = time.time()
    timing = {}

    # Split story into sentences
    if progress:
        progress.update("Splitting story into sentences...")

    sentences = separation_fn(story, config.max_sentence_length)

    if progress:
        progress.set_total_steps(
            len(sentences) * 4 + 2
        )  # Video search + audio + subtitle + combine + final
        progress.update(f"Processing {len(sentences)} sentences")

    # Create semaphore for LLM concurrency control
    llm_semaphore = asyncio.Semaphore(config.max_parallel_llm_calls)

    # Phase 1: Find matching videos for all sentences (PARALLEL LLM calls with semaphore)
    if progress:
        progress.update(
            f"Phase 1: Finding matching videos (parallel, max {config.max_parallel_llm_calls} concurrent)..."
        )

    phase1_start = time.time()
    video_match_tasks = [
        find_best_video_for_sentence(
            sentence, llm, video_retriever, config, progress, llm_semaphore
        )
        for sentence in sentences
    ]
    video_matches = await asyncio.gather(*video_match_tasks)
    timing["video_search"] = time.time() - phase1_start

    # Phase 2: Generate audio for all sentences (SERIAL, local processing)
    if progress:
        progress.update("Phase 2: Generating audio (serial)...")

    phase2_start = time.time()
    audio_files = []
    for i, sentence in enumerate(sentences):
        if progress:
            progress.increment_step(f"Generating audio {i + 1}/{len(sentences)}")

        audio_path = config.get_temp_path(f"audio_{i}_{hash(sentence)}.wav")
        audio_path = tts.generate_speech(sentence, audio_path)
        audio_files.append(audio_path)
    timing["audio_generation"] = time.time() - phase2_start

    # Phase 3: Generate subtitles (SERIAL, local processing)
    if progress:
        progress.update("Phase 3: Generating subtitles (serial)...")

    phase3_start = time.time()
    subtitle_files = []
    for i, audio_path in enumerate(audio_files):
        if progress:
            progress.increment_step(f"Generating subtitles {i + 1}/{len(audio_files)}")

        srt_path = config.get_temp_path(f"subtitle_{i}.srt")
        srt_path = stt.transcribe_to_srt(audio_path, srt_path)
        subtitle_files.append(srt_path)
    timing["subtitle_generation"] = time.time() - phase3_start

    # Phase 4: Combine video + audio + subtitles for each segment (SERIAL)
    if progress:
        progress.update("Phase 4: Combining segments...")

    phase4_start = time.time()
    video_segments = []
    for i, (sentence, match, audio, subtitle) in enumerate(
        zip(sentences, video_matches, audio_files, subtitle_files)
    ):
        if progress:
            progress.increment_step(f"Combining segment {i + 1}/{len(sentences)}")

        if not match.selected_video:
            print(f"Warning: No video for sentence {i}, skipping")
            continue

        # Combine video and audio
        combined_path = config.get_temp_path(f"combined_{i}.mp4")
        combine_video_and_short_audio(match.selected_video, audio, combined_path)

        # Add subtitles
        subtitled_path = config.get_temp_path(f"segment_{i}.mp4")
        add_subtitle_from_srt(
            combined_path,
            subtitle,
            subtitled_path,
            fontsize=config.video_processing.fontsize,
        )

        video_segments.append(subtitled_path)
    timing["segment_composition"] = time.time() - phase4_start

    # Phase 5: Final concatenation
    if progress:
        progress.update("Phase 5: Creating final video...")

    phase5_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = config.get_output_path(f"video_{timestamp}.mp4")
    concat_file = config.get_temp_path("concat_list.txt")

    combine_part_in_concat_file(video_segments, concat_file, final_video_path)
    timing["final_concatenation"] = time.time() - phase5_start

    # Phase 6: Upload final video to MinIO storage
    if progress:
        progress.update("Phase 6: Uploading to storage...")

    phase6_start = time.time()
    storage = get_storage_client()
    minio_video_key = f"videos/{timestamp}/video_{timestamp}.mp4"
    await storage.upload_file(final_video_path, minio_video_key)
    timing["storage_upload"] = time.time() - phase6_start
    timing["total"] = time.time() - start_time

    # Create comprehensive config dump
    config_dump = create_config_dump(
        story=story,
        sentences=sentences,
        video_matches=video_matches,
        audio_files=audio_files,
        subtitle_files=subtitle_files,
        video_segments=video_segments,
        config=config,
        final_video_path=final_video_path,
        timing=timing,
        story_output=story_output,
    )

    # Save config dump if enabled and upload to MinIO
    config_dump_path = None
    minio_config_key = None
    if config.enable_config_dump:
        config_dump_path = config.get_output_path(f"config_{timestamp}.json")
        config_dump.save(config_dump_path)
        # Upload config dump to MinIO
        minio_config_key = f"videos/{timestamp}/config_{timestamp}.json"
        await storage.upload_file(config_dump_path, minio_config_key)

    if progress:
        progress.update(f"Video generated successfully in {timing['total']:.2f}s")

    return GenerationResult(
        video_path=final_video_path,
        config_dump_path=config_dump_path,
        story_output=story_output,  # Include structured story output
        metadata={
            "sentence_count": len(sentences),
            "total_duration": get_length(final_video_path),
            "timestamp": datetime.now().isoformat(),
            "timing": timing,
            "minio_video_key": minio_video_key,
            "minio_config_key": minio_config_key,
        },
    )


def create_config_dump(
    story: str,
    sentences: List[str],
    video_matches: List[VideoMatchResult],
    audio_files: List[str],
    subtitle_files: List[str],
    video_segments: List[str],
    config: VideoGenerationConfig,
    final_video_path: str,
    timing: Dict[str, float],
    story_output: Optional[StoryOutput] = None,
) -> ConfigDump:
    """
    Create a comprehensive config dump for reproducibility.

    This dump includes all information needed to recreate the video
    without redoing LLM/API calls.
    """
    input_data = {"story": story, "sentences": sentences}

    # Add structured story output if available
    if story_output:
        input_data["story_output"] = {
            "title": story_output.title,
            "story_plan": story_output.story_plan,
            "dialog": story_output.dialog,
        }

    return ConfigDump(
        version="1.0",
        timestamp=datetime.now().isoformat(),
        input=input_data,
        config=config.to_dict(),
        execution={
            "video_matches": [match.to_dict() for match in video_matches],
            "audio_files": audio_files,
            "subtitle_files": subtitle_files,
            "video_segments": video_segments,
            "sentences_with_details": [
                {
                    "index": i,
                    "text": sent,
                    "video": match.to_dict(),
                    "audio": audio,
                    "subtitle": subtitle,
                    "segment": segment,
                }
                for i, (sent, match, audio, subtitle, segment) in enumerate(
                    zip(
                        sentences,
                        video_matches,
                        audio_files,
                        subtitle_files,
                        video_segments,
                    )
                )
            ],
        },
        output={
            "final_video": final_video_path,
            "duration": get_length(final_video_path),
            "file_size": os.path.getsize(final_video_path),
        },
        models={
            "llm": {
                "provider": config.llm.provider,
                "model": config.llm.model,
                "temperature": config.llm.temperature,
                "vision_model": config.llm.vision_model,
            },
            "tts": {
                "provider": config.tts.provider,
                "host": config.tts.host,
                "port": config.tts.port,
            },
            "stt": {"provider": config.stt.provider, "model": config.stt.model},
            "video_retrieval": {
                "method": config.video_retrieval.method,
                "index_path": config.video_retrieval.index_path,
            },
        },
        timing=timing,
    )


async def recreate_from_config_dump(
    config_dump_path: str,
    tts: TTSInterface,
    stt: STTInterface,
    config: VideoGenerationConfig,
    progress: Optional[ProgressCallback] = None,
) -> GenerationResult:
    """
    Recreate a video from a config dump without redoing LLM calls.

    This skips expensive API calls and just regenerates audio/subtitles
    and recombines the segments.

    Args:
        config_dump_path: Path to config dump JSON
        tts: TTS interface
        stt: STT interface
        config: Configuration (for output paths)
        progress: Optional progress callback

    Returns:
        GenerationResult with new video
    """
    if progress:
        progress.update("Loading config dump...")

    dump = ConfigDump.load(config_dump_path)

    sentences = dump.input["sentences"]
    execution = dump.execution

    if progress:
        progress.set_total_steps(len(sentences) * 3 + 1)

    # Reuse video selections from dump
    video_paths = [match["selected_video"] for match in execution["video_matches"]]

    # Regenerate audio (serial)
    if progress:
        progress.update("Regenerating audio...")

    audio_files = []
    for i, sentence in enumerate(sentences):
        if progress:
            progress.increment_step(f"Audio {i + 1}/{len(sentences)}")
        audio_path = config.get_temp_path(f"audio_regen_{i}.wav")
        audio_path = tts.generate_speech(sentence, audio_path)
        audio_files.append(audio_path)

    # Regenerate subtitles (serial)
    if progress:
        progress.update("Regenerating subtitles...")

    subtitle_files = []
    for i, audio_path in enumerate(audio_files):
        if progress:
            progress.increment_step(f"Subtitles {i + 1}/{len(audio_files)}")
        srt_path = config.get_temp_path(f"subtitle_regen_{i}.srt")
        srt_path = stt.transcribe_to_srt(audio_path, srt_path)
        subtitle_files.append(srt_path)

    # Recombine segments
    if progress:
        progress.update("Recombining segments...")

    video_segments = []
    for i, (video, audio, subtitle) in enumerate(
        zip(video_paths, audio_files, subtitle_files)
    ):
        if progress:
            progress.increment_step(f"Segment {i + 1}/{len(video_paths)}")

        if not video:
            continue

        combined_path = config.get_temp_path(f"combined_regen_{i}.mp4")
        combine_video_and_short_audio(video, audio, combined_path)

        subtitled_path = config.get_temp_path(f"segment_regen_{i}.mp4")
        add_subtitle_from_srt(combined_path, subtitle, subtitled_path)

        video_segments.append(subtitled_path)

    # Final video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = config.get_output_path(f"video_recreated_{timestamp}.mp4")
    concat_file = config.get_temp_path("concat_list_regen.txt")

    combine_part_in_concat_file(video_segments, concat_file, final_video_path)

    if progress:
        progress.update("Video recreated successfully!")

    return GenerationResult(
        video_path=final_video_path,
        metadata={
            "recreated_from": config_dump_path,
            "sentence_count": len(sentences),
            "total_duration": get_length(final_video_path),
        },
    )
