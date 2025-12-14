"""
Sentence Processor Agent.

This is the only custom agent in the video generation pipeline.
It orchestrates sentence-level processing with:
- Loops over sentences
- Dynamic ParallelAgent creation for video matching
- Deterministic TTS/STT/video combining

This agent uses custom _run_async_impl because it needs complex
iteration and dynamic agent creation that can't be expressed with
standard ADK patterns.
"""

import logging
import secrets
from typing import Any, AsyncGenerator, Dict, List, Optional

from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from virtual_streamer.agents.common.state_keys import (
    SENTENCES,
    VIDEO_MATCHES,
    AUDIO_FILES,
    SUBTITLE_FILES,
    VIDEO_SEGMENTS,
    task_key,
    result_key,
    keyword_key,
)
from virtual_streamer.agents.common.utils import (
    extract_middle_frame,
    combine_segment,
)
from virtual_streamer.agents.video_matcher import get_video_matcher
from virtual_streamer.agents.keyword_generator import get_keyword_generator
from virtual_streamer.video_generation.interfaces import (
    VideoRetrieverInterface,
    TTSInterface,
    STTInterface,
)

logger = logging.getLogger(__name__)


class SentenceProcessorAgent(BaseAgent):
    """
    Custom agent that orchestrates sentence-level video generation.
    
    For each sentence:
    1. Retrieves candidate videos from the index
    2. Creates VideoMatcherAgents and runs them in parallel
    3. Selects the best matching video
    4. Optionally generates alternative keywords and retries
    5. Generates audio using TTS
    6. Generates subtitles using STT
    7. Combines video + audio + subtitles into a segment
    
    This is the only agent that requires custom _run_async_impl because
    it has complex iteration and dynamic agent creation.
    """
    
    def __init__(
        self,
        video_retriever: VideoRetrieverInterface,
        tts: TTSInterface,
        stt: STTInterface,
        output_dir: str,
        temp_dir: str,
        max_video_candidates: int = 5,
        max_search_attempts: int = 3,
        fontsize: int = 14,
    ):
        """
        Initialize the sentence processor.
        
        Args:
            video_retriever: Interface for searching video clips
            tts: Interface for text-to-speech
            stt: Interface for speech-to-text (subtitles)
            output_dir: Directory for output files
            temp_dir: Directory for temporary files
            max_video_candidates: Max videos to judge per sentence
            max_search_attempts: Max keyword generation attempts
            fontsize: Font size for subtitles
        """
        super().__init__(name="sentence_processor")
        self.video_retriever = video_retriever
        self.tts = tts
        self.stt = stt
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.max_video_candidates = max_video_candidates
        self.max_search_attempts = max_search_attempts
        self.fontsize = fontsize
    
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Process all sentences and generate video segments.
        
        This method:
        1. Iterates over all sentences
        2. For each sentence, runs parallel video matching
        3. Generates audio and subtitles
        4. Combines into segments
        5. Stores all results in state
        
        Args:
            ctx: Invocation context with session state
        
        Yields:
            Events for state updates and sub-agent execution
        """
        sentences = ctx.session.state.get(SENTENCES, [])
        
        if not sentences:
            logger.warning("No sentences found in state")
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text="No sentences to process")],
                ),
            )
            return
        
        logger.info(f"Processing {len(sentences)} sentences")
        
        # Results accumulators
        all_matches: List[Dict[str, Any]] = []
        all_audio: List[str] = []
        all_subtitles: List[str] = []
        all_segments: List[str] = []
        
        for idx, sentence in enumerate(sentences):
            run_id = f"s{idx}"
            
            # Yield progress event
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"Processing sentence {idx + 1}/{len(sentences)}")],
                ),
            )
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Retrieve candidate videos
            # ═══════════════════════════════════════════════════════════════
            candidates = self.video_retriever.search(sentence, top_k=10)
            
            if not candidates:
                raise Exception("No candidate for this sentence : ", sentence)
            
            # Extract frames for vision LLM
            frames = {}
            for vp in candidates[:self.max_video_candidates]:
                frame = extract_middle_frame(vp)
                if frame:
                    frames[vp] = frame
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Set up tasks in state for parallel matchers
            # ═══════════════════════════════════════════════════════════════
            task_delta = {task_key(run_id, "sentence"): sentence}
            matchers = []
            
            for v_idx, video_path in enumerate(list(frames.keys())):
                worker_name = f"m{v_idx}"
                task_delta[task_key(run_id, f"{worker_name}:video")] = video_path
                task_delta[task_key(run_id, f"{worker_name}:frame")] = frames[video_path]
                matchers.append(get_video_matcher(run_id, worker_name))
            
            # Yield state update event
            yield Event(
                author=self.name,
                content=types.Content(
                    role=self.name,
                    parts=[types.Part(text=f"Matching {len(matchers)} videos for sentence {idx}")],
                ),
                actions=EventActions(state_delta=task_delta),
            )
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Run matchers in parallel
            # ═══════════════════════════════════════════════════════════════
            if matchers:
                parallel = ParallelAgent(
                    name=f"parallel_{run_id}",
                    sub_agents=matchers,
                )
                async for ev in parallel.run_async(ctx):
                    yield ev
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 4: Collect results and select best match
            # ═══════════════════════════════════════════════════════════════
            judgements = []
            for key, value in ctx.session.state.items():
                if key.startswith(f"result:{run_id}:"):
                    judgements.append(value)
            
            best = self._select_best_match(judgements)
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 5: Retry with alternative keywords if needed
            # ═══════════════════════════════════════════════════════════════
            prev_keywords = [sentence]
            
            if best["rating"] == "NOT_CONTEXTUAL" and self.max_search_attempts > 0:
                for attempt in range(self.max_search_attempts):
                    kw_run_id = f"{run_id}_kw{attempt}"
                    
                    # Set up keyword generation task
                    yield Event(
                        author=self.name,
                        actions=EventActions(state_delta={
                            task_key(kw_run_id, "sentence"): sentence,
                            task_key(kw_run_id, "prev_keywords"): prev_keywords,
                        }),
                    )
                    
                    # Run keyword generator
                    kw_agent = get_keyword_generator(kw_run_id)
                    async for ev in kw_agent.run_async(ctx):
                        yield ev
                    
                    # Get generated keyword
                    new_keyword = ctx.session.state.get(keyword_key(kw_run_id), "")
                    if new_keyword:
                        prev_keywords.append(new_keyword)
                        
                        # Search with new keyword
                        alt_candidates = self.video_retriever.search(new_keyword, top_k=5)
                        if alt_candidates:
                            # Extract frames and set up matchers
                            alt_run_id = f"{run_id}_alt{attempt}"
                            alt_task_delta = {task_key(alt_run_id, "sentence"): sentence}
                            alt_matchers = []
                            
                            for v_idx, vp in enumerate(alt_candidates[:3]):
                                frame = extract_middle_frame(vp)
                                if frame:
                                    worker_name = f"m{v_idx}"
                                    alt_task_delta[task_key(alt_run_id, f"{worker_name}:video")] = vp
                                    alt_task_delta[task_key(alt_run_id, f"{worker_name}:frame")] = frame
                                    alt_matchers.append(get_video_matcher(alt_run_id, worker_name))
                            
                            if alt_matchers:
                                yield Event(
                                    author=self.name,
                                    actions=EventActions(state_delta=alt_task_delta),
                                )
                                
                                parallel = ParallelAgent(
                                    name=f"parallel_{alt_run_id}",
                                    sub_agents=alt_matchers,
                                )
                                async for ev in parallel.run_async(ctx):
                                    yield ev
                                
                                # Check if we found a better match
                                alt_judgements = []
                                for key, value in ctx.session.state.items():
                                    if key.startswith(f"result:{alt_run_id}:"):
                                        alt_judgements.append(value)
                                
                                alt_best = self._select_best_match(alt_judgements)
                                if alt_best["rating"] in ["CONTEXTUAL", "NEUTRAL"]:
                                    best = alt_best
                                    break
                    
                    # Stop if we found a good match
                    if best["rating"] in ["CONTEXTUAL", "NEUTRAL"]:
                        break
            
            # Add sentence to match result
            best["sentence"] = sentence
            all_matches.append(best)
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 6: Generate audio using TTS
            # ═══════════════════════════════════════════════════════════════
            import os
            os.makedirs(self.temp_dir, exist_ok=True)
            
            audio_path = os.path.join(self.temp_dir, f"audio_{idx}.wav")
            try:
                audio_path = self.tts.generate_speech(sentence, audio_path)
                all_audio.append(audio_path)
            except Exception as e:
                logger.error(f"TTS failed for sentence {idx}: {e}")
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 7: Generate subtitles using STT
            # ═══════════════════════════════════════════════════════════════
            subtitle_path = os.path.join(self.temp_dir, f"subtitle_{idx}.srt")
            try:
                subtitle_path = self.stt.transcribe_to_srt(audio_path, subtitle_path)
                all_subtitles.append(subtitle_path)
            except Exception as e:
                logger.error(f"STT failed for sentence {idx}: {e}")
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 8: Combine video + audio + subtitles
            # ═══════════════════════════════════════════════════════════════
            if best["video_path"]:
                segment_path = os.path.join(self.temp_dir, f"segment_{idx}.mp4")
                try:
                    segment_path = combine_segment(
                        best["video_path"],
                        audio_path,
                        subtitle_path,
                        segment_path,
                        fontsize=self.fontsize,
                    )
                    all_segments.append(segment_path)
                except Exception as e:
                    logger.error(f"Segment combination failed for sentence {idx}: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 9: Store all results in state
        # ═══════════════════════════════════════════════════════════════
        yield Event(
            author=self.name,
            content=types.Content(
                role=self.name,
                parts=[types.Part(text=f"Completed processing {len(sentences)} sentences")],
            ),
            actions=EventActions(state_delta={
                VIDEO_MATCHES: all_matches,
                AUDIO_FILES: all_audio,
                SUBTITLE_FILES: all_subtitles,
                VIDEO_SEGMENTS: all_segments,
            }),
        )
        
        logger.info(
            f"Sentence processing complete: "
            f"{len(all_segments)} segments created"
        )
    
    def _select_best_match(self, judgements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best video match from a list of judgements.
        
        Priority:
        1. CONTEXTUAL rating (best)
        2. NEUTRAL rating
        3. Highest grade among NOT_CONTEXTUAL
        
        Args:
            judgements: List of judgement dicts with rating, grade, video_path
        
        Returns:
            Best matching judgement dict
        """
        if not judgements:
            return {
                "video_path": "",
                "rating": "NOT_CONTEXTUAL",
                "grade": 0,
                "reasoning": "No judgements available",
            }
        
        # Find CONTEXTUAL matches
        for j in judgements:
            if j.get("rating") == "CONTEXTUAL":
                return j
        
        # Find NEUTRAL matches
        for j in judgements:
            if j.get("rating") == "NEUTRAL":
                return j
        
        # Return highest grade among NOT_CONTEXTUAL
        return max(judgements, key=lambda x: x.get("grade", 0))

root_agent = None