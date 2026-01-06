"""
Concrete implementations of video generation interfaces.

This module provides working implementations for:
- LLM providers (Anthropic, OpenAI, LiteLLM)
- TTS providers (Fish-Speech, Solero, Coqui)
- STT providers (Whisper, Faster-Whisper)
- Video retrievers (VideoSearchRetriever using remote VideoSearchClient)
- Prompt providers (Local, MLflow)
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
import anthropic
import openai
from litellm import completion as litellm_completion, acompletion as litellm_acompletion
import stable_whisper

from virtual_streamer.video_generation.interfaces import (
    LLMInterface,
    TTSInterface,
    STTInterface,
    VideoRetrieverInterface,
    PromptProviderInterface,
)
from virtual_streamer.video_generation.config import (
    LLMConfig,
    TTSConfig,
    STTConfig,
    VideoRetrievalConfig,
    PromptConfig,
)
from virtual_streamer.utils.utils import txt_to_speech_call_fish, txt_to_speech_call_fish_async, get_length
from virtual_streamer.video_server.models import Character
from virtual_streamer.api.dependencies import get_path_resolver
from virtual_streamer.api.clients.character_client import CharacterClient
from virtual_streamer.video_search.client import VideoSearchClient, VideoSearchResult


# ============================================================================
# LLM Implementations
# ============================================================================


class AnthropicLLM(LLMInterface):
    """Anthropic Claude implementation."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._async_client = None

    @property
    def async_client(self):
        """Lazy initialization of async client to avoid event loop conflicts."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._async_client

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion asynchronously."""
        response = await self.async_client.messages.create(
            model=self.config.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def complete_structured(
        self, prompt: str, response_model: Type[BaseModel], **kwargs
    ) -> BaseModel:
        """Generate structured completion using Anthropic's beta feature."""
        # Use prompt engineering to get structured output
        # Anthropic doesn't have native structured output yet, so we'll use JSON mode
        schema = response_model.model_json_schema()

        structured_prompt = f"""{prompt}

Please respond with a JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with valid JSON, no other text."""

        response = await self.async_client.messages.create(
            model=self.config.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": structured_prompt}],
        )

        # Parse JSON response
        response_text = response.content[0].text
        # Try to extract JSON if it's wrapped in markdown
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        response_data = json.loads(response_text)
        return response_model(**response_data)

    async def complete_with_vision(
        self, prompt: str, image_base64: str, **kwargs
    ) -> str:
        """Generate completion with vision input."""
        response = await self.async_client.messages.create(
            model=kwargs.get("model", self.config.vision_model),
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text


class OpenAILLM(LLMInterface):
    """OpenAI GPT implementation."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of async client to avoid event loop conflicts."""
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion asynchronously."""
        response = await self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def complete_structured(
        self, prompt: str, response_model: Type[BaseModel], **kwargs
    ) -> BaseModel:
        """Generate structured completion using OpenAI's structured outputs."""
        # OpenAI supports structured outputs natively with response_format
        response = await self.client.beta.chat.completions.parse(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        return response.choices[0].message.parsed

    async def complete_with_vision(
        self, prompt: str, image_base64: str, **kwargs
    ) -> str:
        """Generate completion with vision input."""
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4-vision-preview"),
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content


class LiteLLM(LLMInterface):
    """LiteLLM multi-provider implementation."""

    def __init__(self, config: LLMConfig):
        self.config = config
        if config.api_key:
            os.environ["OPENAI_API_KEY"] = config.api_key

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate text completion asynchronously."""
        response = await litellm_acompletion(
            model=self.config.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def complete_structured(
        self, prompt: str, response_model: Type[BaseModel], **kwargs
    ) -> BaseModel:
        """Generate structured completion using prompt engineering."""
        schema = response_model.model_json_schema()

        structured_prompt = f"""{prompt}

Please respond with a JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with valid JSON, no other text."""

        response = await litellm_acompletion(
            model=self.config.model,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": structured_prompt}],
        )

        response_text = response.choices[0].message.content
        # Try to extract JSON if it's wrapped in markdown
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        response_data = json.loads(response_text)
        return response_model(**response_data)

    async def complete_with_vision(
        self, prompt: str, image_base64: str, **kwargs
    ) -> str:
        """Generate completion with vision input."""
        # LiteLLM supports vision through specific models
        response = await litellm_acompletion(
            model=kwargs.get("model", self.config.vision_model),
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content


# ============================================================================
# TTS Implementations
# ============================================================================


class FishSpeechTTS(TTSInterface):
    """Fish-Speech TTS implementation."""

    def __init__(self, config: TTSConfig):
        self.config = config

    def generate_speech(
        self, text: str, output_path: Optional[str] = None, **kwargs
    ) -> str:
        """Generate speech using Fish-Speech API (synchronous)."""
        return txt_to_speech_call_fish(
            speech_lines=text,
            reference_audio=self.config.reference_audio,
            reference_text=self.config.reference_text,
            outpath=output_path,
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            chunk_length=kwargs.get("chunk_length", self.config.chunk_length),
            top_p=kwargs.get("top_p", self.config.top_p),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.config.repetition_penalty
            ),
            temperature=kwargs.get("temperature", self.config.temperature),
            host=self.config.host,
            port=self.config.port,
        )

    async def generate_speech_async(
        self, text: str, output_path: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generate speech using Fish-Speech API (asynchronous).
        
        Native async implementation using httpx for non-blocking HTTP requests.
        """
        return await txt_to_speech_call_fish_async(
            speech_lines=text,
            reference_audio=self.config.reference_audio,
            reference_text=self.config.reference_text,
            outpath=output_path,
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            chunk_length=kwargs.get("chunk_length", self.config.chunk_length),
            top_p=kwargs.get("top_p", self.config.top_p),
            repetition_penalty=kwargs.get(
                "repetition_penalty", self.config.repetition_penalty
            ),
            temperature=kwargs.get("temperature", self.config.temperature),
            host=self.config.host,
            port=self.config.port,
        )

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file."""
        return get_length(audio_path)


class SoleroTTS(TTSInterface):
    """Solero TTS implementation."""

    def __init__(self, config: TTSConfig):
        self.config = config

    def generate_speech(
        self, text: str, output_path: Optional[str] = None, **kwargs
    ) -> str:
        """Generate speech using Solero TTS API."""
        import requests
        import tempfile

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        url = f"http://{self.config.host}:{self.config.port}/tts/generate"
        data = {
            "text": text,
            "speaker": kwargs.get("speaker", "default"),
            "session": kwargs.get("session", "default"),
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file."""
        return get_length(audio_path)


class CoquiTTS(TTSInterface):
    """Coqui TTS implementation."""

    def __init__(self, config: TTSConfig):
        self.config = config

    def generate_speech(
        self, text: str, output_path: Optional[str] = None, **kwargs
    ) -> str:
        """Generate speech using Coqui TTS."""
        import subprocess
        import tempfile
        from urllib import parse

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        safe_text = parse.quote_plus(text)
        speaker = kwargs.get("speaker", "male-pt-3%0A")
        url = f"http://{self.config.host}:{self.config.port}/api/tts?text={safe_text}&speaker_id={speaker}&style_wav=&language_id=fr-fr"

        result = subprocess.run(
            ["curl", "-L", "-X", "GET", url, "--output", output_path],
            capture_output=True,
        )

        if result.returncode != 0:
            raise Exception(f"TTS call failed: {result.stderr.decode()}")

        return output_path

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file."""
        return get_length(audio_path)


# ============================================================================
# STT Implementations
# ============================================================================


class WhisperSTT(STTInterface):
    """Stable-Whisper STT implementation."""

    def __init__(self, config: STTConfig):
        self.config = config
        self.model = stable_whisper.load_faster_whisper(config.model)

    def transcribe(self, audio_path: str, **kwargs) -> Any:
        """Transcribe audio to text."""
        return self.model.transcribe(audio_path)

    def transcribe_to_srt(self, audio_path: str, srt_output_path: str, **kwargs) -> str:
        """Transcribe audio and save as SRT file."""
        result = self.transcribe(audio_path, **kwargs)
        result.to_srt_vtt(srt_output_path)
        return srt_output_path


class FasterWhisperSTT(STTInterface):
    """Faster-Whisper STT implementation."""

    def __init__(self, config: STTConfig):
        self.config = config
        from faster_whisper import WhisperModel

        self.model = WhisperModel(config.model, device="cuda", compute_type="float16")

    def transcribe(self, audio_path: str, **kwargs) -> Any:
        """Transcribe audio to text."""
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        return list(segments)

    def transcribe_to_srt(self, audio_path: str, srt_output_path: str, **kwargs) -> str:
        """Transcribe audio and save as SRT file."""
        segments = self.transcribe(audio_path, **kwargs)

        with open(srt_output_path, "w") as f:
            for i, segment in enumerate(segments, start=1):
                f.write(f"{i}\n")
                f.write(
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
                )
                f.write(f"{segment.text.strip()}\n\n")

        return srt_output_path


def format_timestamp(seconds: float) -> str:
    """Format timestamp for SRT file."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ============================================================================
# Video Retriever Implementations
# ============================================================================


class VideoSearchRetriever(VideoRetrieverInterface):
    """Video retriever using remote VideoSearchClient with VideoPrism embeddings.
    
    Connects to a remote video embedding server that provides similarity search
    using VideoPrism embeddings stored in Qdrant.
    """

    def __init__(self, config: VideoRetrievalConfig):
        self.config = config
        self.client = VideoSearchClient(server_url=config.server_url)
        self.collection = config.collection

    def search(
        self, query: str, top_k: int = 10, tags: Optional[List[str]] = None
    ) -> List[VideoSearchResult]:
        """Search for videos using VideoPrism embeddings via remote server.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            tags: Optional list of tags to filter by (e.g., ['person:fred'])
            
        Returns:
            List of VideoSearchResult objects with path, similarity, tags, etc.
        """
        # Use character_filter as default tag if no tags provided
        search_tags = tags
        if search_tags is None and self.config.character_filter:
            search_tags = [self.config.character_filter]

        search_tags = search_tags
        collection = "friends_v1"
        
        return self.client.search(
            query=query,
            collection=collection,
            top_k=top_k,
            tags=search_tags,
        )

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get metadata for a video.
        
        Note: With the remote VideoSearchClient, metadata is included in search results.
        This method returns basic path info for compatibility.
        """
        return {"path": video_path}


# ============================================================================
# Prompt Provider Implementations
# ============================================================================


class LocalPromptProvider(PromptProviderInterface):
    """Local file-based prompt provider."""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.prompts = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from local file or directory."""
        if self.config.local_file:
            if os.path.isfile(self.config.local_file):
                # Load single file
                with open(self.config.local_file, "r") as f:
                    if self.config.local_file.endswith(".json"):
                        self.prompts = json.load(f)
                    else:
                        # Single prompt file
                        name = os.path.splitext(
                            os.path.basename(self.config.local_file)
                        )[0]
                        self.prompts[name] = f.read()
            elif os.path.isdir(self.config.local_file):
                # Load all prompts from directory
                for filename in os.listdir(self.config.local_file):
                    if filename.endswith((".txt", ".md", ".json")):
                        filepath = os.path.join(self.config.local_file, filename)
                        name = os.path.splitext(filename)[0]
                        with open(filepath, "r") as f:
                            if filename.endswith(".json"):
                                data = json.load(f)
                                self.prompts.update(data)
                            else:
                                self.prompts[name] = f.read()
        else:
            # Use default prompts from creation_interface.py
            from apps.creation_interface import DEFAULT_PROMPT

            self.prompts["story_generation"] = DEFAULT_PROMPT

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get a prompt template by name and format it."""
        template = self.prompts.get(prompt_name, "")
        if kwargs:
            return template.format(**kwargs)
        return template

    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        return list(self.prompts.keys())

    def get_raw_prompt(self, prompt_name: str) -> str:
        """Get the raw (unformatted) prompt template."""
        return self.prompts.get(prompt_name, "")


class MLflowPromptProvider(PromptProviderInterface):
    """MLflow-based prompt provider."""

    def __init__(self, config: PromptConfig):
        self.config = config
        try:
            import mlflow

            self.mlflow = mlflow
            if config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            if config.mlflow_experiment:
                mlflow.set_experiment(config.mlflow_experiment)
        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get a prompt from MLflow and format it."""
        template = self._load_from_mlflow(prompt_name)
        if kwargs:
            return template.format(**kwargs)
        return template

    def list_prompts(self) -> List[str]:
        """List all available prompts in MLflow."""
        # This would need to be implemented based on MLflow's prompt registry
        # For now, return empty list
        return []

    def get_raw_prompt(self, prompt_name: str) -> str:
        """Get the raw prompt template from MLflow."""
        return self._load_from_mlflow(prompt_name)

    def _load_from_mlflow(self, prompt_name: str) -> str:
        """Load prompt from MLflow."""
        # Implement MLflow prompt loading
        # This is a placeholder - actual implementation would use MLflow's API
        client = self.mlflow.tracking.MlflowClient()
        # Load prompt from MLflow (implementation depends on how prompts are stored)
        # For example, as artifacts or parameters
        raise NotImplementedError("MLflow prompt loading not yet implemented")


# ============================================================================
# Factory Functions
# ============================================================================


def create_llm(config: LLMConfig) -> LLMInterface:
    """Create LLM instance based on configuration."""
    if config.provider == "anthropic":
        return AnthropicLLM(config)
    elif config.provider == "openai":
        return OpenAILLM(config)
    elif config.provider == "litellm":
        return LiteLLM(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


async def create_tts(config: TTSConfig, character_name: Optional[str] = None) -> TTSInterface:
    """
    Create TTS instance based on configuration.

    If character_name is provided and config doesn't have reference_audio,
    loads character data from entity service and populates reference_audio
    and reference_text from the first voice sample.

    Args:
        config: TTS configuration
        character_name: Optional character name to load voice samples from

    Returns:
        TTSInterface implementation with voice cloning configured
    """
    # Load character voice samples if character_name is provided
    if character_name and not config.reference_audio:
        try:
            print(f"🎤 Loading voice samples for character: {character_name}")
            async with CharacterClient() as client:
                character: Character = await client.get_character(character_name)

            if character.voice_samples and len(character.voice_samples) > 0:
                # Use the first voice sample for voice cloning
                first_sample = character.voice_samples[0]

                # Resolve the path using path resolver
                path_resolver = get_path_resolver()
                resolved_audio_path = path_resolver.resolve_audio(
                    first_sample.sample_storage_path
                )

                config.reference_audio = resolved_audio_path
                config.reference_text = first_sample.transcript
                print(f"✓ Voice cloning configured for '{character.name}':")
                print(f"  Reference audio: {config.reference_audio}")
                print(f"  Reference text: {config.reference_text[:50]}...")

                if not path_resolver.exists(first_sample.sample_storage_path):
                    print(
                        f"⚠ Warning: Reference audio file not found at {resolved_audio_path}"
                    )
            else:
                print(f"⚠ Warning: Character '{character.name}' has no voice samples.")
                print(f"  TTS will use default voice (no voice cloning).")
        except Exception as e:
            print(f"⚠ Warning: Could not load character '{character_name}': {e}")
            print(
                f"  TTS will proceed with default voice or config-specified reference audio."
            )

    if config.provider == "fish":
        return FishSpeechTTS(config)
    elif config.provider == "solero":
        return SoleroTTS(config)
    elif config.provider == "coqui":
        return CoquiTTS(config)
    else:
        raise ValueError(f"Unknown TTS provider: {config.provider}")


def create_stt(config: STTConfig) -> STTInterface:
    """Create STT instance based on configuration."""
    if config.provider == "whisper":
        return WhisperSTT(config)
    elif config.provider == "faster-whisper":
        return FasterWhisperSTT(config)
    else:
        raise ValueError(f"Unknown STT provider: {config.provider}")


def create_video_retriever(config: VideoRetrievalConfig) -> VideoRetrieverInterface:
    """Create video retriever instance using remote VideoSearchClient."""
    return VideoSearchRetriever(config)


def create_prompt_provider(config: PromptConfig) -> PromptProviderInterface:
    """Create prompt provider instance based on configuration."""
    if config.provider == "local":
        return LocalPromptProvider(config)
    elif config.provider == "mlflow":
        return MLflowPromptProvider(config)
    else:
        raise ValueError(f"Unknown prompt provider: {config.provider}")
