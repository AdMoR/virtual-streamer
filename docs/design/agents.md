# ADK Agents

This document describes the Google ADK agent architecture for Virtual Streamer.

## Agent Structure

All agents follow the Google ADK conventions:

```
agents/
├── agent_name/
│   ├── __init__.py
│   ├── agent.py      # Agent definition (REQUIRED - all agent definitions go here)
│   ├── prompt.py     # Prompt templates
│   ├── schema.py     # Pydantic models for input/output (optional)
│   └── callback.py   # Event callbacks (optional)
```

### Rules for Agent Definition

1. **All agent definitions MUST be in `agent.py`** - This is the single source of truth for what the agent does
2. **One agent folder = One logical agent** - Each agent folder represents a distinct agent capability
3. **Factory functions go in `agent.py`** - If your agent needs a factory (e.g., `create_my_agent()`), define it in `agent.py`

### Map-Reduce Agents

When building a MapReduceAgent, create a **separate agent folder** that contains:

```
agents/
├── my_worker_agent/          # The base worker agent (e.g., rubric_builder_agent)
│   ├── agent.py              # BaseLlmAgent or LlmAgent definition
│   ├── prompt.py             # Worker prompt
│   └── schema.py             # Shared schemas (input/output models)
│
├── my_worker_map_reduce/     # The map-reduce orchestrator (separate folder!)
│   ├── __init__.py
│   ├── agent.py              # Contains: Mapper, Aggregator, StatefulWorker, factory function
│   └── callback.py           # Stateful callbacks for the worker
```

**Key points for MapReduceAgent:**
- The MapReduceAgent is a **new agent** and deserves its own folder
- Keep Mapper, Aggregator, and StatefulWorker factory all in `agent.py`
- Import shared schemas/prompts from the base worker agent
- The factory function (e.g., `create_rubric_builder_map_reduce()`) is the main entry point

**Example structure for `agent.py` in a map-reduce agent:**
```python
# agents/my_worker_map_reduce/agent.py

# 1. Stateful worker factory
def get_stateful_worker(run_id: str) -> StatefulLlmAgent:
    ...

# 2. Mapper class
class MyMapper(MapperAgent):
    def build_items_from_state(self, ctx): ...

# 3. Aggregator class
class MyAggregator(AggregatorAgent[OutputSchema]):
    async def aggregation_fn(self, results): ...

# 4. Main factory function
def create_my_map_reduce_agent(...) -> MapReduceAgent:
    ...
```

## Agent Factory

Agents can be loaded dynamically using the factory registry pattern:

```python
# agents/factory.py

from virtual_streamer.agents.factory import get_agent, list_agents, register_agent

# List available agents
available = list_agents()  # ["greeting_jesus_agent", "answering_jesus_agent"]

# Load an agent by name
agent = get_agent("greeting_jesus_agent")

# Register a new agent
@register_agent("my_custom_agent")
def _get_my_agent():
    from virtual_streamer.agents.my_agent.agent import get_my_agent
    return get_my_agent()
```

The factory uses lazy imports to avoid loading all agents at startup.

---

## Agent Hierarchy

```
                    ┌─────────────────────┐
                    │   video_creator     │  ← Orchestrator
                    │   (orchestrates)    │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   tts_agent   │    │  lip_sync_agent │    │ video_composer  │
│  (sub-agent)  │    │   (sub-agent)   │    │   (sub-agent)   │
└───────────────┘    └─────────────────┘    └─────────────────┘
        │                    │                      │
        ▼                    ▼                      ▼
   TTSClient            Wav2LipClient         FFmpeg/Video
                       FaceDetectionClient      processing
```

---

## Jesus Character Agents

These agents generate personalized responses for the "AI Jesus" character.

### Greeting Jesus Agent

Generates personalized greetings for new viewers, analyzing their username with "divine insight".

```python
# agents/greeting_jesus_agent/agent.py

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.greeting_jesus_agent.prompt import PROMPT

class GreetingJesusAgent(BaseLlmAgent):
    def __init__(self):
        super().__init__(
            name="greeting_jesus_agent",
            instruction=PROMPT,
            output_schema=None,  # Free-form text output
        )

def get_greeting_jesus_agent() -> BaseLlmAgent:
    return GreetingJesusAgent()

root_agent = get_greeting_jesus_agent()
```

**Prompt characteristics:**
- Speaks in French
- Witty, sarcastic, slightly condescending
- Analyzes username as a "prophetic sign"
- Includes biblical references (real or invented)
- Max 4 lines of output

### Answering Jesus Agent

Answers viewer questions with dark humor and biblical references.

```python
# agents/answering_jesus_agent/agent.py

from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.answering_jesus_agent.prompt import PROMPT

class AnsweringJesusAgent(BaseLlmAgent):
    def __init__(self):
        super().__init__(
            name="answering_jesus_agent",  
            instruction=PROMPT,
            output_schema=None,  # Free-form text output
        )

def get_answering_jesus_agent() -> BaseLlmAgent:
    return AnsweringJesusAgent()

root_agent = get_answering_jesus_agent()
```

**Prompt characteristics:**
- Stand-up comedian pretending to be Jesus
- Mean, sarcastic, dark humor
- Uses French street slang
- Calls audience "mon petit pécheur"
- Integrates biblical references
- ~4 sentences output

### API Integration

Both agents are exposed via the Jesus Agents API:

```bash
# Greeting
POST /api/v1/jesus-agents/greeting/submit
{"user_name": "DrPetitesFesses"}

# Answering
POST /api/v1/jesus-agents/answering/submit
{"question": "Comment fait-on les hosties?", "user_name": "Dany"}
```

The API wraps the agent with a full video pipeline:
`Agent → TTS → Wav2Lip → Subtitles → MinIO`

---

## Sub-Agents

### TTS Agent

Generates speech audio from text.

```python
# agents/sub_agents/tts_agent/agent.py

from google.adk import LlmAgent
from google.adk.tools import tool

from vs_core.api_clients.tts import create_tts_client
from vs_core.models import Character


@tool
def generate_speech(
    text: str,
    character_id: str,
    output_path: str
) -> dict:
    """
    Generate speech audio for the given text using character's voice.
    
    Args:
        text: The text to synthesize
        character_id: ID of the character whose voice to use
        output_path: Where to save the audio file
        
    Returns:
        Dictionary with audio_path and duration
    """
    with create_tts_client() as client:
        result_path = client.synthesize(
            text=text,
            character_id=character_id,
            output_path=output_path
        )
        return {
            "audio_path": result_path,
            "status": "completed"
        }


tts_agent = LlmAgent(
    name="tts_agent",
    description="Generates speech audio from text using a character's voice",
    model="gemini-2.0-flash",
    tools=[generate_speech],
)
```

```python
# agents/sub_agents/tts_agent/prompt.py

SYSTEM_PROMPT = """
You are a TTS (Text-to-Speech) agent responsible for generating audio from text.

When asked to generate speech:
1. Use the generate_speech tool with the provided text and character_id
2. Return the path to the generated audio file
3. Report any errors clearly

Available characters and their voices are managed by the entity service.
"""
```

### Lip Sync Agent

Generates lip-synced video from audio and reference video.

```python
# agents/sub_agents/lip_sync_agent/agent.py

from google.adk import LlmAgent
from google.adk.tools import tool

from vs_core.api_clients.wav2lip import create_wav2lip_client
from vs_core.api_clients.face_detection import create_face_detection_client
from vs_core.api_models.wav2lip import Wav2LipRequest
from vs_core.api_models.face_detection import PreprocessedFacesRequest


@tool
def preprocess_character_video(
    video_path: str,
    character_id: str
) -> dict:
    """
    Preprocess a character's video for lip-sync (caches face detection).
    
    Should be called once when setting up a new character.
    
    Args:
        video_path: Path to character's reference video
        character_id: Unique character identifier
        
    Returns:
        Dictionary with cache_path and frame_count
    """
    with create_face_detection_client() as client:
        response = client.preprocess_for_wav2lip(
            PreprocessedFacesRequest(
                video_path=video_path,
                character_id=character_id
            )
        )
        return response.model_dump()


@tool
def generate_lip_sync_video(
    video_path: str,
    audio_path: str,
    output_path: str
) -> dict:
    """
    Generate a lip-synced video from source video and audio.
    
    Args:
        video_path: Path to the source video (character's reference)
        audio_path: Path to the audio file to sync
        output_path: Where to save the output video
        
    Returns:
        Dictionary with status and output_path
    """
    with create_wav2lip_client() as client:
        response = client.generate(
            Wav2LipRequest(
                video_path=video_path,
                audio_path=audio_path,
                output_path=output_path
            )
        )
        return response.model_dump()


lip_sync_agent = LlmAgent(
    name="lip_sync_agent",
    description="Generates lip-synced videos by synchronizing audio with video",
    model="gemini-2.0-flash",
    tools=[
        preprocess_character_video,
        generate_lip_sync_video
    ],
)
```

### Video Composer Agent

Combines video clips and adds post-processing.

```python
# agents/sub_agents/video_composer_agent/agent.py

from google.adk import LlmAgent
from google.adk.tools import tool
import subprocess
import os


@tool
def combine_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str
) -> dict:
    """
    Combine video and audio tracks using ffmpeg.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Where to save combined video
        
    Returns:
        Dictionary with output_path
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return {"output_path": output_path, "status": "completed"}


@tool
def add_subtitles(
    video_path: str,
    subtitle_text: str,
    output_path: str,
    style: dict = None
) -> dict:
    """
    Add subtitles to a video.
    
    Args:
        video_path: Path to input video
        subtitle_text: Text to display as subtitle
        output_path: Where to save output video
        style: Optional style configuration
        
    Returns:
        Dictionary with output_path
    """
    # Create temporary SRT file
    srt_path = output_path.replace(".mp4", ".srt")
    with open(srt_path, "w") as f:
        f.write(f"1\n00:00:00,000 --> 00:10:00,000\n{subtitle_text}\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    os.remove(srt_path)
    return {"output_path": output_path, "status": "completed"}


@tool
def concatenate_videos(
    video_paths: list,
    output_path: str
) -> dict:
    """
    Concatenate multiple videos into one.
    
    Args:
        video_paths: List of video file paths
        output_path: Where to save combined video
        
    Returns:
        Dictionary with output_path
    """
    # Create concat file
    concat_file = "/tmp/concat_list.txt"
    with open(concat_file, "w") as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    os.remove(concat_file)
    return {"output_path": output_path, "status": "completed"}


video_composer_agent = LlmAgent(
    name="video_composer_agent",
    description="Composes and post-processes videos (combine, subtitle, concatenate)",
    model="gemini-2.0-flash",
    tools=[
        combine_video_audio,
        add_subtitles,
        concatenate_videos
    ],
)
```

---

## Top-Level Agents

### QA Responder Agent

Used by AI Jesus - answers questions and generates video responses.

```python
# agents/qa_responder/agent.py

from google.adk import LlmAgent
from google.adk.tools import tool

from agents.sub_agents.tts_agent.agent import tts_agent
from agents.sub_agents.lip_sync_agent.agent import lip_sync_agent
from agents.sub_agents.video_composer_agent.agent import video_composer_agent


qa_responder = LlmAgent(
    name="qa_responder",
    description="""
    Answers questions in character and generates video responses.
    
    Workflow:
    1. Generate answer text based on character personality
    2. Use TTS to generate audio
    3. Use lip-sync to generate video
    4. Compose final video with optional subtitles
    """,
    model="gemini-2.0-flash",
    sub_agents=[
        tts_agent,
        lip_sync_agent,
        video_composer_agent
    ],
)
```

```python
# agents/qa_responder/prompt.py

SYSTEM_PROMPT = """
You are {character_name}, responding to questions from your audience.

Character description: {character_description}

Personality traits:
{personality_traits}

When responding:
1. Stay in character at all times
2. Be {tone} in your responses
3. Keep responses concise (under 60 seconds when spoken)

After generating your response text, coordinate with sub-agents to:
1. Generate audio using tts_agent
2. Generate lip-synced video using lip_sync_agent  
3. Add subtitles if requested using video_composer_agent
"""

def get_prompt(character: dict, options: dict = None) -> str:
    return SYSTEM_PROMPT.format(
        character_name=character.get("name", "Unknown"),
        character_description=character.get("description", ""),
        personality_traits=character.get("personality", ""),
        tone=options.get("tone", "engaging")
    )
```

### Story Generator Agent

Used by Fred & Jamy - generates parody stories.

```python
# agents/story_generator/agent.py

from google.adk import LlmAgent
from google.adk.tools import tool


@tool
def structure_dialogue(
    raw_story: str,
    characters: list
) -> dict:
    """
    Parse raw story text into structured dialogue entries.
    
    Args:
        raw_story: The generated story text
        characters: List of character IDs in the story
        
    Returns:
        Dictionary with structured dialogue list
    """
    # Parse dialogue format: "FRED: line\nJAMY: line\n..."
    lines = []
    for line in raw_story.strip().split("\n"):
        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker = speaker.strip().lower()
            if speaker in [c.lower() for c in characters]:
                lines.append({
                    "character_id": speaker,
                    "text": text.strip()
                })
    
    return {"dialogue": lines, "count": len(lines)}


story_generator = LlmAgent(
    name="story_generator",
    description="""
    Generates parody stories featuring specified characters.
    
    Specializes in C'est pas Sorcier style humor with:
    - Fred's bombastic energy
    - Jamy's skeptical reactions
    - Nostalgic references
    - Absurd escalation
    """,
    model="gemini-2.0-flash",
    tools=[structure_dialogue],
)
```

```python
# agents/story_generator/prompt.py

STORY_GENERATION_PROMPT = """
Tu es un scénariste spécialisé dans les parodies de "C'est pas Sorcier".

Ton objectif: créer des dialogues humoristiques entre Fred et Jamy.

Style à respecter:
- Fred est enthousiaste, avec des idées grandioses et absurdes
- Jamy est le straight-man sceptique
- Utilise des références à la France des années 90-2000
- L'humour doit être affectueux, jamais méchant
- Escalade vers l'absurde

Format de sortie:
FRED: [dialogue]
JAMY: [dialogue]
...

Sujet: {title}

Génère une histoire de 5-10 échanges.
"""
```

### Video Creator Agent

Orchestrates the full video creation pipeline.

```python
# agents/video_creator/agent.py

from google.adk import LlmAgent

from agents.sub_agents.tts_agent.agent import tts_agent
from agents.sub_agents.lip_sync_agent.agent import lip_sync_agent
from agents.sub_agents.video_composer_agent.agent import video_composer_agent


video_creator = LlmAgent(
    name="video_creator",
    description="""
    Orchestrates complete video creation from dialogue.
    
    Input: Structured dialogue with character IDs and text
    Output: Final composed video
    
    Pipeline:
    1. For each dialogue line:
       a. Generate audio with tts_agent
       b. Generate lip-synced clip with lip_sync_agent
    2. Concatenate all clips with video_composer_agent
    3. Add optional post-processing (subtitles, effects)
    """,
    model="gemini-2.0-flash",
    sub_agents=[
        tts_agent,
        lip_sync_agent,
        video_composer_agent
    ],
)
```

---

## Application Configuration

Applications don't define agents - they configure which agents to use:

```yaml
# apps/ai_jesus/config.yaml
name: "AI Jesus"
description: "Q&A streamer responding to Twitch chat"

# Character to use
character: "jesus"

# Agent configuration
agents:
  root: "qa_responder"
  
# Agent-specific settings
agent_settings:
  qa_responder:
    personality:
      tone: "sarcastic"
      style: "religious_parody"
    options:
      add_subtitles: true
      subtitle_mode: "question"

# Service endpoints
services:
  tts:
    host: "${TTS_HOST:-localhost}"
    port: "${TTS_PORT:-8003}"
  wav2lip:
    host: "${WAV2LIP_HOST:-localhost}"
    port: "${WAV2LIP_PORT:-8001}"
```

```yaml
# apps/fred_et_jamy/config.yaml
name: "Fred et Jamy"
description: "C'est pas Sorcier parody video generator"

# Characters
characters:
  - "fred"
  - "jamy"

# Agent configuration  
agents:
  root: "story_generator"
  pipeline:
    - "story_generator"
    - "video_creator"

# Agent-specific settings
agent_settings:
  story_generator:
    style: "cest_pas_sorcier_parody"
    language: "fr"
  video_creator:
    add_subtitles: true
```

---

## Running Agents

```bash
# Start ML services (required)
cd infra && docker compose up -d

# Wait for services to be healthy
docker compose ps

# Run ADK server
adk web --agents-dir ./agents

# Or run specific agent
adk run qa_responder --input "What is the meaning of life?"
```

