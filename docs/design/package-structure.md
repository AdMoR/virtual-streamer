# Package Structure

## Current Directory Tree

```
virtual-streamer/
│
├── pyproject.toml                     # UV workspace root
├── uv.lock
├── README.md
│
├── docs/                              # 📐 Design documentation
│   └── design/
│       ├── README.md
│       ├── architecture.md
│       ├── package-structure.md       # (this file)
│       ├── data-models.md
│       ├── api-services.md
│       ├── agents.md
│       ├── streaming.md
│       └── migration-plan.md
│
├── configs/                           # ⚙️ Agent and app configurations
│   ├── agents/                        # Agent-specific configs
│   │   ├── example_agent.yaml
│   │   ├── greeting_jesus_agent.yaml
│   │   ├── keyword_generator.yaml
│   │   ├── rubric_builder.yaml
│   │   ├── story_generator.yaml
│   │   ├── video_matcher.yaml
│   │   └── virtual_streamer.yaml
│   ├── default_config.yaml
│   └── virtual_streamer_tools.yaml
│
├── prompts/                           # 📝 Prompt templates
│   ├── cest_pas_sorcier.txt
│   ├── gorafi_rules.txt
│   ├── story_generation.txt
│   └── thread_horreur.txt
│
├── virtual_streamer/                  # 📦 Main Python package
│   │
│   ├── agents/                        # 🤖 ADK Agents
│   │   ├── factory.py                 # Agent registry/factory
│   │   ├── common/                    # Shared agent utilities
│   │   │   ├── callbacks.py
│   │   │   ├── state_keys.py
│   │   │   └── utils.py
│   │   │
│   │   ├── greeting_jesus_agent/      # ✅ Greets Twitch viewers
│   │   │   ├── agent.py
│   │   │   └── prompt.py
│   │   │
│   │   ├── answering_jesus_agent/     # ✅ Answers viewer questions
│   │   │   ├── agent.py
│   │   │   └── prompt.py
│   │   │
│   │   ├── story_generator/           # ✅ Generates parody stories
│   │   │   ├── agent.py
│   │   │   ├── prompt.py
│   │   │   ├── schema.py
│   │   │   └── callback.py
│   │   │
│   │   ├── keyword_generator/         # ✅ Extracts keywords from text
│   │   │   ├── agent.py
│   │   │   └── prompt.py
│   │   │
│   │   ├── video_matcher/             # ✅ Matches videos to dialogue
│   │   │   ├── agent.py
│   │   │   ├── aggregator.py
│   │   │   ├── prompt.py
│   │   │   ├── schema.py
│   │   │   └── callback.py
│   │   │
│   │   ├── sentence_video_matcher/    # ✅ Per-sentence video matching
│   │   │   ├── agent.py
│   │   │   ├── schema.py
│   │   │   └── utils.py
│   │   │
│   │   ├── rubric_builder_agent/      # ✅ Builds video rubrics
│   │   │   ├── agent.py
│   │   │   ├── prompt.py
│   │   │   └── schema.py
│   │   │
│   │   ├── rubric_builder_map_reduce/ # ✅ Map-reduce rubric builder
│   │   │   ├── agent.py
│   │   │   └── callback.py
│   │   │
│   │   ├── orchestrator/              # ✅ Pipeline orchestrator
│   │   │   └── agent.py
│   │   │
│   │   └── virtual_streamer_agent/    # ✅ Main streaming agent
│   │       ├── agent.py
│   │       ├── prompt.py
│   │       ├── schema.py
│   │       ├── callbacks/             # Context injection callbacks
│   │       │   └── context_injector.py
│   │       ├── context/               # Context providers
│   │       │   ├── builder.py
│   │       │   ├── chat_provider.py
│   │       │   ├── conversation.py
│   │       │   ├── providers.py
│   │       │   └── queue_provider.py
│   │       └── tools/                 # Agent tools
│   │           ├── base.py
│   │           ├── create_video.py
│   │           ├── send_message.py
│   │           └── factory.py
│   │
│   ├── api/                           # 🌐 FastAPI REST API
│   │   ├── main.py                    # Main app entry point
│   │   ├── adk_app.py                 # ADK agents mounting
│   │   ├── dependencies.py            # Dependency injection
│   │   │
│   │   ├── low_level/                 # Entity CRUD endpoints
│   │   │   ├── characters.py          # /api/v1/characters
│   │   │   ├── clips.py               # /api/v1/clips
│   │   │   ├── story_templates.py     # /api/v1/story-templates
│   │   │   ├── articles.py            # /api/v1/articles
│   │   │   ├── streams.py             # /api/v1/streams
│   │   │   ├── programmations.py      # /api/v1/programmations
│   │   │   └── playlist.py            # /api/v1/playlist
│   │   │
│   │   ├── medium_level/              # ML service wrappers
│   │   │   ├── tts.py                 # /api/v1/tts
│   │   │   ├── stt.py                 # /api/v1/stt
│   │   │   ├── wav2lip.py             # /api/v1/wav2lip
│   │   │   └── face_detection.py      # /api/v1/face-detection
│   │   │
│   │   ├── high_level/                # Application workflows
│   │   │   ├── video_generation.py    # /api/v1/video-generation
│   │   │   ├── jesus_agents.py        # /api/v1/jesus-agents
│   │   │   └── legacy_qa.py           # /process (deprecated)
│   │   │
│   │   ├── clients/                   # Internal API clients
│   │   │   └── character_client.py
│   │   │
│   │   └── utils/                     # API utilities
│   │       ├── gpu_semaphore.py
│   │       └── mount_app.py
│   │
│   ├── lib/                           # 🔧 Core library classes
│   │   ├── agents/                    # Agent base classes
│   │   │   ├── base.py                # BaseLlmAgent
│   │   │   ├── stateful_agent.py      # StatefulLlmAgent
│   │   │   ├── dynamic_parallel_processor.py  # MapReduceAgent
│   │   │   ├── callbacks.py           # Callback base classes
│   │   │   └── stateful_callbacks.py  # State callbacks
│   │   │
│   │   ├── config/                    # Configuration loading
│   │   │   └── loader.py
│   │   │
│   │   └── providers/                 # LLM providers
│   │       └── llm.py
│   │
│   ├── streaming/                     # 📺 OBS Streaming Infrastructure
│   │   ├── models.py                  # StreamConfig, PlaylistEntry, etc.
│   │   ├── store.py                   # Database operations
│   │   │
│   │   ├── video_server/              # HTML5 video player
│   │   │   ├── app.py                 # FastAPI proxy
│   │   │   └── static/
│   │   │       └── index.html         # Browser source player
│   │   │
│   │   ├── twitch/                    # Twitch integration
│   │   │   └── chat_reader.py         # IRC chat reader
│   │   │
│   │   └── agent_loop/                # Continuous agent runner
│   │       ├── main.py
│   │       ├── runner.py
│   │       └── chat_store.py
│   │
│   ├── utils/                         # 🛠️ Shared utilities
│   │   ├── character_loader.py        # Character loading helper
│   │   ├── minio_client.py            # MinIO storage client
│   │   ├── storage_interface.py       # Storage abstraction
│   │   ├── entity_repository.py       # Entity persistence
│   │   ├── file_manager.py            # File operations
│   │   ├── job_store.py               # Background job tracking
│   │   ├── subtitle_utils.py          # SRT generation
│   │   ├── syllable_estimator.py      # Audio timing
│   │   └── utils.py                   # General utilities
│   │
│   ├── video_generation/              # 🎬 Video generation pipeline
│   │   ├── core.py                    # Core video logic
│   │   ├── interfaces.py              # Abstract interfaces
│   │   ├── implementations.py         # Concrete implementations
│   │   ├── config.py                  # Generation config
│   │   ├── story_to_video.py          # Story-to-video conversion
│   │   └── visualizer.py              # Debug visualization
│   │
│   ├── video_search/                  # 🔍 Video search client
│   │   └── client.py                  # VideoSearchClient
│   │
│   ├── video_server/                  # 🗄️ Entity models
│   │   └── models.py                  # Character, VoiceSample, etc.
│   │
│   ├── wav2lip/                       # 👄 Wav2Lip inference
│   │   ├── inference.py
│   │   ├── preprocessing.py
│   │   ├── audio.py
│   │   └── helpers.py
│   │
│   └── news/                          # 📰 News fetching (for Gorafi)
│       ├── fetcher.py
│       ├── models.py
│       ├── selector.py
│       └── store.py
│
├── face_detection/                    # 👤 Face detection module
│   ├── api.py
│   ├── core.py
│   ├── models.py
│   ├── utils.py
│   └── detection/
│       └── sfd/                       # SFD face detector
│
├── models/                            # 🧠 ML model definitions
│   ├── wav2lip.py
│   ├── syncnet.py
│   └── conv.py
│
├── docker/                            # 🐳 Docker configurations
│   ├── docker_base/Dockerfile
│   ├── docker_unified_api/Dockerfile
│   ├── docker_tts/Dockerfile
│   ├── docker_fr_tts/Dockerfile
│   ├── docker_stt/Dockerfile
│   ├── docker_entity_api/Dockerfile
│   ├── docker_worker/Dockerfile
│   └── streaming/                     # Streaming stack
│       ├── obs/
│       ├── twitch/
│       ├── video_server/
│       └── virtual_streamer_agent/
│
├── apps/                              # 📱 Demo/UI applications
│   ├── agent_test_interface.py        # Agent testing UI
│   ├── video_generation_ui.py         # Video generation UI
│   ├── creation_interface.py          # Content creation UI
│   └── create_video_from_dialogue_demo.py
│
├── scripts/                           # 🔨 Utility scripts
│   ├── api_helpers/                   # API setup helpers
│   │   ├── register_character.py
│   │   ├── register_story_template.py
│   │   ├── seed_streaming_data.py
│   │   ├── setup_streaming_tables.py
│   │   └── bootstrap_playlist.py
│   │
│   ├── batch_video_generation.py
│   ├── batch_scene_split.py
│   ├── generate_video_adk.py
│   └── cest_pas_sorcier_parody_scorer.py
│
├── tests/                             # 🧪 Test suite
│   ├── test_file_manager.py
│   ├── test_map_reduce_agents.py
│   ├── test_stateful_callbacks.py
│   ├── test_storage.py
│   └── test_video_generation.py
│
├── compose.yaml                       # Main services docker-compose
├── compose_streaming.yml              # Streaming stack compose
├── checkpoints/                       # Model checkpoints
└── assets/                            # Static assets
    └── obs_config/                    # OBS configuration
```

---

## Package Details

### `virtual_streamer/`

The main Python package containing all application code.

```toml
# pyproject.toml (workspace root)
[project]
name = "virtual-streamer"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.20",
    "pydantic>=2.0",
    "httpx>=0.28",
    "torch>=2.0",
    "google-adk>=0.1.0",
    # ... other dependencies
]
```

### `virtual_streamer/agents/`

ADK agents following Google ADK conventions.

Each agent folder contains:
- `agent.py` - Agent definition (REQUIRED)
- `prompt.py` - Prompt templates
- `schema.py` - Input/output Pydantic models (optional)
- `callback.py` - Event callbacks (optional)

**Agent Types:**
1. **Simple agents** - Direct LLM-based agents (greeting_jesus_agent, answering_jesus_agent)
2. **Tool agents** - Agents with external tool access (virtual_streamer_agent)
3. **Map-reduce agents** - Parallel processing agents (rubric_builder_map_reduce)

### `virtual_streamer/api/`

FastAPI application with layered architecture:

| Layer | Prefix | Purpose |
|-------|--------|---------|
| Low-level | `/api/v1/` | Entity CRUD (characters, clips, streams) |
| Medium-level | `/api/v1/` | ML service wrappers (TTS, STT, Wav2Lip) |
| High-level | `/api/v1/` | Application workflows (video generation) |
| ADK | `/adk/` | Google ADK agents |

### `virtual_streamer/lib/`

Core library providing:
- `BaseLlmAgent` - Base class for all agents
- `StatefulLlmAgent` - Agent with state management
- `MapReduceAgent` - Parallel processing agent
- Callback base classes

### `virtual_streamer/streaming/`

OBS streaming infrastructure:
- Video server (HTML5 player for browser source)
- Twitch chat reader (IRC integration)
- Playlist management (database-backed)
- Agent loop (continuous video generation)

### `virtual_streamer/utils/`

Shared utilities including:
- `character_loader.py` - Load characters from storage
- `minio_client.py` - MinIO S3-compatible client
- `job_store.py` - Background job tracking
- `subtitle_utils.py` - SRT generation

---

## Import Patterns

### From agents:
```python
from virtual_streamer.agents.factory import get_agent, list_agents
from virtual_streamer.agents.greeting_jesus_agent.agent import get_greeting_jesus_agent
```

### From lib:
```python
from virtual_streamer.lib.agents import (
    BaseLlmAgent,
    StatefulLlmAgent,
    MapReduceAgent,
    MapperAgent,
    AggregatorAgent,
)
```

### From utilities:
```python
from virtual_streamer.utils.character_loader import load_character
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.job_store import get_global_job_store
```

### From API clients:
```python
from virtual_streamer.api.clients.character_client import CharacterClient
from virtual_streamer.video_search.client import VideoSearchClient
```

---

## Key Shared Utilities

### `utils/character_loader.py`

Centralized character loading used by TTS, Wav2Lip, and high-level APIs:

```python
from virtual_streamer.utils.character_loader import load_character

character = await load_character("jesus")
# Returns Character model with voice_samples, video_clip_path, etc.
```

### `agents/factory.py`

Agent factory registry for dynamic agent loading:

```python
from virtual_streamer.agents.factory import get_agent, list_agents, register_agent

# List available agents
available = list_agents()  # ["greeting_jesus_agent", "answering_jesus_agent"]

# Load agent by name
agent = get_agent("greeting_jesus_agent")

# Register new agent
@register_agent("my_agent")
def _get_my_agent():
    from virtual_streamer.agents.my_agent.agent import get_my_agent
    return get_my_agent()
```

### `lib/agents/`

Agent base classes for building new agents:

```python
from virtual_streamer.lib.agents import BaseLlmAgent

class MyAgent(BaseLlmAgent):
    def __init__(self):
        super().__init__(
            name="my_agent",
            instruction="Your prompt here",
            output_schema=MyOutputSchema,  # Optional
        )
```
