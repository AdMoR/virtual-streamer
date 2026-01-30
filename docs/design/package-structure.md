# Package Structure

## Complete Directory Tree

```
virtual-streamer/
│
├── pyproject.toml                     # UV workspace root
├── uv.lock
├── README.md
│
├── design/                            # 📐 Design documentation (this folder)
│   ├── README.md
│   ├── architecture.md
│   ├── package-structure.md
│   ├── data-models.md
│   ├── api-services.md
│   ├── agents.md
│   └── migration-plan.md
│
├── packages/                          # 📦 Shared Python packages
│   │
│   └── vs-core/                       # Core library (pip installable)
│       ├── pyproject.toml
│       ├── README.md
│       └── src/vs_core/
│           ├── __init__.py
│           │
│           ├── models/                # 🗃️ Data models
│           │   ├── __init__.py
│           │   ├── character.py       # Character entity
│           │   ├── video_clip.py      # VideoClip entity
│           │   └── collection.py      # Collection entity
│           │
│           ├── api_clients/           # 📞 HTTP API clients
│           │   ├── __init__.py
│           │   ├── base.py            # Base client class
│           │   ├── tts.py             # TTS API client
│           │   ├── stt.py             # STT API client
│           │   ├── wav2lip.py         # Wav2Lip API client
│           │   └── face_detection.py  # Face Detection API client
│           │
│           ├── api_models/            # 📋 Request/Response models
│           │   ├── __init__.py
│           │   ├── tts.py
│           │   ├── stt.py
│           │   ├── wav2lip.py
│           │   └── face_detection.py
│           │
│           ├── storage/               # 💾 Storage backends
│           │   ├── __init__.py
│           │   ├── base.py
│           │   ├── s3_client.py
│           │   └── local_fs_client.py
│           │
│           └── utils/                 # 🛠️ Common utilities
│               ├── __init__.py
│               ├── video.py           # Video processing utils
│               ├── audio.py           # Audio processing utils
│               └── subtitle.py        # Subtitle utilities
│
├── services/                          # 🔧 ML Service APIs (Docker)
│   │
│   ├── wav2lip_service/
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── main.py                # FastAPI app
│   │       ├── inference.py           # ML inference logic
│   │       └── models.py              # Pydantic models
│   │
│   ├── face_detection_service/
│   │   ├── pyproject.toml
│   │   ├── Dockerfile
│   │   └── src/
│   │       ├── main.py
│   │       ├── inference.py
│   │       └── models.py
│   │
│   └── entity_service/
│       ├── pyproject.toml
│       ├── Dockerfile
│       └── src/
│           ├── main.py
│           └── models.py
│
├── agents/                            # 🤖 ADK Agents
│   │
│   ├── factory.py                     # 🏭 Agent factory registry
│   │
│   ├── greeting_jesus_agent/          # Greeting agent (AI Jesus)
│   │   ├── agent.py
│   │   └── prompt.py
│   │
│   ├── answering_jesus_agent/         # Q&A agent (AI Jesus)
│   │   ├── agent.py
│   │   └── prompt.py
│   │
│   ├── sub_agents/                    # Reusable sub-agents
│   │   │
│   │   ├── tts_agent/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── prompt.py
│   │   │   └── callback.py
│   │   │
│   │   ├── lip_sync_agent/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py
│   │   │   ├── prompt.py
│   │   │   └── callback.py
│   │   │
│   │   └── video_composer_agent/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── prompt.py
│   │       └── callback.py
│   │
│   ├── qa_responder/                  # Q&A agent (legacy)
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── prompt.py
│   │   └── callback.py
│   │
│   ├── story_generator/               # Story generation (Fred & Jamy)
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── prompt.py
│   │   └── callback.py
│   │
│   └── video_creator/                 # Orchestrator agent
│       ├── __init__.py
│       ├── agent.py
│       ├── prompt.py
│       └── callback.py
│
├── tools/                             # 🔨 CLI Tools
│   │
│   └── video_indexer/                 # Video dataset processing
│       ├── pyproject.toml
│       ├── README.md
│       └── src/
│           ├── __init__.py
│           ├── cli.py                 # CLI entry point
│           ├── pipeline.py            # Main processing pipeline
│           ├── scene_splitter.py      # Scene detection
│           ├── character_detector.py  # Character detection
│           ├── transcriber.py         # Audio transcription
│           └── visual_captioner.py    # Visual captioning (Florence)
│
├── apps/                              # 📱 Application Configurations
│   │
│   ├── ai_jesus/
│   │   ├── config.yaml                # Agent and character config
│   │   ├── characters/
│   │   │   └── jesus.yaml
│   │   ├── prompts/
│   │   │   └── qa_prompt.txt
│   │   └── README.md
│   │
│   ├── fred_et_jamy/
│   │   ├── config.yaml
│   │   ├── characters/
│   │   │   ├── fred.yaml
│   │   │   └── jamy.yaml
│   │   ├── prompts/
│   │   │   └── story_generation.txt
│   │   ├── scorer.py                  # Parody evaluation
│   │   └── README.md
│   │
│   └── obs_streamer/                  # OBS streaming infrastructure
│       ├── docker-compose.yml
│       ├── video_server.py
│       ├── twitch_reader.py
│       └── configs/
│           └── obs/
│
├── infra/                             # 🐳 Infrastructure
│   │
│   ├── docker/
│   │   ├── base.Dockerfile            # Base image
│   │   └── README.md
│   │
│   ├── compose.yml                    # All ML services
│   │
│   └── models/                        # Model checkpoints
│       └── README.md                  # Download instructions
│
├── config/                            # ⚙️ Global Configuration
│   │
│   └── defaults.yaml                  # Default settings
│
└── data/                              # 📁 Data directory (gitignored)
    ├── audio/
    ├── video/
    ├── cache/
    └── output/
```

---

## Package Details

### `packages/vs-core/`

The core library that all other components depend on.

```toml
# packages/vs-core/pyproject.toml
[project]
name = "vs-core"
version = "0.1.0"
description = "Virtual Streamer core library"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "httpx>=0.28",
    "boto3>=1.40",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx[http2]"]
```

**What it contains:**
- Data models (`Character`, `VideoClip`, `Collection`)
- API clients for all ML services
- Storage backends (S3, local filesystem)
- Common utilities

**What it does NOT contain:**
- ML model code (that's in services/)
- Agent logic (that's in agents/)
- Application-specific code

### `services/wav2lip_service/`

Standalone FastAPI service for Wav2Lip inference.

```toml
# services/wav2lip_service/pyproject.toml
[project]
name = "wav2lip-service"
version = "0.1.0"
dependencies = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.20",
    "torch>=2.0",
    "opencv-python>=4.6",
    "numpy<2",
]
```

**Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8001
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### `agents/`

ADK agents following Google's conventions.

Each agent folder contains:
- `agent.py` - LlmAgent definition with tools
- `prompt.py` - Prompt templates
- `callback.py` - Event callbacks (optional)

Agents can reference other agents as sub-agents:

```python
# agents/video_creator/agent.py
from agents.sub_agents.tts_agent.agent import tts_agent
from agents.sub_agents.lip_sync_agent.agent import lip_sync_agent

video_creator = LlmAgent(
    name="video_creator",
    sub_agents=[tts_agent, lip_sync_agent],
    ...
)
```

### `tools/video_indexer/`

CLI tool for processing video datasets (not an agent, not a service).

```bash
# Example usage
video-indexer process \
    --input /path/to/cest_pas_sorcier/ \
    --output /data/indexed/ \
    --characters fred.yaml jamy.yaml
```

### `apps/`

Application configurations - mostly YAML files, minimal code.

```yaml
# apps/ai_jesus/config.yaml
name: "AI Jesus"
description: "Q&A streamer responding to Twitch chat"

character: "jesus"

agents:
  root: "qa_responder"
  
services:
  tts:
    host: "localhost"
    port: 8003
  wav2lip:
    host: "localhost"
    port: 8001
```

---

## UV Workspace Configuration

```toml
# pyproject.toml (workspace root)
[project]
name = "virtual-streamer-workspace"
version = "0.1.0"
requires-python = ">=3.10"

[tool.uv.workspace]
members = [
    "packages/vs-core",
    "services/wav2lip_service",
    "services/face_detection_service",
    "services/entity_service",
    "tools/video_indexer",
]

[tool.uv.sources]
vs-core = { workspace = true }
```

---

## Import Patterns

### From an agent:
```python
from vs_core.api_clients.tts import TTSClient
from vs_core.api_clients.wav2lip import Wav2LipClient
from vs_core.api_models.wav2lip import Wav2LipRequest
from vs_core.models import Character
```

### From a service:
```python
# Services have their own models that mirror vs_core.api_models
from models import Wav2LipRequest, Wav2LipResponse
```

### From a tool:
```python
from vs_core.models import VideoClip, Collection
from vs_core.storage import LocalFSClient
```

### From the main API (virtual_streamer):
```python
# Shared utilities
from virtual_streamer.utils.character_loader import load_character
from virtual_streamer.utils.minio_client import get_storage_client
from virtual_streamer.utils.job_store import get_global_job_store

# Agent factory
from virtual_streamer.agents.factory import get_agent, list_agents

# Medium-level services
from virtual_streamer.api.medium_level.tts import generate_tts
from virtual_streamer.api.medium_level.wav2lip import generate_wav2lip
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
