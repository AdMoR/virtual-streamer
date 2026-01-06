# Video Generation Script - Complete Documentation

**Single execution script for generating videos from stories using AI models with async optimization.**

---

## 📖 Table of Contents

1. [Quick Start](#-quick-start)
2. [Features](#-features)
3. [Installation](#-installation)
4. [Usage](#-usage)
5. [Configuration](#%EF%B8%8F-configuration)
6. [Architecture](#-architecture)
7. [Advanced Features](#-advanced-features)
8. [Troubleshooting](#-troubleshooting)
9. [API Reference](#-api-reference)

---

## ⚡ Quick Start

### 3-Step Setup

```bash
# 1. Setup configuration (once)
cp env.example .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env

# 2. Run video generation
python scripts/generate_video.py --title "Fred se lance dans l'IA"

# 3. Find your video
ls -lh output/*.mp4
```

**That's it!** 🎉

---

## 🚀 Features

### Core Capabilities
- **Async optimization**: Parallel LLM calls for I/O-bound operations
- **Concurrency control**: Semaphore-based rate limiting (prevents API errors)
- **Structured output**: Stories return title, plan, and dialog separately
- **Reproducibility**: Complete config dumps enable exact recreation
- **Event loop safety**: Lazy initialization prevents async errors

### Configuration
- **Pydantic Settings**: Type-safe configuration with validation
- **Multiple sources**: CLI args, environment variables, .env files, or YAML
- **Priority system**: CLI > Env vars > .env > .env.public > Defaults
- **Zero boilerplate**: 44% less code than traditional ArgumentParser

### Production Ready
- **Interface-based design**: Easy to swap LLM/TTS/STT implementations
- **Comprehensive logging**: Track every step of generation
- **Error handling**: Graceful degradation and clear error messages
- **Well-tested**: Unit and integration tests included

---

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install pydantic pydantic-settings aiohttp pytest pytest-asyncio

# Install video generation specific requirements
pip install -r requirements_video_generation.txt
```

### System Requirements
- Python 3.10+
- FFmpeg (for video processing)
- TTS service (Fish-Speech, Solero, or Coqui)
- LLM API key (Anthropic Claude, OpenAI, or LiteLLM)

---

## 💻 Usage

### Basic Commands

```bash
# Generate video from title
python scripts/generate_video.py --title "Fred découvre l'IA"

# Generate from story file
python scripts/generate_video.py --story-file story.txt

# Recreate from config dump (skips LLM calls!)
python scripts/generate_video.py --from-config-dump output/config_20251111.json

# Quiet mode (minimal output)
python scripts/generate_video.py --title "Fred" --quiet
```

### Configuration Options

```bash
# Use YAML config file
python scripts/generate_video.py --title "Fred" --config configs/custom.yaml

# Override settings via CLI
python scripts/generate_video.py \
  --title "Fred" \
  --max-parallel-llm-calls 10 \
  --output_dir ./my_videos

# Use custom .env file
python scripts/generate_video.py --title "Fred" --env-file production.env
```

### Advanced Usage

```bash
# Custom concurrency (rate limiting)
python scripts/generate_video.py --title "Fred" --max-parallel-llm-calls 15

# Verbose mode with timing
python scripts/generate_video.py --title "Fred" --verbose

# Custom prompt file
python scripts/generate_video.py --title "Fred" --prompt-file prompts/custom.txt
```

---

## ⚙️ Configuration

### Configuration Priority

Settings are loaded in this order (highest to lowest priority):

1. **CLI arguments** → `--title "Fred" --max-parallel-llm-calls 10`
2. **Environment variables** → `export VG_MAX_PARALLEL_LLM_CALLS=10`
3. **.env file** → `VG_MAX_PARALLEL_LLM_CALLS=10` (secrets)
4. **.env.public file** → `VG_OUTPUT_DIR=./output` (non-secrets)
5. **Default values** → Built into code

### Configuration via .env Files

**Recommended approach for secrets:**

```bash
# .env (API keys - DO NOT COMMIT!)
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here

# .env.public (non-secrets - CAN COMMIT)
VG_LLM__PROVIDER=anthropic
VG_LLM__MODEL=claude-sonnet-4-5-20250929
VG_LLM__TEMPERATURE=0.7
VG_MAX_PARALLEL_LLM_CALLS=5
VG_OUTPUT_DIR=./output
VG_TEMP_DIR=./temp
```

### Configuration via Environment Variables

```bash
# Override any setting
export VG_LLM__PROVIDER=openai
export VG_LLM__MODEL=gpt-4o
export VG_MAX_PARALLEL_LLM_CALLS=10

# Run with overrides
python scripts/generate_video.py --title "Fred"
```

### Configuration via YAML

```yaml
# configs/my_config.yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  temperature: 0.7
  max_tokens: 4096

tts:
  provider: fish
  host: 127.0.0.1
  port: 8003
  reference_audio: /path/to/reference.mp4
  reference_text: "Reference text for voice cloning"

stt:
  provider: whisper
  model: base

video_retrieval:
  method: bm25
  index_path: /path/to/clips
  character_filter: fred

max_parallel_llm_calls: 5
max_sentence_length: 35
output_dir: ./output
temp_dir: ./temp
enable_config_dump: true
```

**Use it:**
```bash
python scripts/generate_video.py --title "Fred" --config configs/my_config.yaml
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_parallel_llm_calls` | 5 | Max concurrent LLM API calls |
| `max_sentence_length` | 35 | Max words per sentence segment |
| `max_search_attempts` | 3 | Max alternative keyword attempts |
| `max_video_judgement_attempts` | 5 | Max videos to judge per sentence |
| `output_dir` | `./output` | Output directory for videos |
| `temp_dir` | `./temp` | Temporary files directory |
| `enable_config_dump` | `true` | Save config dump for reproducibility |

---

## 🏗️ Architecture

### Components

```
virtual_streamer/
└── video_generation/              # Main video generation module ⭐
    ├── __init__.py                # Public API exports
    ├── config.py                  # Pydantic configuration models
    ├── interfaces.py              # Abstract base classes
    ├── implementations.py         # Concrete implementations
    └── core.py                    # Core async logic

scripts/
└── generate_video.py              # CLI entry point (SINGLE SCRIPT)

tests/
└── test_video_generation.py       # Unit and integration tests
```

### Workflow

```
Title/Story Input
    ↓
[Story Generation] ─→ LLM (async)
    ↓
[Sentence Separation] ─→ Text processing
    ↓
[Video Search & Matching] ─→ PARALLEL LLM calls (semaphore-controlled)
    ├─ Video judgement (vision API)
    ├─ Keyword generation
    └─ Alternative searches
    ↓
[Audio Generation] ─→ SERIAL TTS (local)
    ↓
[Subtitle Generation] ─→ SERIAL STT (local)
    ↓
[Segment Combination] ─→ FFmpeg
    ↓
[Final Video + Config Dump]
```

### Interface-Based Design

All components use abstract interfaces for flexibility:

- **`LLMInterface`**: Anthropic Claude, OpenAI GPT, LiteLLM
- **`TTSInterface`**: Fish-Speech, Solero, Coqui
- **`STTInterface`**: Whisper (stable-whisper), Faster-Whisper
- **`VideoRetrieverInterface`**: BM25, Vector, Hybrid
- **`PromptProviderInterface`**: Local files, MLflow

**Easy to extend:**
```python
class MyCustomLLM(LLMInterface):
    async def complete(self, prompt: str, **kwargs) -> str:
        # Your implementation
        pass
```

---

## 🔧 Advanced Features

### 1. Concurrency Control (Rate Limiting)

**Problem:** Too many concurrent API calls → Rate limit errors (429)  
**Solution:** Semaphore-based concurrency control

```python
# Automatically limits concurrent LLM calls
llm_semaphore = asyncio.Semaphore(config.max_parallel_llm_calls)
```

**Configuration:**
```bash
# Conservative (small API plan)
--max-parallel-llm-calls 3

# Default (balanced)
--max-parallel-llm-calls 5

# Aggressive (large API plan)
--max-parallel-llm-calls 15
```

**Benefits:**
- ✅ No more rate limit errors
- ✅ Predictable resource usage
- ✅ Better cost control
- ✅ Configurable per API provider

**Monitoring:**
```
Phase 1: Finding matching videos (parallel, max 5 concurrent)...
```

### 2. Event Loop Safety

**Problem:** "Task got Future attached to a different loop" error  
**Solution:** Lazy initialization of async clients

```python
class AnthropicLLM(LLMInterface):
    def __init__(self, config: LLMConfig):
        self._async_client = None  # Not created yet
    
    @property
    def async_client(self):
        """Created only when first accessed (in correct event loop)"""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._async_client
```

**Benefits:**
- ✅ No event loop conflicts
- ✅ Transparent to calling code
- ✅ Zero performance overhead

### 3. Structured Output

LLM responses are structured using Pydantic models:

```python
class StoryOutput(BaseModel):
    title: str                # Refined title
    story_plan: str          # Creative planning
    dialog: str              # Actual dialogue lines
```

**Example output:**
```json
{
  "title": "Fred se lance dans l'IA (Version complète)",
  "story_plan": "Fred va découvrir l'IA avec humour...",
  "dialog": "Fred: Eh dis donc Jamy! Fred: J'ai découvert l'IA!"
}
```

### 4. Config Dumps (Reproducibility)

Every run generates a comprehensive config dump:

```json
{
  "version": "1.0",
  "timestamp": "2025-11-11T17:30:00",
  "input": {
    "story": "...",
    "sentences": ["...", "..."]
  },
  "config": { "llm": {...}, "tts": {...} },
  "execution": {
    "video_matches": [...],
    "audio_files": [...],
    "timing": {"total": 78.45}
  }
}
```

**Recreate exact video without LLM calls:**
```bash
python scripts/generate_video.py --from-config-dump output/config_20251111.json
```

### 5. Parallel vs Serial Processing

**Parallel (async with semaphore):**
- Story generation
- Video judgement (vision API)
- Keyword generation
- Video matching

**Serial (local processing):**
- Audio generation (TTS)
- Subtitle generation (STT)
- Video combination (FFmpeg)

**Why?** LLM calls are I/O-bound (benefit from parallelism). TTS/STT use local GPU/CPU (would compete for resources).

---

## 🐛 Troubleshooting

### API Key Issues

```bash
# Check .env file
cat .env | grep API_KEY

# Verify it's loaded
python -c "import os; print(os.getenv('ANTHROPIC_API_KEY'))"

# Test configuration
python -c "
from virtual_streamer.video_generation import VideoGenerationConfig
config = VideoGenerationConfig()
print(f'Provider: {config.llm.provider}')
print(f'Model: {config.llm.model}')
"
```

### Rate Limit Errors

```bash
# Reduce concurrent calls
python scripts/generate_video.py --title "Fred" --max-parallel-llm-calls 3

# Or set permanently in .env.public
echo "VG_MAX_PARALLEL_LLM_CALLS=3" >> .env.public
```

### Event Loop Errors

The lazy initialization should prevent these, but if you see:
- "Task got Future attached to a different loop"
- "No running event loop"

**Solution:** Update to latest version (includes lazy initialization fix).

### TTS Connection Issues

```bash
# Check TTS service is running
curl http://127.0.0.1:8003/

# Check configuration
python -c "
from virtual_streamer.video_generation import VideoGenerationConfig
config = VideoGenerationConfig()
print(f'TTS Host: {config.tts.host}:{config.tts.port}')
"

# Test TTS connection
python -c "
import requests
resp = requests.get('http://127.0.0.1:8003/')
print(resp.status_code)
"
```

### Video Not Found

```bash
# Check video index path
ls -lh /media/amor/data1/Downloads/CPS/clip_infos/

# Verify configuration
python -c "
from virtual_streamer.video_generation import VideoGenerationConfig
config = VideoGenerationConfig()
print(f'Index: {config.video_retrieval.index_path}')
"
```

### Slow Performance

```bash
# Increase concurrency (if not hitting rate limits)
python scripts/generate_video.py --title "Fred" --max-parallel-llm-calls 10

# Check timing breakdown
# Look for bottlenecks in the output:
#   Video search: 45s
#   Audio generation: 12s
#   Subtitle generation: 9s
```

---

## 📊 HTML Visualization Reports

### Overview

Every video generation can produce an **interactive HTML report** showing the complete process with detailed information about video selection, LLM judgements, and performance metrics.

### Features

**For Each Sentence:**
- 📝 Sentence text
- 🎬 Video candidates evaluated
- 🤖 LLM judgement results:
  - Rating (CONTEXTUAL, NEUTRAL, NOT_CONTEXTUAL)
  - Grade (0-10 scale)
  - Detailed reasoning from LLM
- 🔍 Alternative keywords tried
- ✅ Final selected video

**Summary Information:**
- Story title and creative plan
- Total sentences processed
- Video duration
- Number of LLM API calls
- Performance timing breakdown

**Visual Design:**
- Beautiful, responsive interface
- Color-coded ratings (green/orange/red)
- Mobile-friendly layout
- No external dependencies (pure HTML/CSS)

### Usage

#### Automatic Generation

Reports are automatically created during video generation:

```bash
python scripts/generate_video.py --title "Fred découvre l'IA"
# → Creates video + config dump + HTML report automatically
```

#### From Existing Config Dump

```bash
# Generate report from specific config dump
python scripts/create_report.py output/config_20251111_173000.json

# Use latest config dump
python scripts/create_report.py --latest

# Open in browser after generation
python scripts/create_report.py output/config_20251111.json --open

# Custom output path
python scripts/create_report.py config.json --output reports/analysis.html
```

#### Python API

```python
from virtual_streamer.video_generation import (
    create_html_report,
    create_html_report_from_dump
)

# From GenerationResult (via API response)
# The video generation API returns a GenerationResult
html_path = create_html_report(result, output_path="report.html")

# From config dump file
html_path = create_html_report_from_dump(
    "output/config_20251111.json",
    output_path="custom_report.html"
)
```

### Report Sections

#### 1. Header & Summary Cards
- Total sentences, duration, LLM calls, timing
- Color-coded cards with key metrics

#### 2. Story Section (if available)
- Refined story title
- Story plan/reasoning from LLM
- Full dialog text

#### 3. Sentence Analysis
For each sentence, shows:
- **Sentence Text**: The dialogue line
- **Selected Video**: Final video chosen (highlighted in green)
- **Rating Badge**: CONTEXTUAL (green), NEUTRAL (orange), NOT_CONTEXTUAL (red)
- **Grade**: 0-10 score from LLM
- **Reasoning**: LLM's explanation for the judgement
- **Alternative Searches**: Keywords tried if first search failed

#### 4. Performance Metrics
- Video search time
- Audio generation time
- Subtitle generation time
- Total processing time

### Example Workflow

```bash
# 1. Generate video (creates config dump)
python scripts/generate_video.py --title "Fred et la blockchain"
# Output: video_20251111_173000.mp4
#         config_20251111_173000.json
#         video_generation_report_20251111_173000.html

# 2. View report
open video_generation_report_20251111_173000.html

# 3. Or regenerate report later
python scripts/create_report.py output/config_20251111_173000.json --open
```

### Use Cases

**Debugging:**
- Understand why specific videos were selected
- Identify poor matches
- Optimize prompts and keywords

**Quality Assurance:**
- Review LLM judgements
- Verify video-text alignment
- Check alternative search effectiveness

**Performance Analysis:**
- Identify bottlenecks
- Optimize concurrency settings
- Monitor API usage

**Documentation:**
- Share generation process with team
- Archive decision rationale
- Create project reports

**Training:**
- Understand how the system works
- Learn LLM reasoning patterns
- Improve prompt engineering

### CLI Reference

```bash
python scripts/create_report.py [CONFIG_DUMP] [OPTIONS]

Arguments:
  CONFIG_DUMP          Path to config dump JSON file

Options:
  -o, --output PATH    Output HTML file path
  --latest             Use latest config dump from output dir
  --open               Open report in browser after generation
  --output-dir DIR     Directory to search for config dumps (default: ./output)
  -h, --help           Show help message
```

### Examples

```bash
# Generate from specific dump
python scripts/create_report.py output/config_20251111_173000.json

# Auto-find latest, open in browser
python scripts/create_report.py --latest --open

# Custom output location
python scripts/create_report.py config.json -o reports/fred_ia.html

# Search in custom directory
python scripts/create_report.py --latest --output-dir /custom/outputs
```

---

## 📚 API Reference

### Main Script

```bash
python scripts/generate_video.py [OPTIONS]
```

**Input Options (mutually exclusive):**
- `--title TEXT` - Generate story from title
- `--story-file PATH` - Load story from file
- `--from-config-dump PATH` - Recreate from config dump

**Configuration:**
- `--config PATH` - YAML config file
- `--env-file PATH` - Additional .env file
- `--prompt-file PATH` - Custom prompt file

**Processing:**
- `--max-parallel-llm-calls INT` - Max concurrent LLM calls (default: 5)
- `--max-sentence-length INT` - Max words per sentence (default: 35)
- `--max-search-attempts INT` - Max alternative searches (default: 3)

**Output:**
- `--output_dir PATH` - Output directory (default: ./output)
- `--temp_dir PATH` - Temp directory (default: ./temp)
- `--enable_config_dump BOOL` - Save config dump (default: true)

**Display:**
- `--verbose` - Verbose output with timing
- `--quiet` - Minimal output

### Environment Variables

All configuration can be set via environment variables with `VG_` prefix:

```bash
# Format: VG_<SECTION>__<PARAMETER>
VG_LLM__PROVIDER=anthropic
VG_LLM__MODEL=claude-sonnet-4-5-20250929
VG_LLM__TEMPERATURE=0.7
VG_TTS__HOST=127.0.0.1
VG_TTS__PORT=8003
VG_MAX_PARALLEL_LLM_CALLS=5
VG_OUTPUT_DIR=./output
```

**Note:** Use double underscore `__` for nested config (e.g., `llm.provider` → `VG_LLM__PROVIDER`).

### Python API

```python
import requests
import time

# Video generation is now handled through the API using ADK agents
# and StoryTemplates for configuration.

def generate_video_via_api(
    title: str,
    story_template_id: str,
    api_url: str = "http://localhost:8000"
) -> dict:
    """Submit a video generation job and wait for completion."""
    
    # Submit the job
    response = requests.post(
        f"{api_url}/api/v1/video/submit",
        json={
            "title": title,
            "story_template_id": story_template_id,
        }
    )
    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]
    
    print(f"Job submitted: {job_id}")
    
    # Poll for completion
    while True:
        status_response = requests.get(f"{api_url}/api/v1/video/status/{job_id}")
        status_data = status_response.json()
        
        if status_data["status"] == "completed":
            print(f"Video created: {status_data['result']['video_path']}")
            return status_data["result"]
        elif status_data["status"] == "failed":
            raise RuntimeError(f"Job failed: {status_data.get('error')}")
        
        print(f"Status: {status_data['status']}...")
        time.sleep(5)


# Example usage
result = generate_video_via_api(
    title="Fred découvre l'IA",
    story_template_id="cest-pas-sorcier"
)
```

---

## 📊 Examples

### Example 1: Quick Video Generation

```bash
python scripts/generate_video.py --title "Fred et la blockchain"
```

### Example 2: Custom Configuration

```yaml
# custom_config.yaml
llm:
  provider: openai
  model: gpt-4o
  temperature: 0.8

max_parallel_llm_calls: 10
output_dir: /custom/output
```

```bash
python scripts/generate_video.py --title "Fred" --config custom_config.yaml
```

### Example 3: Environment-Based Config

```bash
# Development
export VG_LLM__PROVIDER=anthropic
export VG_MAX_PARALLEL_LLM_CALLS=3
python scripts/generate_video.py --title "Fred"

# Production
export VG_LLM__PROVIDER=openai
export VG_MAX_PARALLEL_LLM_CALLS=10
python scripts/generate_video.py --title "Fred"
```

### Example 4: Batch Processing

```bash
#!/bin/bash
# generate_batch.sh

titles=(
  "Fred et l'IA"
  "Fred et la crypto"
  "Fred et le metaverse"
)

for title in "${titles[@]}"; do
  python scripts/generate_video.py --title "$title" --quiet
done
```

### Example 5: Recreate from Dump

```bash
# Original generation (expensive)
python scripts/generate_video.py --title "Fred"
# → Creates: output/config_20251111_173000.json

# Later: recreate exact video (fast, no LLM calls)
python scripts/generate_video.py --from-config-dump output/config_20251111_173000.json
```

---

## 🔄 Migration from Old Version

If you were using `generate_video_simple.py`, migrate to the single script:

**Old:**
```bash
python scripts/generate_video_simple.py --title "Fred"
```

**New (same functionality):**
```bash
python scripts/generate_video.py --title "Fred"
```

All `.env` configuration works exactly the same!

---

## 📈 Performance Tuning

### Concurrency Settings by Use Case

| Use Case | `max_parallel_llm_calls` | Why |
|----------|--------------------------|-----|
| Development | 3 | Reduce API costs |
| Small API Plan | 3-5 | Avoid rate limits |
| Default/Balanced | 5 | Good for most cases |
| Large API Plan | 10-15 | Maximum speed |
| Production | 5-10 | Balance speed/reliability |

### Monitoring Performance

```bash
python scripts/generate_video.py --title "Fred" --verbose
```

**Output includes timing breakdown:**
```
Timing:
  Video search: 45.23s
  Audio generation: 12.34s
  Subtitle generation: 8.92s
  Segment composition: 5.67s
  Final concatenation: 2.13s
  Total: 78.45s
```

**Optimization tips:**
- If `Video search` is slow → increase `max_parallel_llm_calls`
- If hitting rate limits → decrease `max_parallel_llm_calls`
- If `Audio generation` is slow → check TTS service performance
- If `Subtitle generation` is slow → use faster Whisper model

---

## 🤝 Contributing

### Running Tests

```bash
# Run all tests
pytest tests/test_video_generation.py

# Run specific test
pytest tests/test_video_generation.py::TestStoryGeneration::test_generate_story

# Run with coverage
pytest --cov=scripts tests/test_video_generation.py
```

### Code Style

Follow existing patterns:
- Use type hints
- Add docstrings to functions
- Keep functions focused and small
- Use async/await for I/O operations

### Adding New LLM Provider

```python
# 1. Implement interface
class MyLLM(LLMInterface):
    async def complete(self, prompt: str, **kwargs) -> str:
        # Implementation
        pass
    
    async def complete_structured(self, prompt: str, response_model, **kwargs):
        # Implementation
        pass
    
    async def complete_with_vision(self, prompt: str, image_base64: str, **kwargs):
        # Implementation
        pass

# 2. Add to factory
def create_llm(config: LLMConfig) -> LLMInterface:
    if config.provider == "my_provider":
        return MyLLM(config)
    # ...

# 3. Add config
class LLMConfig(BaseModel):
    provider: str = Field(default="anthropic")  # Add "my_provider" as option
```

---

## 📄 License

Same as parent project.

---

## 🎉 Summary

- **Single script**: `scripts/generate_video.py` (no duplicates!)
- **Flexible configuration**: CLI, env vars, .env files, or YAML
- **Production-ready**: Rate limiting, error handling, reproducibility
- **Well-documented**: Complete guide in this single file
- **Easy to use**: 3 steps to your first video

**Start generating videos now!** 🎬✨

```bash
python scripts/generate_video.py --title "Fred découvre l'IA"
```
