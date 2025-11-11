# Video Generation Script

Standalone async video generation script for creating videos from stories using AI models.

## ⚡ Quick Start

### Two Ways to Use

**1. Simplified (Recommended)** - Minimal CLI, .env configuration:
```bash
# Setup (once)
cp env.example .env
cp env.public.example .env.public
echo "ANTHROPIC_API_KEY=your-key" >> .env

# Run (always)
python scripts/generate_video_simple.py --title "Fred se lance dans l'IA"
```

**2. Original** - Full CLI arguments:
```bash
python scripts/generate_video.py --title "Fred" --llm-provider anthropic --tts-host localhost ...
```

## Features

- **Async optimization**: Parallel LLM calls for efficiency, serial TTS/STT for local processing
- **Structured output**: Stories return title, plan, and dialog separately
- **Simple configuration**: .env files or full CLI arguments
- **Comprehensive config management**: Pydantic settings with YAML support
- **Reproducibility**: Complete config dumps enable exact recreation
- **Interface-based design**: Easy to swap implementations
- **Well-tested**: Unit and integration tests included

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install pydantic pydantic-settings aiohttp pytest pytest-asyncio

# Generate video from title
python scripts/generate_video.py --title "Fred se lance dans l'IA"

# Use custom config
python scripts/generate_video.py --title "Fred" --config configs/example_custom.yaml

# Recreate from config dump (skips expensive LLM calls)
python scripts/generate_video.py --from-config-dump output/config_20251111_103000.json
```

## Architecture

### Components

1. **`video_generation_config.py`**: Pydantic configuration models
2. **`video_generation_interfaces.py`**: Abstract interfaces for all components
3. **`video_generation_impl.py`**: Concrete implementations (LLM, TTS, STT, etc.)
4. **`video_generation_core.py`**: Core async logic
5. **`generate_video.py`**: CLI script with comprehensive help
6. **`test_video_generation.py`**: Unit and integration tests

### Workflow

```
Title/Story Input
    ↓
Story Generation (async LLM)
    ↓
Sentence Separation
    ↓
Video Search & Matching (PARALLEL LLM calls)
    ↓
Audio Generation (SERIAL, local TTS)
    ↓
Subtitle Generation (SERIAL, local STT)
    ↓
Segment Combination
    ↓
Final Video + Config Dump
```

## Configuration

### Via YAML File

```yaml
# configs/my_config.yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  temperature: 0.7

tts:
  provider: fish
  host: 127.0.0.1
  port: 8003
  reference_audio: /path/to/reference.mp4
  reference_text: "Reference text for voice cloning"

output_dir: ./output
max_parallel_llm_calls: 5
```

### Via Environment Variables

```bash
export VG_LLM__PROVIDER=openai
export VG_LLM__MODEL=gpt-4o
export VG_TTS__HOST=192.168.1.100
export VG_OUTPUT_DIR=/custom/output
```

### Via Command-Line

```bash
python scripts/generate_video.py \
  --title "Fred" \
  --llm-provider anthropic \
  --llm-model claude-sonnet-4-5-20250929 \
  --tts-host 127.0.0.1 \
  --output-dir ./output
```

## Interfaces

All components use abstract interfaces for extensibility:

### LLM Providers
- `AnthropicLLM`: Claude models
- `OpenAILLM`: GPT models
- `LiteLLM`: Multi-provider support

### TTS Providers
- `FishSpeechTTS`: Fish-Speech (default)
- `SoleroTTS`: Solero TTS
- `CoquiTTS`: Coqui TTS

### STT Providers
- `WhisperSTT`: Stable-Whisper (default)
- `FasterWhisperSTT`: Faster-Whisper

### Video Retrieval
- `BM25VideoRetriever`: BM25 text search (default)
- `VectorVideoRetriever`: Embedding-based search
- `HybridVideoRetriever`: Combines both methods

### Prompt Providers
- `LocalPromptProvider`: Local files (default)
- `MLflowPromptProvider`: MLflow prompt management

## Config Dump

Every run generates a comprehensive config dump that includes:

- All input parameters (title, story, sentences)
- All configuration settings
- All intermediate selections (video matches, audio files, subtitle files)
- All model versions and parameters
- Timing information for each phase

### Using Config Dumps

Recreate the exact same video without rerunning expensive LLM calls:

```bash
python scripts/generate_video.py --from-config-dump output/config_20251111_103000.json
```

This will:
- Skip story generation
- Skip video search and judgement (expensive LLM calls)
- Regenerate audio (local TTS)
- Regenerate subtitles (local STT)
- Recombine everything into final video

## Testing

```bash
# Run all tests
pytest tests/test_video_generation.py -v

# Run specific test
pytest tests/test_video_generation.py::TestConfiguration::test_default_config -v

# Run with coverage
pytest tests/test_video_generation.py --cov=scripts --cov-report=html
```

## Parallelization Strategy

The script is optimized for efficiency:

### Parallel Processing (LLM Calls)
- Video search and keyword generation
- Video-dialogue judgement (multiple videos evaluated simultaneously)
- All async I/O-bound API calls

### Serial Processing (Local Resources)
- TTS audio generation (uses local GPU/CPU)
- STT subtitle generation (uses local GPU/CPU)
- Video composition (ffmpeg operations)

This approach maximizes throughput while avoiding resource contention.

## Examples

### Basic Usage

```bash
# Generate from title
python scripts/generate_video.py --title "Fred se lance dans l'IA"

# Generate from existing story
python scripts/generate_video.py --story-file my_story.txt
```

### Advanced Usage

```bash
# Use custom config and prompt
python scripts/generate_video.py \
  --title "Fred" \
  --config configs/my_config.yaml \
  --prompt-file prompts/my_prompt.txt

# Override specific settings
python scripts/generate_video.py \
  --title "Fred" \
  --llm-provider openai \
  --llm-model gpt-4o \
  --tts-reference-audio /path/to/fred_voice.mp4 \
  --max-parallel-llm 10

# Verbose output for debugging
python scripts/generate_video.py --title "Fred" -v

# Quiet mode (only output video path)
python scripts/generate_video.py --title "Fred" --quiet
```

## Troubleshooting

### Common Issues

1. **LLM API errors**: Check API keys in environment variables
2. **TTS connection errors**: Ensure TTS service is running on specified host/port
3. **Video not found**: Check video index path in configuration
4. **Out of memory**: Reduce `max_parallel_llm_calls` or use smaller models

### Debug Mode

```bash
# Enable verbose output
python scripts/generate_video.py --title "Fred" -v
```

## Development

### Adding New Providers

1. Implement the appropriate interface in `video_generation_impl.py`
2. Add factory logic to create your implementation
3. Update configuration to support the new provider
4. Add tests in `test_video_generation.py`

Example:

```python
class MyCustomLLM(LLMInterface):
    async def complete(self, prompt: str, **kwargs) -> str:
        # Your implementation
        pass
```

### Project Structure

```
virtual-streamer/
├── scripts/
│   ├── generate_video.py              # Main CLI script
│   ├── video_generation_config.py     # Pydantic config models
│   ├── video_generation_interfaces.py # Abstract interfaces
│   ├── video_generation_impl.py       # Concrete implementations
│   └── video_generation_core.py       # Core logic functions
├── tests/
│   └── test_video_generation.py       # Unit and integration tests
├── configs/
│   ├── default_config.yaml            # Default configuration
│   └── example_custom.yaml            # Example custom config
├── prompts/
│   └── story_generation.txt           # Story generation prompt
└── output/
    └── (generated videos and config dumps)
```

## Documentation

### Essential Docs (in root)
- **This file** - Complete guide
- `QUICK_START_SIMPLIFIED.md` - 3-step quick start
- `USAGE_EXAMPLES.md` - 20+ examples
- `DOCS_INDEX.md` - Navigation guide

### Detailed Docs (in docs/archived/)
- Implementation details
- Migration guides  
- Comparisons
- Changelogs

## License

Same as the parent project.

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass before submitting

## References

- Original streamlit app: `apps/creation_interface.py`
- Workflow implementation: `virtual_streamer/workflows/virtual_streamer_workflow.py`
- Utils: `virtual_streamer/utils/utils.py`

