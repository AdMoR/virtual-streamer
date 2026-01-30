# Migration Plan

Step-by-step roadmap to refactor the Virtual Streamer monorepo.

## Migration Status Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Core package setup |
| Phase 2 | ✅ Complete | Unified API structure |
| Phase 3 | ✅ Complete | ADK agents |
| Phase 4 | 🔄 In Progress | Streaming infrastructure |
| Phase 5 | 📋 Planned | Cleanup & documentation |

---

## Phase 1: Core Package Setup ✅ COMPLETE

**Goal:** Create core library with clean data models and utilities.

### Completed Tasks

- [x] **1.1** Created `virtual_streamer/` package structure
- [x] **1.2** Created workspace `pyproject.toml` with UV configuration
- [x] **1.3** Implemented data models in `virtual_streamer/video_server/models.py`
  - `Character`, `VoiceSample`
  - `VideoClip`, `CharacterPresence`
  - Streaming models (`StreamConfig`, `MediaProgrammation`, `PlaylistEntry`)
- [x] **1.4** Implemented utilities in `virtual_streamer/utils/`
  - `character_loader.py` - Centralized character loading
  - `minio_client.py` - MinIO storage operations
  - `job_store.py` - Background job tracking
  - `file_manager.py` - File operations
  - `subtitle_utils.py` - SRT generation
  - `storage_interface.py` - Storage abstraction

### Validation

```bash
# Package is installed and working
python -c "from virtual_streamer.utils.character_loader import load_character; print('OK')"
python -c "from virtual_streamer.video_server.models import Character; print('OK')"
```

---

## Phase 2: Unified API Structure ✅ COMPLETE

**Goal:** Create layered API architecture with all services integrated.

### Completed Tasks

- [x] **2.1** Created unified API at `virtual_streamer/api/main.py`
- [x] **2.2** Implemented low-level entity APIs
  - `low_level/characters.py` - Character CRUD
  - `low_level/clips.py` - Video clip management
  - `low_level/streams.py` - Stream configuration
  - `low_level/programmations.py` - Scheduling
  - `low_level/playlist.py` - Playlist management
  - `low_level/story_templates.py` - Story templates
  - `low_level/articles.py` - Article management
- [x] **2.3** Implemented medium-level service APIs
  - `medium_level/tts.py` - Text-to-speech wrapper
  - `medium_level/stt.py` - Speech-to-text wrapper
  - `medium_level/wav2lip.py` - Lip-sync generation
- [x] **2.4** Implemented high-level application APIs
  - `high_level/video_generation.py` - Video generation pipeline
  - `high_level/jesus_agents.py` - AI Jesus agent endpoints
  - `high_level/legacy_qa.py` - Backward compatibility

### Validation

```bash
# Start the API
uvicorn virtual_streamer.api.main:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test character API
curl http://localhost:8000/api/v1/characters

# Test Jesus agent
curl -X POST http://localhost:8000/api/v1/jesus-agents/greeting/submit \
  -H "Content-Type: application/json" \
  -d '{"user_name": "TestUser"}'
```

---

## Phase 3: ADK Agents ✅ COMPLETE

**Goal:** Create ADK agents with proper structure.

### Completed Tasks

- [x] **3.1** Created `virtual_streamer/lib/agents/` base classes
  - `BaseLlmAgent` - Simple LLM agent
  - `StatefulLlmAgent` - Agent with state management
  - `MapReduceAgent` - Parallel processing agent
  - `MapperAgent`, `AggregatorAgent` - Map-reduce components
  - Callback base classes
- [x] **3.2** Created agent factory at `virtual_streamer/agents/factory.py`
- [x] **3.3** Implemented character agents
  - `greeting_jesus_agent/` - Twitch greeting agent
  - `answering_jesus_agent/` - Q&A agent
- [x] **3.4** Implemented content generation agents
  - `story_generator/` - Parody story generation
  - `keyword_generator/` - Keyword extraction
  - `rubric_builder_agent/` - Video rubric building
  - `rubric_builder_map_reduce/` - Parallel rubric building
- [x] **3.5** Implemented video processing agents
  - `video_matcher/` - Video-to-dialogue matching
  - `sentence_video_matcher/` - Per-sentence matching
- [x] **3.6** Implemented orchestration agents
  - `orchestrator/` - Pipeline coordination
  - `virtual_streamer_agent/` - Main streaming agent with tools

### Validation

```bash
# Test agent factory
python -c "
from virtual_streamer.agents.factory import list_agents, get_agent
print('Available agents:', list_agents())
agent = get_agent('greeting_jesus_agent')
print('Agent loaded:', agent.name)
"
```

---

## Phase 4: Streaming Infrastructure 🔄 IN PROGRESS

**Goal:** Complete streaming infrastructure and integration.

### Completed Tasks

- [x] **4.1** Created streaming models in `virtual_streamer/streaming/models.py`
  - `StreamConfig` - Stream configuration
  - `MediaProgrammation` - Time-based scheduling
  - `PlaylistEntry` - Playlist entries
  - `PlaylistStatus` - Status enum
- [x] **4.2** Created streaming store in `virtual_streamer/streaming/store.py`
- [x] **4.3** Created video server in `virtual_streamer/streaming/video_server/`
  - `app.py` - FastAPI proxy
  - `static/index.html` - HTML5 video player
- [x] **4.4** Created Twitch integration in `virtual_streamer/streaming/twitch/`
  - `chat_reader.py` - IRC chat reader
- [x] **4.5** Created agent loop in `virtual_streamer/streaming/agent_loop/`
  - Continuous agent runner for streaming

### Remaining Tasks

- [ ] **4.6** Test end-to-end streaming flow
- [ ] **4.7** Update Docker compose for streaming stack
- [ ] **4.8** Create streaming setup scripts

### Validation

```bash
# Test streaming models
python -c "
from virtual_streamer.streaming.models import StreamConfig, MediaProgrammation
print('Streaming models OK')
"

# Test video server
python -c "
from virtual_streamer.streaming.video_server.app import app
print('Video server app OK')
"
```

---

## Phase 5: Cleanup & Documentation 📋 PLANNED

**Goal:** Remove deprecated files and finalize documentation.

### Documentation Updates

- [x] Update `docs/design/package-structure.md`
- [x] Update `docs/design/architecture.md`
- [x] Update `docs/design/agents.md`
- [x] Update `docs/design/api-services.md`
- [x] Update `docs/design/migration-plan.md`
- [ ] Update main `README.md`

### Files to Keep (Reference)

```bash
# Keep for now as reference
virtual_streamer/wav2lip/          # Wav2Lip inference code
face_detection/                    # Face detection module
models/                            # ML model definitions
```

### Files to Eventually Remove

```bash
# Deprecated when migration complete
requirements.txt                   # Use pyproject.toml instead
requirements_*.txt                 # Various requirement files
setup.py                          # Use pyproject.toml instead
```

---

## Migration Checklist

### Before Starting
- [x] Create git branch for migration
- [x] Document all environment variables

### After Each Phase
- [x] Phase 1: Tests pass, imports work
- [x] Phase 2: API endpoints respond
- [x] Phase 3: Agents load and run
- [x] Phase 4: Streaming stack works
- [ ] Phase 5: Documentation complete

### Before Final Release
- [ ] All deprecated files removed
- [ ] Documentation updated
- [ ] CI/CD updated
- [ ] Team review completed

---

## Architecture Decisions Made

### Decision 1: Unified API Instead of Microservices

**Why?**
- Single GPU instance for all ML models
- Simplified deployment and monitoring
- Consistent API patterns
- Easier development and testing

### Decision 2: ADK Over LlamaIndex Workflows

**Why?**
- Better tooling and debugging
- More flexible agent composition
- Native support for structured output
- Better state management

### Decision 3: MinIO for All Storage

**Why?**
- S3-compatible API
- Self-hosted for development
- Can migrate to real S3 for production
- Single storage interface

### Decision 4: MySQL for Streaming State

**Why?**
- Robust transactional support
- Easy querying for playlist management
- Proven at scale
- Good tooling ecosystem

---

## Rollback Plan

If issues arise during migration:

1. **Phase 1-2:** Old code still works, just don't use new packages
2. **Phase 3:** Old workflows remain until agents are validated
3. **Phase 4:** Apps are config-only, easy to revert
4. **Phase 5:** Create backup branch before deletions

```bash
# Create safety branch before Phase 5
git checkout -b backup/pre-cleanup
git checkout main
```

---

## Next Steps

1. Complete Phase 4 streaming validation
2. Update main README.md
3. Clean up deprecated code
4. Final testing and release
