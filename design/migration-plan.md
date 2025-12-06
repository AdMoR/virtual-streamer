# Migration Plan

Step-by-step roadmap to refactor the Virtual Streamer monorepo.

## Overview

| Phase | Duration | Focus |
|-------|----------|-------|
| Phase 1 | Week 1 | Core package setup |
| Phase 2 | Week 1-2 | ML service APIs |
| Phase 3 | Week 2 | ADK agents |
| Phase 4 | Week 2-3 | Application configs |
| Phase 5 | Week 3 | Cleanup & documentation |

---

## Phase 1: Core Package Setup

**Goal:** Create `packages/vs-core/` with clean data models and API clients.

### Tasks

- [ ] **1.1** Create workspace structure
  ```bash
  mkdir -p packages/vs-core/src/vs_core/{models,api_clients,api_models,storage,utils}
  ```

- [ ] **1.2** Create workspace `pyproject.toml`
  ```toml
  [project]
  name = "virtual-streamer-workspace"
  version = "0.1.0"
  
  [tool.uv.workspace]
  members = ["packages/vs-core"]
  ```

- [ ] **1.3** Create `packages/vs-core/pyproject.toml`
  ```toml
  [project]
  name = "vs-core"
  version = "0.1.0"
  dependencies = ["pydantic>=2.0", "httpx>=0.28"]
  ```

- [ ] **1.4** Implement data models
  - `models/character.py` - Character, VoiceSample
  - `models/video_clip.py` - VideoClip, CharacterPresence
  - `models/collection.py` - Collection
  - `models/__init__.py` - Exports

- [ ] **1.5** Implement API models
  - `api_models/wav2lip.py` - Request/Response
  - `api_models/face_detection.py` - Request/Response
  - `api_models/tts.py` - Request/Response
  - `api_models/stt.py` - Request/Response

- [ ] **1.6** Implement API clients
  - `api_clients/base.py` - BaseAPIClient
  - `api_clients/wav2lip.py` - Wav2LipClient
  - `api_clients/face_detection.py` - FaceDetectionClient
  - `api_clients/tts.py` - TTSClient
  - `api_clients/stt.py` - STTClient

- [ ] **1.7** Migrate storage utilities
  - `storage/local_fs_client.py` ← from `virtual_streamer/utils/local_fs_client.py`
  - `storage/s3_client.py` ← from `virtual_streamer/utils/s3_client.py`

- [ ] **1.8** Migrate common utils
  - `utils/video.py` ← relevant parts from `virtual_streamer/utils/utils.py`
  - `utils/audio.py`
  - `utils/subtitle.py`

### Validation

```bash
# Install the package
uv pip install -e packages/vs-core

# Run tests
cd packages/vs-core && pytest

# Verify imports work
python -c "from vs_core.models import Character; print('OK')"
python -c "from vs_core.api_clients import Wav2LipClient; print('OK')"
```

---

## Phase 2: ML Service APIs

**Goal:** Create standalone FastAPI services for Wav2Lip and Face Detection.

### Tasks

- [ ] **2.1** Create service structure
  ```bash
  mkdir -p services/{wav2lip_service,face_detection_service,entity_service}/src
  ```

- [ ] **2.2** Create Wav2Lip service
  - `services/wav2lip_service/src/main.py` - FastAPI app
  - `services/wav2lip_service/src/inference.py` - ML logic (from `virtual_streamer/wav2lip/`)
  - `services/wav2lip_service/src/models.py` - Request/Response models
  - `services/wav2lip_service/Dockerfile`
  - `services/wav2lip_service/pyproject.toml`

- [ ] **2.3** Create Face Detection service
  - `services/face_detection_service/src/main.py`
  - `services/face_detection_service/src/inference.py` (from `face_detection/`)
  - `services/face_detection_service/src/models.py`
  - `services/face_detection_service/Dockerfile`
  - `services/face_detection_service/pyproject.toml`

- [ ] **2.4** Create Entity service
  - `services/entity_service/src/main.py` ← from `virtual_streamer/video_server/entity_webservice.py`
  - `services/entity_service/src/models.py`
  - `services/entity_service/Dockerfile`
  - `services/entity_service/pyproject.toml`

- [ ] **2.5** Create unified docker-compose
  - `infra/compose.yml` - All services

- [ ] **2.6** Test services
  ```bash
  cd infra && docker compose up -d
  curl http://localhost:8001/health  # Wav2Lip
  curl http://localhost:8005/health  # Face Detection
  curl http://localhost:8002/health  # Entity
  ```

### Validation

```bash
# Test Wav2Lip API
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/data/test.mp4", "audio_path": "/data/test.wav"}'

# Test via client
python -c "
from vs_core.api_clients.wav2lip import Wav2LipClient
client = Wav2LipClient()
print(client.health())
"
```

---

## Phase 3: ADK Agents

**Goal:** Create ADK agents with proper structure.

### Tasks

- [ ] **3.1** Create agent structure
  ```bash
  mkdir -p agents/{sub_agents/{tts_agent,lip_sync_agent,video_composer_agent},qa_responder,story_generator,video_creator}
  ```

- [ ] **3.2** Implement sub-agents
  - `agents/sub_agents/tts_agent/agent.py`
  - `agents/sub_agents/tts_agent/prompt.py`
  - `agents/sub_agents/lip_sync_agent/agent.py`
  - `agents/sub_agents/lip_sync_agent/prompt.py`
  - `agents/sub_agents/video_composer_agent/agent.py`

- [ ] **3.3** Implement QA responder (AI Jesus)
  - `agents/qa_responder/agent.py`
  - `agents/qa_responder/prompt.py`
  - `agents/qa_responder/callback.py`

- [ ] **3.4** Implement story generator (Fred & Jamy)
  - `agents/story_generator/agent.py`
  - `agents/story_generator/prompt.py`
  - Migrate prompts from `prompts/story_generation.txt`
  - Migrate scorer from `scripts/cest_pas_sorcier_parody_scorer.py`

- [ ] **3.5** Implement video creator orchestrator
  - `agents/video_creator/agent.py`
  - `agents/video_creator/prompt.py`

- [ ] **3.6** Remove old workflow
  - Delete `virtual_streamer/workflows/virtual_streamer_workflow.py` (LlamaIndex)
  - Delete `virtual_streamer/workflows/prompts.py`

### Validation

```bash
# Test agent import
python -c "from agents.qa_responder.agent import qa_responder; print(qa_responder.name)"

# Run agent via ADK
adk run qa_responder --input "What is love?"
```

---

## Phase 4: Application Configs

**Goal:** Create app configurations and migrate app-specific code.

### Tasks

- [ ] **4.1** Create app structure
  ```bash
  mkdir -p apps/{ai_jesus,fred_et_jamy,obs_streamer}/{characters,prompts}
  ```

- [ ] **4.2** AI Jesus app
  - `apps/ai_jesus/config.yaml`
  - `apps/ai_jesus/characters/jesus.yaml`
  - `apps/ai_jesus/prompts/qa_prompt.txt`
  - `apps/ai_jesus/README.md`

- [ ] **4.3** Fred & Jamy app
  - `apps/fred_et_jamy/config.yaml`
  - `apps/fred_et_jamy/characters/fred.yaml`
  - `apps/fred_et_jamy/characters/jamy.yaml`
  - `apps/fred_et_jamy/prompts/story_generation.txt` ← from `prompts/`
  - `apps/fred_et_jamy/scorer.py` ← from `scripts/cest_pas_sorcier_parody_scorer.py`
  - `apps/fred_et_jamy/README.md`

- [ ] **4.4** OBS Streamer app
  - `apps/obs_streamer/docker-compose.yml` ← from `compose_obs.yml`
  - `apps/obs_streamer/video_server.py` ← from `apps/obs_video_server.py`
  - `apps/obs_streamer/twitch_reader.py` ← from `virtual_streamer/twitch/`
  - `apps/obs_streamer/configs/` ← from `assets/obs_config/`

- [ ] **4.5** Create video indexer tool
  - `tools/video_indexer/` structure
  - Migrate from `scripts/florence_run.py`
  - Create CLI interface

### Validation

```bash
# Validate configs
python -c "import yaml; yaml.safe_load(open('apps/ai_jesus/config.yaml'))"

# Test video indexer
cd tools/video_indexer && python -m src.cli --help
```

---

## Phase 5: Cleanup & Documentation

**Goal:** Remove deprecated files and update documentation.

### Files to Delete

```bash
# Deprecated root files
rm webservice.py
rm webservice_prod.py
rm compose_demo.yml
rm setup.py

# Old requirements (use pyproject.toml instead)
rm requirements.txt
rm requirements_cpu.txt
rm requirements_demo.txt
rm requirements_video_generation.txt
rm requirements_ui.txt

# Old docker files (replaced by infra/compose.yml)
rm docker-compose.yml

# Deprecated modules
rm -rf virtual_streamer/workflows/  # Replaced by agents/
rm -rf virtual_streamer/video_server/  # Replaced by services/entity_service/

# Scripts (migrated elsewhere)
rm scripts/cest_pas_sorcier_parody_scorer.py  # → apps/fred_et_jamy/
rm scripts/florence_run.py  # → tools/video_indexer/

# Old apps (migrated)
rm apps/obs_video_server.py  # → apps/obs_streamer/
rm apps/demo.py  # Deprecated
```

### Files to Keep (for now)

```bash
# Keep until fully migrated
virtual_streamer/wav2lip/  # Reference until services/wav2lip_service/ is complete
face_detection/  # Reference until services/face_detection_service/ is complete
virtual_streamer/utils/  # Reference until packages/vs-core/utils/ is complete
```

### Documentation Updates

- [ ] Update `README.md` with new architecture
- [ ] Update `design/README.md` as single source of truth
- [ ] Add `CONTRIBUTING.md` with development guidelines
- [ ] Add `services/README.md` explaining service development
- [ ] Add `agents/README.md` explaining agent development

### Final Validation

```bash
# Full system test
cd infra && docker compose up -d
sleep 30  # Wait for services

# Test all services
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8005/health

# Test agent
adk run qa_responder --input "Hello"

# Run package tests
cd packages/vs-core && pytest
cd services/wav2lip_service && pytest
cd services/face_detection_service && pytest
```

---

## Migration Checklist

### Before Starting
- [ ] Create git branch: `git checkout -b refactor/modular-architecture`
- [ ] Backup current working state
- [ ] Document all environment variables in use

### After Each Phase
- [ ] All tests pass
- [ ] Docker services start and respond
- [ ] No import errors
- [ ] Git commit with descriptive message

### Before Merge
- [ ] All deprecated files removed
- [ ] Documentation updated
- [ ] CI/CD updated (if applicable)
- [ ] Team review completed

---

## Rollback Plan

If issues arise:

1. **Phase 1-2:** Old code still works, just don't use new packages
2. **Phase 3:** Old workflows remain until agents are validated
3. **Phase 4:** Apps are config-only, easy to revert
4. **Phase 5:** Create backup branch before deletions

```bash
# Create safety branch before Phase 5
git checkout -b backup/pre-cleanup
git checkout refactor/modular-architecture
```

