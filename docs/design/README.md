# Virtual Streamer - Refactoring Design

This folder contains the design documentation for refactoring the Virtual Streamer monorepo into a clean, maintainable architecture.

## Design Documents

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | High-level system architecture and principles |
| [Package Structure](./package-structure.md) | Detailed folder and package organization |
| [Data Models](./data-models.md) | Core entity models: Character, VideoClip, Collection |
| [API Services](./api-services.md) | ML service APIs: Wav2Lip, Face Detection, TTS, STT |
| [Agents](./agents.md) | Google ADK agent architecture and patterns |
| [Migration Plan](./migration-plan.md) | Step-by-step migration roadmap |

## Key Principles

1. **All ML processing via HTTP APIs** - Agents communicate with ML services through standardized REST APIs
2. **Shared agents, not per-app** - Agents are reusable across projects; apps are configurations
3. **Clean data models** - Single `Character` model, no redundant `CharacterReference`
4. **ADK structure** - Agents follow Google ADK conventions: `agent.py`, `prompt.py`, `callback.py`
5. **Docker for services only** - ML services run in Docker; agents run via ADK server

## Quick Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION CONFIGS                          │
│       (ai_jesus/, fred_et_jamy/, obs_streamer/)                │
└────────────────────────────┬────────────────────────────────────┘
                             │ configure
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ADK AGENTS                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              video_creator (orchestrator)                │   │
│  │                         │                                │   │
│  │    ┌────────────────────┼────────────────────┐          │   │
│  │    ▼                    ▼                    ▼          │   │
│  │ tts_agent         lip_sync_agent      video_composer    │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP API calls
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   API CLIENTS (@vs/core)                        │
│    TTSClient | STTClient | Wav2LipClient | FaceDetectionClient │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DOCKER COMPOSE (ML Services)                    │
│   Fish TTS:8003 | Whisper:8004 | Wav2Lip:8001 | FaceDet:8005   │
└─────────────────────────────────────────────────────────────────┘
```

## Current State Issues

The current monorepo mixes several concerns:
- AI Jesus (Twitch Q&A streamer)
- Fred & Jamy (video parody generation)  
- OBS streaming infrastructure
- Core ML services
- Video understanding/indexing

This leads to:
- Unclear dependencies
- Difficult testing
- Hard to onboard new developers
- Tight coupling between unrelated features

## Target State

A modular architecture where:
- `@vs/core` provides shared models and API clients
- ML services are independent, containerized APIs
- Agents are composable and reusable
- Applications are just configurations over agents

## Getting Started

1. Read [Architecture](./architecture.md) for the big picture
2. Review [Data Models](./data-models.md) for entity design
3. Check [API Services](./api-services.md) for service contracts
4. See [Migration Plan](./migration-plan.md) for implementation steps

