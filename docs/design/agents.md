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

---

## Agent Factory

Agents can be loaded dynamically using the factory registry pattern:

```python
# virtual_streamer/agents/factory.py

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
                         ┌──────────────────────┐
                         │ virtual_streamer_agent│ ← Main streaming agent
                         └──────────┬───────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ story_generator │      │  video_matcher  │      │   orchestrator  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
          │                        │
          │                        ├── sentence_video_matcher
          │                        │
          │              ┌─────────────────┐
          │              │ keyword_generator│
          │              └─────────────────┘
          │
┌─────────────────┐
│ rubric_builder  │── rubric_builder_map_reduce
└─────────────────┘
```

---

## Current Agents

### Character Agents (AI Jesus)

#### Greeting Jesus Agent

Generates personalized greetings for new viewers, analyzing their username with "divine insight".

**Location:** `virtual_streamer/agents/greeting_jesus_agent/`

```python
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

#### Answering Jesus Agent

Answers viewer questions with dark humor and biblical references.

**Location:** `virtual_streamer/agents/answering_jesus_agent/`

```python
from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.answering_jesus_agent.prompt import PROMPT

class AnsweringJesusAgent(BaseLlmAgent):
    def __init__(self):
        super().__init__(
            name="answering_jesus_agent",  
            instruction=PROMPT,
            output_schema=None,  # Free-form text output
        )
```

**Prompt characteristics:**
- Stand-up comedian pretending to be Jesus
- Mean, sarcastic, dark humor
- Uses French street slang
- Calls audience "mon petit pécheur"
- ~4 sentences output

### Content Generation Agents

#### Story Generator

Generates parody stories for C'est pas Sorcier and similar content.

**Location:** `virtual_streamer/agents/story_generator/`

**Files:**
- `agent.py` - Agent definition with structured output
- `prompt.py` - Story generation prompt
- `schema.py` - `StoryGeneratorOutput` model
- `callback.py` - State management callbacks

```python
class StoryGeneratorOutput(BaseModel):
    title: str
    dialogue: List[DialogueLine]
    
class DialogueLine(BaseModel):
    character_id: str
    text: str
```

#### Keyword Generator

Extracts keywords from text for video search.

**Location:** `virtual_streamer/agents/keyword_generator/`

```python
class KeywordGeneratorAgent(BaseLlmAgent):
    """Extracts relevant keywords from dialogue text."""
```

#### Rubric Builder Agent

Builds video rubrics (structured descriptions) for video matching.

**Location:** `virtual_streamer/agents/rubric_builder_agent/`

**Files:**
- `agent.py` - Agent definition
- `prompt.py` - Rubric building prompt
- `schema.py` - `RubricOutput` model

#### Rubric Builder Map-Reduce

Parallel processing version of rubric builder for batch operations.

**Location:** `virtual_streamer/agents/rubric_builder_map_reduce/`

**Files:**
- `agent.py` - Contains Mapper, Aggregator, and factory function
- `callback.py` - State callbacks for parallel workers

```python
class RubricMapper(MapperAgent):
    """Splits input into items for parallel processing."""
    
    def build_items_from_state(self, ctx):
        # Split dialogue into individual lines
        ...

class RubricAggregator(AggregatorAgent[RubricAggregatorOutput]):
    """Combines results from parallel workers."""
    
    async def aggregation_fn(self, results):
        # Merge all rubric outputs
        ...

def create_rubric_builder_map_reduce() -> MapReduceAgent:
    """Factory function for the map-reduce agent."""
    ...
```

### Video Processing Agents

#### Video Matcher

Matches videos to dialogue lines using semantic search.

**Location:** `virtual_streamer/agents/video_matcher/`

**Files:**
- `agent.py` - Main agent definition
- `aggregator.py` - Results aggregation logic
- `prompt.py` - Video matching prompt
- `schema.py` - Input/output schemas
- `callback.py` - State management

```python
class VideoMatcherInput(BaseModel):
    dialogue_lines: List[DialogueLine]
    collection: str
    
class VideoMatcherOutput(BaseModel):
    matches: List[VideoMatch]
    
class VideoMatch(BaseModel):
    dialogue_index: int
    video_path: str
    similarity_score: float
```

#### Sentence Video Matcher

Per-sentence video matching for fine-grained video selection.

**Location:** `virtual_streamer/agents/sentence_video_matcher/`

**Files:**
- `agent.py` - Agent definition
- `schema.py` - Input/output models
- `utils.py` - Helper functions

### Orchestration Agents

#### Orchestrator

Coordinates the full video generation pipeline.

**Location:** `virtual_streamer/agents/orchestrator/`

```python
class OrchestratorAgent:
    """
    Coordinates the full video generation pipeline:
    1. Story generation
    2. Keyword extraction
    3. Video matching
    4. Video generation
    """
```

#### Virtual Streamer Agent

Main streaming agent with tool access for live operations.

**Location:** `virtual_streamer/agents/virtual_streamer_agent/`

**Structure:**
```
virtual_streamer_agent/
├── agent.py           # Main agent definition
├── prompt.py          # System prompt
├── schema.py          # Input/output schemas
├── callbacks/
│   └── context_injector.py   # Injects context before agent
├── context/
│   ├── builder.py           # Context construction
│   ├── chat_provider.py     # Chat context provider
│   ├── conversation.py      # Conversation management
│   ├── providers.py         # Provider registry
│   └── queue_provider.py    # Queue-based context
└── tools/
    ├── base.py              # Tool base class
    ├── create_video.py      # Video creation tool
    ├── send_message.py      # Message sending tool
    └── factory.py           # Tool factory
```

**Tools Available:**
- `create_video` - Generate video from text
- `send_message` - Send chat message

---

## Agent Base Classes

Located in `virtual_streamer/lib/agents/`:

### BaseLlmAgent

Simple LLM-based agent with optional structured output.

```python
from virtual_streamer.lib.agents import BaseLlmAgent

class MyAgent(BaseLlmAgent):
    def __init__(self):
        super().__init__(
            name="my_agent",
            instruction="Your prompt here",
            output_schema=MyOutputSchema,  # Optional Pydantic model
        )
```

### StatefulLlmAgent

Agent with state management across invocations.

```python
from virtual_streamer.lib.agents import StatefulLlmAgent

class MyStatefulAgent(StatefulLlmAgent):
    def __init__(self):
        super().__init__(
            name="my_stateful_agent",
            instruction="Your prompt",
            state_input_callback=MyInputCallback(),
            state_output_callback=MyOutputCallback(),
        )
```

### MapReduceAgent

Agent for parallel processing of large inputs.

```python
from virtual_streamer.lib.agents import (
    MapReduceAgent,
    MapperAgent,
    AggregatorAgent,
)

class MyMapper(MapperAgent):
    def build_items_from_state(self, ctx):
        # Return list of items to process
        return items

class MyAggregator(AggregatorAgent[OutputSchema]):
    async def aggregation_fn(self, results):
        # Combine results from all workers
        return combined_result

def create_my_map_reduce_agent() -> MapReduceAgent:
    return MapReduceAgent(
        mapper=MyMapper(),
        aggregator=MyAggregator(),
        worker_factory=lambda: get_worker_agent(),
    )
```

---

## Callbacks

### BeforeModelCallback / AfterModelCallback

Pre/post processing around LLM calls.

```python
from virtual_streamer.lib.agents import BeforeModelCallback, AfterModelCallback

class MyBeforeCallback(BeforeModelCallback):
    async def run_before(self, ctx, llm_request):
        # Modify request before LLM
        return llm_request

class MyAfterCallback(AfterModelCallback):
    async def run_after(self, ctx, llm_response):
        # Process response after LLM
        return llm_response
```

### StateInputCallback / StateOutputCallback

State injection and extraction for stateful agents.

```python
from virtual_streamer.lib.agents import StateInputCallback, StateOutputCallback

class MyInputCallback(StateInputCallback):
    async def inject_state(self, ctx):
        # Add state to context before agent runs
        ctx.state["my_data"] = load_data()

class MyOutputCallback(StateOutputCallback):
    async def extract_state(self, ctx, response):
        # Save state after agent completes
        save_data(ctx.state["my_data"])
```

---

## API Integration

Both Jesus agents are exposed via the Jesus Agents API:

```bash
# Greeting
POST /api/v1/jesus-agents/greeting/submit
{"user_name": "DrPetitesFesses"}

# Answering
POST /api/v1/jesus-agents/answering/submit
{"question": "Comment fait-on les hosties?", "user_name": "Dany"}
```

The API wraps the agent with a full video pipeline:
```
Agent → TTS → Wav2Lip → STT → Subtitles → MinIO Upload
```

### Pipeline Details

1. **Agent**: Generate text response using ADK agent
2. **TTS**: Convert text to audio using character's voice
3. **Wav2Lip**: Generate lip-synced video from character's reference video
4. **Combine**: Merge video and audio tracks
5. **STT**: Transcribe audio to SRT subtitles
6. **Subtitles**: Burn subtitles into video
7. **Upload**: Store final video in MinIO

---

## Running Agents

### Via API

```bash
# Start the API server
uvicorn virtual_streamer.api.main:app --host 0.0.0.0 --port 8000

# Call an agent endpoint
curl -X POST http://localhost:8000/api/v1/jesus-agents/greeting/submit \
  -H "Content-Type: application/json" \
  -d '{"user_name": "TestUser"}'
```

### Via ADK CLI

```bash
# ADK agents are mounted at /adk
# Use the ADK web interface
adk web --agents-dir ./virtual_streamer/agents

# Or run directly via Python
python -c "
from virtual_streamer.agents.factory import get_agent
agent = get_agent('greeting_jesus_agent')
result = agent.run({'user_name': 'TestUser'})
print(result)
"
```

### Via Test Interface

```bash
# Run the agent test interface
python apps/agent_test_interface.py
```

---

## Adding New Agents

1. **Create agent folder:**
```bash
mkdir -p virtual_streamer/agents/my_new_agent
touch virtual_streamer/agents/my_new_agent/__init__.py
touch virtual_streamer/agents/my_new_agent/agent.py
touch virtual_streamer/agents/my_new_agent/prompt.py
```

2. **Define the agent in `agent.py`:**
```python
from virtual_streamer.lib.agents import BaseLlmAgent
from virtual_streamer.agents.my_new_agent.prompt import PROMPT

class MyNewAgent(BaseLlmAgent):
    def __init__(self):
        super().__init__(
            name="my_new_agent",
            instruction=PROMPT,
        )

def get_my_new_agent():
    return MyNewAgent()

root_agent = get_my_new_agent()
```

3. **Register in factory:**
```python
# virtual_streamer/agents/factory.py

@register_agent("my_new_agent")
def _get_my_new_agent():
    from virtual_streamer.agents.my_new_agent.agent import get_my_new_agent
    return get_my_new_agent()
```

4. **Test the agent:**
```python
from virtual_streamer.agents.factory import get_agent

agent = get_agent("my_new_agent")
result = agent.run({"input": "test"})
```
