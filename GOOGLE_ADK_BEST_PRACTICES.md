# Google ADK Agents: Best Practices Guide

A comprehensive guide for building production-ready agents using Google's Agent Development Kit (ADK), derived from real-world implementations.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Agent Types](#agent-types)
- [Best Practices Checklist](#best-practices-checklist)
- [Code Patterns](#code-patterns)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Architecture Overview

Google ADK agents follow a **modular, dependency-injected architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Hydra Configuration                       │
│              (conf/agentic/stable.yaml)                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ - LLM model (provider, model, parameters)                   │ │
│  │ - Agent metadata (name, version, description, owners)       │ │
│  │ - Behavior controls (transfer policies, include_contents)   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AdkraftLlmAgent                             │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │ InstructionProvider │ Output Schema  │  │    Callbacks    │  │
│  │ (dynamic prompts)   │  (Pydantic)    │  │  (lifecycle)    │  │
│  └──────────────────┘  └────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Shared State                             │
│  (CallbackContext.state - persisted across agent lifecycle)      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Configuration-Driven**: Model selection, parameters, and behavior controls live in YAML, not Python code
2. **Dependency Injection**: All dependencies passed through constructors via factory functions
3. **Factory Pattern**: `get_<agent_name>()` functions create fully-configured agent instances
4. **Singleton Caching**: `@lru_cache` ensures single instances of callbacks and providers
5. **Type Safety**: Pydantic schemas for all inputs/outputs

---

## Directory Structure

Each agent lives in its own directory with a standardized file structure:

```
sub_agents_adk/
├── common/                          # Shared utilities
│   └── before_agent_callbacks.py    # Reusable callbacks
├── my_agent/
│   ├── __init__.py                  # Exports (required)
│   ├── agent.py                     # Agent class + factory (required)
│   ├── schema.py                    # Pydantic output model (required)
│   ├── prompt.py                    # Static prompts/templates (required)
│   ├── callback.py                  # Lifecycle callbacks (optional)
│   ├── instruction_provider.py      # Dynamic prompts (optional)
│   └── temp_state_variables.py      # State key constants (optional)
```

### File Responsibilities

| File | Purpose |
|------|---------|
| `__init__.py` | Export the agent instance: `from .agent import my_agent` |
| `agent.py` | Define agent class, wire dependencies, expose `root_agent` |
| `schema.py` | Pydantic models for structured LLM output |
| `prompt.py` | Static prompt strings and templates |
| `callback.py` | Process inputs/outputs at lifecycle stages |
| `instruction_provider.py` | Generate context-aware prompts dynamically |

---

## Core Components

### 1. Agent Class (`agent.py`)

The agent class wires together all components and inherits from `AdkraftLlmAgent`:

```python
from your_project.config import AgenticSettings, get_agentic_configuration
from your_project.lib.agents import AdkraftLlmAgent  # Your ADK wrapper
from .prompt import MY_AGENT_PROMPT
from .schema import MyAgentOutput
from .callback import get_process_callback

class MyAgent(AdkraftLlmAgent):
    def __init__(
        self,
        agentic_configuration: AgenticSettings,
        process_callback,  # Inject dependencies
    ):
        super().__init__(
            name="my_agent",  # Must match conf/agentic/stable.yaml
            agentic_configuration=agentic_configuration,
            instruction=MY_AGENT_PROMPT,
            output_schema=MyAgentOutput,
            after_model_callback=[process_callback],
        )

def get_my_agent():
    """Factory function - the single entry point for creating this agent."""
    return MyAgent(
        agentic_configuration=get_agentic_configuration(),
        process_callback=get_process_callback(),
    )

root_agent = get_my_agent()  # Required: expose to ADK server
```

**Key Points:**
- Always expose `root_agent` at module level
- Use factory function pattern (`get_my_agent()`)
- Name must match Hydra configuration entry
- All callbacks/providers injected via constructor

### 2. Output Schema (`schema.py`)

Define structured output using Pydantic with rich field descriptions:

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class MyAgentOutput(BaseModel):
    """Structured output for my_agent."""
    
    answer: str = Field(
        description="The main response to the user query."
    )
    risk_level: RiskLevel = Field(
        description="Risk assessment: LOW|MEDIUM|HIGH"
    )
    confidence: Optional[float] = Field(
        description="Confidence score between 0 and 1.",
        default=None,
        ge=0.0,
        le=1.0,
    )
    reasoning: Optional[str] = Field(
        description="Brief explanation of the decision.",
        default=None,
    )
```

**Best Practices:**
- Use enums for categorical outputs (LLM sees the allowed values)
- Add validation constraints (`ge`, `le`, `min_length`)
- Provide detailed `description` - this becomes part of the prompt
- Use `Optional` with sensible defaults for non-required fields

### 3. Prompts (`prompt.py`)

Organize prompts into composable sections:

```python
SYSTEM_ROLE = """You are an expert assistant specializing in {domain}.
Your task is to {primary_task}."""

OUTPUT_FORMAT = """Respond in {language} following this structure:
- Answer: Your main response
- Confidence: A score from 0 to 1"""

EXAMPLES = """Examples:

Example 1:
User: {example_input_1}
Assistant: {example_output_1}

Example 2:
User: {example_input_2}
Assistant: {example_output_2}"""

IMPORTANT_NOTES = """Important:
- Always cite your sources
- If unsure, acknowledge uncertainty"""

# Template for dynamic assembly
PROMPT_TEMPLATE = """
{system_role}

{context}

{output_format}

{examples}

{important_notes}
"""

def build_prompt():
    """Build the complete prompt from components."""
    return f"""
{SYSTEM_ROLE}

{OUTPUT_FORMAT}

{EXAMPLES}

{IMPORTANT_NOTES}
""".strip()

MY_AGENT_PROMPT = build_prompt()
```

**Best Practices:**
- Separate concerns (role, format, examples, constraints)
- Use template placeholders for dynamic content
- Export a ready-to-use constant for simple cases
- Provide builder functions for complex prompts

### 4. Callbacks (`callback.py`)

Implement lifecycle hooks using abstract base classes:

```python
from functools import lru_cache
from typing_extensions import override
import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

from your_project.lib.agents.callbacks import (  # Your callback base classes
    AfterModelCallback,
    BeforeModelCallback,
    AgentCallback,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# AFTER MODEL CALLBACK - Runs after LLM returns response
# ═══════════════════════════════════════════════════════════════
class StoreResultCallback(AfterModelCallback):
    """Extract and store LLM response in state."""
    
    @override
    async def __call__(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> None:
        parsed = extract_llm_response_json(llm_response)
        if not isinstance(parsed, dict):
            logger.error("Failed to parse LLM response")
            return
        
        # Store in shared state
        callback_context.state["my_result"] = parsed.get("answer")
        callback_context.state["confidence"] = parsed.get("confidence")

@lru_cache
def get_store_result_callback():
    return StoreResultCallback()

# ═══════════════════════════════════════════════════════════════
# BEFORE MODEL CALLBACK - Runs before LLM call, can skip execution
# ═══════════════════════════════════════════════════════════════
class SkipIfConditionCallback(BeforeModelCallback):
    """Skip agent if condition is met (return Content to skip)."""
    
    def __init__(self, feature_enabled: bool):
        self.feature_enabled = feature_enabled
    
    @override
    async def __call__(self, callback_context: CallbackContext):
        if not self.feature_enabled:
            # Return Content to skip LLM call
            return types.Content(
                parts=[types.Part(text=f"Agent skipped: feature disabled")],
                role="model",
            )
        # Return None to continue normal execution
        return None

@lru_cache
def get_skip_callback():
    from your_project.config import get_settings
    return SkipIfConditionCallback(get_settings().FEATURE_ENABLED)

# ═══════════════════════════════════════════════════════════════
# AGENT CALLBACK - Runs at agent start/end
# ═══════════════════════════════════════════════════════════════
class ComputeDerivedValueCallback(AgentCallback):
    """Compute derived values from state after agent completes."""
    
    @override
    async def __call__(self, callback_context: CallbackContext):
        confidence = callback_context.state.get("confidence", 0.0)
        callback_context.state["is_high_confidence"] = confidence > 0.8
```

**Callback Lifecycle Order:**
1. `before_agent_callback` → Agent starts
2. `before_model_callback` → Before LLM call (can skip)
3. **LLM Execution**
4. `after_model_callback` → After LLM returns
5. `after_agent_callback` → Agent completes

### 5. Instruction Provider (`instruction_provider.py`)

Generate dynamic prompts based on runtime context:

```python
from abc import abstractmethod
from functools import lru_cache
from typing_extensions import override
import logging

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils import instructions_utils

from your_project.lib.agents.instruction_provider import InstructionProvider  # Your base class
from .prompt import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class MyInstructionProvider(InstructionProvider):
    """Base class for instruction generation."""
    
    @abstractmethod
    async def __call__(self, context: ReadonlyContext): ...
    
    def _build_context_section(self, context: ReadonlyContext) -> str:
        """Helper to build context-specific prompt section."""
        user_data = context.state.get("user_context", {})
        parts = []
        if name := user_data.get("name"):
            parts.append(f"User: {name}")
        if history := user_data.get("history"):
            parts.append(f"History: {history}")
        return "\n".join(parts)


class StandardInstructionProvider(MyInstructionProvider):
    """Standard instruction generation with full context."""
    
    @override
    async def __call__(self, context: ReadonlyContext):
        language = context.state.get("language", "en")
        context_section = self._build_context_section(context)
        
        template = PROMPT_TEMPLATE.format(
            language=language,
            context=context_section,
        )
        
        logger.info(f"Generated prompt for language: {language}")
        
        # Inject session state variables into template
        return await instructions_utils.inject_session_state(template, context)


class SimplifiedInstructionProvider(MyInstructionProvider):
    """Simplified instructions for specific scenarios."""
    
    @override
    async def __call__(self, context: ReadonlyContext):
        return "You are a helpful assistant. Answer the user's question."


@lru_cache
def get_instruction_provider():
    """Factory: choose provider based on settings."""
    from your_project.config import get_settings
    if get_settings().USE_SIMPLIFIED_MODE:
        return SimplifiedInstructionProvider()
    return StandardInstructionProvider()
```

---

## Agent Types

### LlmAgent (Single Agent)

Standard agent for single-task scenarios:

```python
class MyAgent(AdkraftLlmAgent):
    def __init__(self, agentic_configuration, ...):
        super().__init__(
            name="my_agent",
            agentic_configuration=agentic_configuration,
            instruction=PROMPT,
            output_schema=OutputSchema,
        )
```

### SequentialAgent (Pipeline)

Execute sub-agents in order, each building on previous results:

```python
from google.adk.agents import SequentialAgent

class MyPipelineAgent(SequentialAgent):
    def __init__(
        self,
        first_agent: FirstAgent,
        second_agent: SecondAgent,
    ):
        super().__init__(
            name="pipeline_agent",
            sub_agents=[first_agent, second_agent],
        )

def get_pipeline_agent():
    return MyPipelineAgent(
        first_agent=get_first_agent(),
        second_agent=get_second_agent(),
    )

root_agent = get_pipeline_agent()
```

**Use Cases:**
- Multi-step reasoning (filter → analyze → decide)
- Progressive refinement
- Chain-of-thought workflows

### ParallelAgent (Concurrent)

Execute sub-agents simultaneously for independent tasks:


**Use Cases:**
- Independent security checks
- Multi-model consensus
- Parallel analysis streams

---

## Best Practices Checklist

### ✅ Configuration

- [ ] Add agent to `conf/agentic/stable.yaml` before implementing
- [ ] Agent `name` in Python matches YAML configuration
- [ ] Model/behavior settings in YAML, not hardcoded in Python
- [ ] Update version in YAML when behavior changes significantly
- [ ] Keep owners list current

### ✅ Code Organization

- [ ] One agent per directory
- [ ] Factory function `get_<agent>()` as single entry point
- [ ] `root_agent` exposed at module level
- [ ] Type hints on all functions and methods

### ✅ Schemas

- [ ] Descriptive field names
- [ ] Detailed `description` for each field (used in prompts!)
- [ ] Validation constraints where applicable
- [ ] Enums for categorical values

### ✅ Prompts

- [ ] Separated into logical sections
- [ ] Template placeholders for dynamic content
- [ ] Examples included for complex tasks
- [ ] Language/locale-aware where needed

### ✅ Callbacks

- [ ] Inherit from correct base class
- [ ] Use `@override` decorator
- [ ] Log important events (INFO) and errors (ERROR)
- [ ] Handle failures gracefully
- [ ] Single responsibility per callback

### ✅ State Management

- [ ] Use `.get()` with defaults for optional state
- [ ] Define state keys as constants
- [ ] Document expected state in docstrings
- [ ] Clean up temporary state when appropriate

---

## Code Patterns

### Pattern 1: Conditional Agent Skipping

Skip agent execution based on state:

```python
class SkipIfNotReady(AgentCallback):
    async def __call__(self, callback_context: CallbackContext):
        if callback_context.state.get("status") != "ready":
            return types.Content(
                parts=[types.Part(text="Skipped: not ready")],
                role="model",
            )
        return None
```

### Pattern 2: State Key Constants

Prevent typos with centralized constants:

```python
# temp_state_variables.py
TASK_STATUS = "task_status"
CONFIDENCE = "confidence"
```

### Pattern 3: Shared Callbacks

Reuse callbacks across agents via a `common/` directory:

```python
# common/before_agent_callbacks.py
@lru_cache
def get_log_entry_callback():
    return LogEntryCallback()

# agent.py
from your_agents.common.before_agent_callbacks import get_log_entry_callback

class MyAgent(AdkraftLlmAgent):
    def __init__(self, ...):
        super().__init__(
            before_agent_callback=[get_log_entry_callback()],
            ...
        )
```

### Pattern 4: Multiple Instruction Provider Variants

Choose provider based on configuration:

```python
@lru_cache
def get_instruction_provider():
    settings = get_settings()
    if settings.use_vanilla_mode:
        return VanillaInstructionProvider()
    elif settings.use_strict_mode:
        return StrictInstructionProvider()
    return StandardInstructionProvider()
```

---

## Anti-Patterns to Avoid

### ❌ Hardcoding Model Configuration

```python
# BAD
super().__init__(
    model="gpt-4",
    temperature=0.7,
    ...
)

# GOOD - Use Hydra configuration
super().__init__(
    agentic_configuration=agentic_configuration,  # Model comes from YAML
    ...
)
```

### ❌ Direct Instantiation Without Factory

```python
# BAD
agent = MyAgent(config)

# GOOD
agent = get_my_agent()
root_agent = get_my_agent()
```

### ❌ Missing @lru_cache on Factories

```python
# BAD - Creates new instance every call
def get_callback():
    return MyCallback()

# GOOD - Singleton
@lru_cache
def get_callback():
    return MyCallback()
```

### ❌ Complex Logic in Callbacks

```python
# BAD - Too much responsibility
class DoEverythingCallback(AfterModelCallback):
    async def __call__(self, ctx, resp):
        data = parse(resp)
        validated = validate(data)
        enriched = enrich(validated)
        saved = save(enriched)
        notified = notify(saved)
        ...

# GOOD - Single responsibility
class ParseResponseCallback(AfterModelCallback): ...
class ValidateCallback(AgentCallback): ...
class EnrichCallback(AgentCallback): ...
```

### ❌ State Key Typos

```python
# BAD - Typo-prone
state["taksStatus"] = "complete"  # Typo!
status = state.get("task_status")  # Returns None

# GOOD - Use constants
from .temp_state_variables import TASK_STATUS
state[TASK_STATUS] = "complete"
status = state.get(TASK_STATUS)
```

---

## Quick Reference

### Callback Base Classes

| Base Class | When It Runs | Can Skip Execution? |
|------------|--------------|---------------------|
| `AgentCallback` | `before_agent` / `after_agent` | Yes (return Content) |
| `BeforeModelCallback` | Before LLM call | Yes (return Content) |
| `AfterModelCallback` | After LLM returns | No |

### Key Imports

```python
# Configuration (adapt paths to your project)
from your_project.config import AgenticSettings, get_agentic_configuration

# Base classes (your wrapper around ADK)
from your_project.lib.agents import AdkraftLlmAgent
from your_project.lib.agents.callbacks import (
    AfterModelCallback,
    BeforeModelCallback,
    AgentCallback,
)
from your_project.lib.agents.instruction_provider import InstructionProvider

# Google ADK agent types (direct from ADK)
from google.adk.agents import SequentialAgent, ParallelAgent

# Google ADK utilities (direct from ADK)
from google.adk.utils import instructions_utils
from google.genai import types
```

### Minimal Agent Template

```python
# schema.py
from pydantic import BaseModel, Field

class MyOutput(BaseModel):
    answer: str = Field(description="The response")

# prompt.py
PROMPT = "You are a helpful assistant."

# agent.py
from your_project.config import get_agentic_configuration
from your_project.lib.agents import AdkraftLlmAgent  # Your ADK wrapper
from .schema import MyOutput
from .prompt import PROMPT

class MyAgent(AdkraftLlmAgent):
    def __init__(self, agentic_configuration):
        super().__init__(
            name="my_agent",
            agentic_configuration=agentic_configuration,
            instruction=PROMPT,
            output_schema=MyOutput,
        )

def get_my_agent():
    return MyAgent(get_agentic_configuration())

root_agent = get_my_agent()

# __init__.py
from .agent import root_agent
__all__ = ["root_agent"]
```

---

*Best practices derived from production Google ADK agent implementations.*

