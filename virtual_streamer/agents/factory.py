"""
Agent Factory Registry.

Provides a simple factory pattern for loading ADK agents by name.
"""

from typing import Dict, Callable, Any

# Type for agent factory functions
AgentFactory = Callable[[], Any]

# Registry of known agents
_AGENT_REGISTRY: Dict[str, AgentFactory] = {}


def register_agent(name: str):
    """Decorator to register an agent factory."""
    def decorator(factory_fn: AgentFactory):
        _AGENT_REGISTRY[name] = factory_fn
        return factory_fn
    return decorator


def get_agent(agent_name: str):
    """
    Get an agent instance by name.
    
    Args:
        agent_name: Name of the agent (e.g., "greeting_jesus_agent")
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If agent not found in registry
    """
    if agent_name not in _AGENT_REGISTRY:
        raise ValueError(
            f"Agent '{agent_name}' not found. "
            f"Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return _AGENT_REGISTRY[agent_name]()


def list_agents() -> list[str]:
    """Return list of registered agent names."""
    return list(_AGENT_REGISTRY.keys())


# =============================================================================
# Register known agents
# =============================================================================

@register_agent("greeting_jesus_agent")
def _get_greeting_jesus():
    from virtual_streamer.agents.greeting_jesus_agent.agent import get_greeting_jesus_agent
    return get_greeting_jesus_agent()


@register_agent("answering_jesus_agent")
def _get_answering_jesus():
    from virtual_streamer.agents.answering_jesus_agent.agent import get_answering_jesus_agent
    return get_answering_jesus_agent()


@register_agent("atari_action_agent")
def _get_atari_action_agent():
    from virtual_streamer.agents.atari_action_agent.agent import get_atari_action_agent
    return get_atari_action_agent()


# Add more agents here as needed:
# @register_agent("story_generator")
# def _get_story_generator():
#     from virtual_streamer.agents.story_generator import get_story_generator
#     return get_story_generator()
