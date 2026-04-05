"""
Atari Game Agent — sequential agent: planning (gated) → action.

Architecture:
  The planning sub-agent runs only on the first move of each session window
  (i.e., when PLANNING_NEEDED is True in session state).  After it runs, it
  clears the flag so subsequent moves in the same session skip planning and
  reuse the stored plan.

  The action sub-agent runs every move, injecting the current frame and the
  stored plan (if any) into the LLM request.

  The outer game loop resets the ADK session every PLAN_INTERVAL moves,
  which also resets PLANNING_NEEDED to True (via the initial session state),
  triggering re-planning at the start of each new window.

  PLAN_INTERVAL is configurable (default 24).
"""

import logging

from google.adk.agents import SequentialAgent

from virtual_streamer.agents.atari_action_agent.agent import AtariActionAgent
from virtual_streamer.agents.atari_planning_agent.agent import (
    AtariPlanningAgent,
    SESSION_KEY_PLANNING_NEEDED,
)

logger = logging.getLogger(__name__)

PLAN_INTERVAL = 24


class AtariGameAgent(SequentialAgent):
    """Sequential agent: planning (gated every PLAN_INTERVAL moves) → action."""

    def __init__(self, plan_interval: int = PLAN_INTERVAL):
        planning_agent = AtariPlanningAgent()
        action_agent = AtariActionAgent()
        super().__init__(
            name="atari_game_agent",
            sub_agents=[planning_agent, action_agent],
        )

    def set_frame(self, jpeg: bytes) -> None:
        """Update the game frame injected into both sub-agents."""
        self.sub_agents[0].set_frame(jpeg)  # AtariPlanningAgent
        self.sub_agents[1].set_frame(jpeg)  # AtariActionAgent


def get_atari_game_agent(plan_interval: int = PLAN_INTERVAL) -> AtariGameAgent:
    return AtariGameAgent(plan_interval=plan_interval)


root_agent = get_atari_game_agent()
