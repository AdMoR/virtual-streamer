"""Entry point: python -m virtual_streamer.mcp_agent"""

import asyncio
import logging

from virtual_streamer.mcp_agent.config import AgentConfig
from virtual_streamer.mcp_agent.agent import MCPAgentLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    config = AgentConfig()
    loop = MCPAgentLoop(config)
    asyncio.run(loop.run())


main()
