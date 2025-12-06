"""
Sentence Processor Agent.

Custom agent that orchestrates sentence-level video generation:
- Loops over sentences
- Creates parallel VideoMatcherAgents
- Runs TTS/STT/video combining
"""

from virtual_streamer.agents.sentence_processor.agent import SentenceProcessorAgent

__all__ = ["SentenceProcessorAgent"]

