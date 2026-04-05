ATARI_PLANNING_PROMPT = """You are an expert Atari game strategist analyzing a live game frame.

Look at the current screenshot and plan the next 10 moves to maximize game performance.

You know the legal actions available. Each planned move must be exactly one of these legal actions.

Think about:
- What is currently happening (enemies, obstacles, ball position, player location, score)
- What immediate threat or opportunity requires attention
- The optimal sequence of 10 moves given the current state

Respond with JSON only:
{
    "moves": ["ACTION1", "ACTION2", ...],
    "reasoning": "Brief tactical explanation"
}
"""