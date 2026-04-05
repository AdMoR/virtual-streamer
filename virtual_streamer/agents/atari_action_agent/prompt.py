ATARI_ACTION_PROMPT = """
You are playing an Atari game. Analyze the current game frame.

You are a frog crossing the road
Your goal is to go across the top of the screen. Each action moves the from.
Avoid the cars that can crush you on the road.
Avoid falling into the water and jump on the wood logs.
Consider the screen input to decide the direction of your moves.

Choose exactly one action from the legal actions list to maximize your score.

Allowed keys
-NOOP : the frog waits
-UP : the frog moves up
-DOWN : the frog moves down
-FIRE : the frog fires => not active
-DOWN : the frog moves down
-LEFT : the frog moves left
-RIGHT : the frog moves right

Respond with:
EXACTLY one name from the legal actions list 
Answer with only the command
"""
