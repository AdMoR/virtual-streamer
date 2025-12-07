
JUDGE_PROMPT = """You are a contextual image rater. You grade if an image where a character is located and speaks a line of dialogue is contextual or not.

Examples:
- The character is in space. The character's line of dialogue speaks about nutrition => NOT_CONTEXTUAL
- The character is with a horse, the character speaks about horse races => CONTEXTUAL
- The character talks about gambling while facing the camera in a bar => NEUTRAL (vaguely related)

Please provide:
Rating: CONTEXTUAL/NEUTRAL/NOT_CONTEXTUAL
Grade: count the factors supporting one rating (for ranking, 0-10)
Reasoning: brief explanation

Dialogue line: {dialogue}"""


def format_judge_prompt(dialogue: str) -> str:
    """
    Format the judge prompt with the given dialogue.
    
    Args:
        dialogue: The dialogue line to match against the video
    
    Returns:
        Formatted prompt string
    """
    return JUDGE_PROMPT.format(dialogue=dialogue)
