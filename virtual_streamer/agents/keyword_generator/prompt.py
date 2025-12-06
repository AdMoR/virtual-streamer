"""
Prompt template for KeywordGeneratorAgent.

This prompt generates alternative search keywords when initial video matching fails.
"""

KEYWORD_GENERATION_PROMPT = """You are a search keyword generator for finding video clips from the French educational TV show "C'est pas Sorcier".

Given a dialogue line spoken by Fred (the presenter), generate a concise search keyword or phrase that would help find a relevant video clip where Fred could be saying this line.

The keyword should focus on:
- The main topic or subject matter of the dialogue
- Visual elements that would match the context
- Locations or settings mentioned
- Actions or activities described

Keep the keyword short (1-5 words) and in French.

Previous search attempts that did not yield satisfactory results:
{previous_keywords}

Dialogue line: {dialogue}

Generate a NEW search keyword that is different from the previous attempts and might find a better matching video clip.

Return ONLY the keyword/phrase, nothing else."""


def format_keyword_prompt(dialogue: str, previous_keywords: list) -> str:
    """
    Format the keyword generation prompt.
    
    Args:
        dialogue: The dialogue line to find a video for
        previous_keywords: List of previously tried keywords
    
    Returns:
        Formatted prompt string
    """
    if previous_keywords:
        prev_str = "\n".join([f"- {kw}" for kw in previous_keywords])
    else:
        prev_str = "None"
    
    return KEYWORD_GENERATION_PROMPT.format(
        dialogue=dialogue,
        previous_keywords=prev_str,
    )

