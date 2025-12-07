
JUDGE_PROMPT = """You are a contextual image rater. You grade if an image where a character is located and speaks a line of dialogue is contextual or not.

Examples:
- The character is in space. The character's line of dialogue speaks about nutrition => NOT_CONTEXTUAL
- The character is with a horse, the character speaks about horse races => CONTEXTUAL
- The character talks about gambling while facing the camera in a bar => NEUTRAL (vaguely related)

Please provide:
Rating: CONTEXTUAL/NEUTRAL/NOT_CONTEXTUAL/FAILURE
Grade: count the factors supporting one rating (for ranking, 0-10)
Reasoning: brief explanation

Next you will be provided a sentence and an image.
Judge if the image is well framed to host the dialog line.
Justify your reasoning by describing the image and how it does not fit the sentence.
Describe the main character, background and main colours of the scene.

If there is no image, use the FAILURE flag
 """


def format_judge_prompt(dialogue: str) -> str:
    """
    Format the judge prompt with the given dialogue.
    
    Args:
        dialogue: The dialogue line to match against the video
    
    Returns:
        Formatted prompt string
    """
    return JUDGE_PROMPT.format(dialogue=dialogue)
