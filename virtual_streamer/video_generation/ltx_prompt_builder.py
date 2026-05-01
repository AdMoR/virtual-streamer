"""
LTX-2 Prompt Builder

Builds prompts for LTX-2 text-to-video generation from DialogLine objects.
Combines scene description with dialog text to create coherent video+audio prompts.
"""

from typing import Optional
from virtual_streamer.video_generation.config import DialogLine, StoryOutput


def build_ltx_prompt(
    dialog_line: DialogLine,
    include_dialog_audio: bool = True,
    style_prefix: Optional[str] = None,
    style_suffix: Optional[str] = None,
) -> str:
    """
    Build an LTX-2 prompt from a DialogLine.
    
    Combines the visual scene_description with the spoken dialog text
    to create a prompt that LTX-2 can use to generate synchronized
    video and audio.
    
    Args:
        dialog_line: DialogLine containing character_id, text, and scene_description
        include_dialog_audio: If True, includes the dialog text for audio generation
        style_prefix: Optional prefix to add styling instructions
        style_suffix: Optional suffix to add quality/style instructions
    
    Returns:
        Formatted prompt string for LTX-2
    
    Example:
        Input DialogLine:
            character_id: "fred"
            text: "Eh dis donc Jamy, tu sais comment marche l'IA?"
            scene_description: "A man in a lab coat speaks enthusiastically to camera"
        
        Output:
            "A man in a lab coat speaks enthusiastically to camera. 
             The character speaks clearly: 'Eh dis donc Jamy, tu sais comment marche l'IA?'
             Cinematic quality, smooth motion."
    """
    parts = []
    
    # Add style prefix if provided
    if style_prefix:
        parts.append(style_prefix)
    
    # Add scene description (visual component)
    scene_desc = dialog_line.scene_description.strip()
    if not scene_desc.endswith('.'):
        scene_desc += '.'
    parts.append(scene_desc)
    
    # Add dialog text for audio generation
    if include_dialog_audio and dialog_line.text:
        dialog_text = dialog_line.text.strip()
        # Format the dialog instruction for LTX-2 to generate speech
        parts.append(f"The character speaks clearly: \"{dialog_text}\"")
    
    # Add style suffix if provided (quality/motion instructions)
    if style_suffix:
        parts.append(style_suffix)
    
    return " ".join(parts)


def build_ltx_prompt_detailed(
    dialog_line: DialogLine,
    character_name: Optional[str] = None,
    emotion: Optional[str] = None,
    camera_angle: Optional[str] = None,
) -> str:
    """
    Build a more detailed LTX-2 prompt with additional parameters.
    
    Args:
        dialog_line: DialogLine containing character_id, text, and scene_description
        character_name: Optional character name to include
        emotion: Optional emotion descriptor (e.g., "enthusiastic", "skeptical")
        camera_angle: Optional camera angle (e.g., "close-up", "medium shot")
    
    Returns:
        Detailed prompt string for LTX-2
    """
    parts = []
    
    # Camera angle if specified
    if camera_angle:
        parts.append(f"{camera_angle}:")
    
    # Scene description
    scene_desc = dialog_line.scene_description.strip()
    if not scene_desc.endswith('.'):
        scene_desc += '.'
    parts.append(scene_desc)
    
    # Character name and emotion
    if character_name or emotion:
        char_desc = character_name or "The character"
        if emotion:
            char_desc += f", looking {emotion},"
        parts.append(f"{char_desc} speaks to camera.")
    
    # Dialog with speech instruction
    if dialog_line.text:
        dialog_text = dialog_line.text.strip()
        parts.append(f"They say: \"{dialog_text}\"")
    
    # Quality suffix
    parts.append("High quality, natural motion, synchronized audio.")
    
    return " ".join(parts)


def build_negative_prompt() -> str:
    """
    Build a standard negative prompt for LTX-2 video generation.
    
    Returns:
        Negative prompt string to avoid common artifacts
    """
    return (
        "blurry, low quality, still frame, static, frozen, "
        "watermark, overlay, titles, subtitles, text on screen, "
        "distorted face, deformed hands, isolated hands, artifacts, glitches, "
        "inconsistent lighting"
    )


def build_prompts_from_story(
    story_output: StoryOutput,
    style_prefix: Optional[str] = None,
    style_suffix: Optional[str] = "Cinematic quality, smooth motion, natural lighting.",
    include_dialog_audio: bool = True,
) -> list[dict]:
    """
    Build LTX-2 prompts for all DialogLines in a StoryOutput.
    
    Args:
        story_output: StoryOutput containing title, story_plan, and dialog lines
        style_prefix: Optional prefix for all prompts
        style_suffix: Optional suffix for all prompts
        include_dialog_audio: Whether to include dialog text for audio
    
    Returns:
        List of dicts with 'prompt', 'negative_prompt', and 'dialog_line' for each line
    """
    prompts = []
    negative = build_negative_prompt()
    
    for dialog_line in story_output.dialog:
        prompt = build_ltx_prompt(
            dialog_line=dialog_line,
            include_dialog_audio=include_dialog_audio,
            style_prefix=style_prefix,
            style_suffix=style_suffix,
        )
        prompts.append({
            "prompt": prompt,
            "negative_prompt": negative,
            "dialog_line": dialog_line,
            "character_id": dialog_line.character_id,
        })
    
    return prompts
