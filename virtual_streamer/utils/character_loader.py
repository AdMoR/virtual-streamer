"""
Character loading utility.

Centralizes character loading logic used across multiple API modules.
"""

from virtual_streamer.video_server.models import Character, VoiceSample
from virtual_streamer.utils.entity_repository import get_entity_repository


async def load_character(character_id: str) -> Character:
    """
    Load a character from the repository.
    
    Args:
        character_id: The character identifier
        
    Returns:
        Character model with all fields populated
        
    Raises:
        ValueError: If character not found
    """
    repo = get_entity_repository()
    character_data = await repo.get_character(character_id)
    
    if character_data is None:
        raise ValueError(f"Character '{character_id}' not found")
    
    return Character(
        character_id=character_data["character_id"],
        name=character_data["name"],
        description=character_data.get("description"),
        video_clip_path=character_data.get("video_clip_path", ""),
        voice_samples=[
            VoiceSample(
                sample_storage_path=s["sample_storage_path"],
                transcript=s["transcript"],
            )
            for s in character_data.get("voice_samples", [])
        ],
        video_search_tag=character_data.get("video_search_tag"),
        identity_images=character_data.get("identity_images", []),
        created_at=character_data.get("created_at"),
        updated_at=character_data.get("updated_at"),
    )
