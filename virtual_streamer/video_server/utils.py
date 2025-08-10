import os
import httpx
from virtual_streamer.video_server.models import Character


ENTITY_SERVICE_HOST = os.environ.get("ENTITY_SERVICE_HOST", "0.0.0.0").rstrip('/') # Ensure no trailing slas


async def get_character_data(character_id: str) -> Character:
    """Helper function to fetch character data from the entity service."""
    character_url = f"http://{ENTITY_SERVICE_HOST}:8002/characters/{character_id}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(character_url)
        response = await client.get(character_url)
        response.raise_for_status() # Raise exception for 4xx/5xx responses
        result_dict = response.json()
        character = Character.model_validate(result_dict)
        return character


def get_character_data_sync(character_id: str) -> Character:
    """Helper function to fetch character data from the entity service."""
    character_url = f"http://{ENTITY_SERVICE_HOST}:8002/characters/{character_id}"
    with httpx.Client(timeout=30.0) as client:
        print(character_url)
        response = client.get(character_url)
        response.raise_for_status() # Raise exception for 4xx/5xx responses
        result_dict = response.json()
        character = Character.model_validate(result_dict)
        return character


async def get_character_data(character_id: str) -> Character:
    """Helper function to fetch character data from the entity service."""
    character_url = f"http://{ENTITY_SERVICE_HOST}:8002/characters"
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(character_url)
        response = await client.get(character_url)
        response.raise_for_status() # Raise exception for 4xx/5xx responses
        result_dict = response.json()
        character = Character.model_validate(k)
        return character