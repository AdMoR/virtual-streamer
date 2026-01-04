"""
DEPRECATED: This module is deprecated.

All character data access should now use:
- Inside API: EntityRepository from virtual_streamer.utils.entity_repository
- Outside API: CharacterClient from virtual_streamer.api.clients.character_client

Example usage for external code:
    from virtual_streamer.api.clients.character_client import CharacterClient

    async with CharacterClient() as client:
        character = await client.get_character("fred")
        characters = await client.list_characters()

For synchronous code:
    import asyncio
    from virtual_streamer.api.clients.character_client import CharacterClient

    async def _fetch():
        async with CharacterClient() as client:
            return await client.get_character("fred")
    
    character = asyncio.run(_fetch())
"""

# Re-export from new location for backward compatibility
# These imports will fail if the new client is used, serving as a reminder to migrate
from virtual_streamer.api.clients.character_client import (
    CharacterClient,
    get_character as _get_character_async,
    list_characters as _list_characters_async,
)

import warnings


def _emit_deprecation_warning(func_name: str):
    warnings.warn(
        f"{func_name} is deprecated. Use CharacterClient from "
        "virtual_streamer.api.clients.character_client instead.",
        DeprecationWarning,
        stacklevel=3,
    )
