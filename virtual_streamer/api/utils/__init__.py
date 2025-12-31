"""
API Utilities.

Utilities for the Virtual Streamer API including mountable apps support.
"""

from virtual_streamer.api.utils.mount_app import (
    MountableApp,
    mount_app,
    check_route_conflicts,
)

__all__ = [
    "MountableApp",
    "mount_app",
    "check_route_conflicts",
]

