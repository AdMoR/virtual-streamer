"""
Video Server - Pure API Proxy for OBS browser source.

This module provides a minimal FastAPI server that:
- Proxies requests to the main Virtual Streamer API
- Serves static HTML/JS for the video player
- Contains zero business logic (all decisions made by the main API)
"""
