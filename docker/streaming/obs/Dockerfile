# OBS Streaming Container
# Based on OBS-NDI for streaming with browser source

FROM mountaintopsolutions/docker-obs-ndi-ng:legacy-latest

# Create config directory
RUN mkdir -p /config/obs-studio

# Copy OBS configuration
# Note: Scene configs can be customized per stream by mounting different configs
COPY ./assets/obs_config/obs-studio /config/obs-studio

# Rename profile and scene to generic names
RUN mv /config/obs-studio/basic/profiles/jesus /config/obs-studio/basic/profiles/Untitled 2>/dev/null || true && \
    mv /config/obs-studio/basic/scenes/jesus_browser.json /config/obs-studio/basic/scenes/Untitled.json 2>/dev/null || true

# The browser source URL should be updated to point to the video_server container
# This is typically done via environment variables or by mounting a custom scene config
