FROM mountaintopsolutions/docker-obs-ndi-ng:legacy-latest

RUN mkdir -p /config/obs-studio
COPY ./assets/obs_config/test.html /config/
COPY ./assets/obs_config/obs-studio /config/obs-studio
RUN mv /config/obs-studio/basic/profiles/jesus /config/obs-studio/basic/profiles/Untitled && \
    mv /config/obs-studio/basic/scenes/jesus_browser.json /config/obs-studio/basic/scenes/Untitled.json
