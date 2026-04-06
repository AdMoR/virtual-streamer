FROM python:3.9-slim

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./virtual_streamer ./virtual_streamer
COPY ./setup.py .
COPY ./configs ./configs
RUN pip install -e .

# Expose the port the video server runs on
EXPOSE 5000

# Run the streaming video server using Uvicorn
CMD ["uvicorn", "virtual_streamer.streaming.video_server.app:app", "--host", "0.0.0.0", "--port", "5000"]