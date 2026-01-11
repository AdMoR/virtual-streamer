# Virtual Streamer

This is the codebase to run [AlloJesusChrist](https://www.twitch.tv/allojesuschrist)

## 🚀 Quick Start with the New Unified API

We've refactored the system into a modern, layered API architecture! 

**Get started in 5 minutes**: See [QUICKSTART_API.md](QUICKSTART_API.md)

### New Features:
- 🎯 **Layered Architecture**: Low-level entities, medium-level services, high-level applications
- 🌐 **REST API**: Everything accessible via HTTP endpoints
- 📊 **Streamlit UI**: User-friendly interface for video generation
- 🔄 **Async Jobs**: Background processing with progress tracking
- 🛠️ **Path Resolution**: Automatic path handling between services
- 📚 **Auto-generated Docs**: Interactive API documentation

### Quick Commands:
```bash
# Start the API server
./scripts/start_api.sh

# Start the Streamlit UI
./scripts/start_ui.sh

# View API docs
open http://localhost:8000/docs
```

For detailed information, see:
- [QUICKSTART_API.md](QUICKSTART_API.md) - Get started quickly
- [README_API.md](README_API.md) - Full API documentation
- [README_VIDEO_GENERATION.md](README_VIDEO_GENERATION.md) - Video generation details

---


## Query the api

```JSON
{
  "audio_path": "string",
  "video": {
    "storage_path": "string",
    "collection_ids": [
      "string"
    ],
    "clip_id": "string",
    "metadata": {
      "duration": 1,
      "scene_description_text": "string",
      "scene_keywords": [
        "string"
      ],
      "character_presences": [
        {
          "character_id": "string",
          "start_time": 0,
          "end_time": 1
        }
      ],
      "source_show_name": "string",
      "source_episode_name": "string",
      "start_time_in_source": 0,
      "end_time_in_source": 1
    },
    "created_at": "2025-04-21T14:12:17.730Z",
    "updated_at": "2025-04-21T14:12:17.730Z"
  },
  "options": {
    "subtitles_enabled": false,
    "subtitle_style": {
      "additionalProp1": {}
    }
  },
  "character_id": "string",
  "output_dir": "string"
}
```


# Running the code

## Basics of the worker

Most of the code is intended to be run with docker compose

```
cp .env.example .env
sudo docker compose --env-file .env up
```

## Different running modes 

Different behaviour activate based on the env variable.
For a local run, you should remove the var RMQ_URL which overrides the local rabbit config.

To do so, have a look in the .env.example that is the template to run the main compose file
```
AWS_ACCESS_KEY=XXXXXX
AWS_SECRET_KEY=YYYYYY
RMQ_URL=amqps://abcd:password@domain.rmq2.cloudamqp.com/abcd"
OPENAI_TOKEN=xxxxxxxx
```

TODO : 
- If not AWS token, stay in local production mode

## Streaming Infrastructure

The streaming module provides database-driven video scheduling and playback for OBS.

### Architecture Overview

```
Twitch Chat ──▶ Main API ──▶ MySQL (Playlists) ◀── Video Server ──▶ OBS
                    │                                    ▲
                    └───────▶ MinIO (Videos) ───────────┘
```

### Quick Start

```bash
# 1. Setup database tables for streaming
python scripts/setup_streaming_tables.py

# 2. Seed test data (optional)
python scripts/seed_streaming_data.py

# 3. Start main services
docker compose up -d

# 4. Setup shared network
./scripts/setup_streaming_network.sh

# 5. Start streaming stack (OBS + Video Server)
docker compose -f compose_streaming.yml up -d
```

### Key Concepts

- **StreamConfig**: Represents a streaming instance (e.g., a Twitch channel)
- **MediaProgrammation**: Time-based schedule linking to a StoryTemplate
- **PlaylistEntry**: A video in the playlist (pending → playing → played)

### Video Selection Logic

1. Get the active programmation based on current time
2. Return first pending video (by order, then creation time)
3. If no pending videos, randomly select from already-played videos (fallback)

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| Video Player | http://localhost:5000 | HTML5 player for OBS |
| OBS (VNC) | vnc://localhost:5901 | OBS control via VNC |
| OBS (noVNC) | http://localhost:6901 | OBS control via browser |
| API Docs | http://localhost:8000/docs | Full API documentation |

### API Endpoints

```bash
# Create a stream
curl -X POST http://localhost:8000/api/v1/streams \
  -H "Content-Type: application/json" \
  -d '{"stream_id": "my_stream", "name": "My Stream"}'

# Get next video for stream
curl http://localhost:8000/api/v1/streams/my_stream/next-video

# Add video to playlist
curl -X POST http://localhost:8000/api/v1/programmations/{id}/playlist \
  -H "Content-Type: application/json" \
  -d '{"video_storage_key": "generated_videos/video.mp4"}'
```

For detailed documentation, see [docs/design/streaming.md](docs/design/streaming.md).

### Legacy OBS (Deprecated)

The old RabbitMQ-based approach is deprecated:
```bash
# Old approach - DO NOT USE
sudo docker compose -f compose_obs.yml --env-file .env up
``` 

# Support code

## Getting training data for a voice model

Using the container from ...

The whisper french model must be transformed in to the CTranslate format

```commandline
ct2-transformers-converter --model BELLE-2/Belle-whisper-large-v3-zh --output_dir models/faster-whisper-large-v3 --copy_files tokenizer.json preprocessor_config.json
```


## Run the "compose up"

Required : 
- AWS creds stored under $HOME/.aws/credentials


First time 
- sudo docker compose up 

Rebuild from scratch
- docker builder prune
- sudo docker compose up --build --force-recreate --no-deps 

Testing a single container of the compose 
- sudo docker run -e OPENAI_TOKEN=xxx -p 7860:7860 --net=host cog-wav2lip-demo:latest


## Pushing the image to the Docker hub

```
#docker login -u amorvend
docker commit virtual-streamer-demo-1 virtual-streamer:demo
docker tag  virtual-streamer:demo amorvend/virtual-teacher-space:demo
docker image push  amorvend/virtual-teacher-space:demo
```


## Preprocessing a video dataset into chunks 

```
import os
dir_ = "/media/amor/data/Downloads/CPS/fred_voice"
files = [os.path.join(dir_, f) for f in os.listdir(dir_) if f.endswith("mp4")]
import subprocess
for f in files:
    rez = subprocess.run(["scenedetect", "-i", f, "split-video", "-o", "/media/amor/data/Downloads/CPS/fred_voice_clips"])

```

## Running fish audio container 

```
sudo docker run --gpus='all' -it --rm -p 7860:7860 fish-speech-fish-speech
```


## Feature for the video chunks 

florence_run

episode_creation_rag

https://huggingface.co/alibaba-pai/VideoCLIP-XL/tree/main/utils/vision_encoder


## Lip Sync 

The most reasonable option remains the Wav2Lip repo, because of : 

- widespread adoption
- variations available : faster, larger image size
- the version of this repo is hacked to be much cheaper

The main issue was the lack of handling for number_of_face != 1


Tested models : 

Very slow and no observable improvement 

```
sudo docker build -t lip-talker .

python /data/inf_demo.py --video_path /data/demo.mp4 --wav_path /data/audio.wav --ckpt_path /data/global_only.pth --avhubert_root /data/av_hubert/
```

## Intelligent subtitle file 

FFmpeg can do subtitle piece by piece with an intermediate file.

```
1
00:00:04,700 --> 00:00:05,090
You know what

2
00:00:05,100 --> 00:00:05,990
we should all do.
```

with the following command : 

`ffmpeg -i sample_video_ffmpeg.mp4 -vf subtitles=sample_video_subtitle_ffmpeg.srt output_srt.mp4`

"""
whisperx --model large-v2 --language de examples/sample_de_01.wav
"""

## Embedding description of video chunks 

ModernBert 

https://huggingface.co/lightonai/modernbert-embed-large




# Previous versions 


## Run the code

- Twitch reader : export SHARED_VOLUME="./" && python3 twitch_call.py && python3 chat_reader.py
- TTS service : python3 TTS/server/server.py --model_name tts_models/multilingual/multi-dataset/your_tts --use_cuda 1
- Worker : python3 inference.py --checkpoint_path ./checkpoints/Wav2Lip.pth
- Stream orchestrator : python3 obs_orchestrator.py
- RabbitMQ : sudo docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:latest
- sudo docker logs 41a123fabae5 --tail 150

