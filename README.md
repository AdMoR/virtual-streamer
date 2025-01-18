# Virtual Streamer

This is the codebase to run [AlloJesusChrist](https://www.twitch.tv/allojesuschrist)


## Setup of the environnement

Using conda 

```
conda create -p ./venv python=3.9
conda activate ./venv
pip install -r requirements.txt
```

The code was tested with cuda 11.8


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

## Side applications 

**OBS streaming** 

It launches an OBS in a container. On the side there is a video server telling what to stream based on a rabbit queue.
```
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



# Previous versions 


## Run the code

- Twitch reader : export SHARED_VOLUME="./" && python3 twitch_call.py && python3 chat_reader.py
- TTS service : python3 TTS/server/server.py --model_name tts_models/multilingual/multi-dataset/your_tts --use_cuda 1
- Worker : python3 inference.py --checkpoint_path ./checkpoints/Wav2Lip.pth
- Stream orchestrator : python3 obs_orchestrator.py
- RabbitMQ : sudo docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:latest
- sudo docker logs 41a123fabae5 --tail 150

