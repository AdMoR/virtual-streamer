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



# Previous versions 


## Run the code

- Twitch reader : export SHARED_VOLUME="./" && python3 twitch_call.py && python3 chat_reader.py
- TTS service : python3 TTS/server/server.py --model_name tts_models/multilingual/multi-dataset/your_tts --use_cuda 1
- Worker : python3 inference.py --checkpoint_path ./checkpoints/Wav2Lip.pth
- Stream orchestrator : python3 obs_orchestrator.py
- RabbitMQ : sudo docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:latest
- sudo docker logs 41a123fabae5 --tail 150


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