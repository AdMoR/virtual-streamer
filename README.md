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


## Run the code

- Twitch reader : export SHARED_VOLUME="./" && python3 twitch_call.py && python3 chat_reader.py
- TTS service : python3 TTS/server/server.py --model_name tts_models/multilingual/multi-dataset/your_tts --use_cuda 1
- Worker : python3 inference.py --checkpoint_path ./checkpoints/Wav2Lip.pth
- Stream orchestrator : python3 obs_orchestrator.py
- RabbitMQ : sudo docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:latest
- sudo docker logs 41a123fabae5 --tail 150


# Run the "compose up"

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