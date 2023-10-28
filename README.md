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
-  sudo docker logs 41a123fabae5 --tail 150
