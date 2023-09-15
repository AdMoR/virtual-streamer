#!/bin/bash
#conda deactivate
cd /home/amor/Documents/code_dw/autovideo-maker/TTS
source venv/bin/activate
python3 TTS/server/server.py --model_name tts_models/multilingual/multi-dataset/your_tts --use_cuda 1