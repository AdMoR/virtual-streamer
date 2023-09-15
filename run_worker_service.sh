#!/bin/bash
sleep 2 && export OPENAI_TOKEN="xxx" && source /media/amor/Storage/code_dw/cog-Wav2Lip/venv/bin/activate && python3 inference.py --checkpoint_path ./checkpoints/Wav2Lip.pth