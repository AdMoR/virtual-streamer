from flask import Flask, request, jsonify
import torch
import os
import numpy as np
import cv2
import json
import time
import datetime
import subprocess
import shutil
from tqdm import tqdm
from virtual_streamer.workflows.character_setup import CHARACTERS
from virtual_streamer.wav2lip import audio
from virtual_streamer.wav2lip.main_logic import preprocess, Config, datagen, do_load, FaceDetectionGroup
from virtual_streamer.utils.utils import sanitize_str, txt_to_speech_call, combine_video_and_audio, add_subtitle, s3_upload, SubtitleMode
from virtual_streamer.workflows.prompts import PROMPT, PROMPT_FR, PROMPT_FR_3, PROMPT_FR_2, SARCASTIC_PROMPT_FR, \
    STAND_UP_PROMPT, SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT
from typing import Dict, Optional

app = Flask(__name__)

# Initialize global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
UPLOAD_BUCKET = os.environ.get("S3_BUCKET_URL", "default-bucket")
args = Config()
args.checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./checkpoints/Wav2Lip.pth")
print(f"Using checkpoint: {args.checkpoint_path}")
model, detector, detector_model = do_load(args.checkpoint_path, device)
mel_step_size = 16

# Initialize face detection groups
print('Reading video frames...')
face_detection_groups: Dict[str, FaceDetectionGroup] = dict()
for k, v in CHARACTERS.items():
    preprocess(args, v.video_clip_path, k, detector, face_detection_groups)


def wav2lip_exec(dirname, audio_path: str, question: str, det_results: FaceDetectionGroup):
    fps = 24

    full_frames = det_results.full_frames
    face_det_results = det_results.face_det_results

    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(question[:30])
    out_path = f'{dirname}/result.avi'
    batch_size = args.wav2lip_batch_size

    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        subprocess.check_call([
            "ffmpeg", "-y",
            "-i", audio_path,
            f"{dirname}/temp.wav",
        ])
        audio_path = 'temp/temp.wav'

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"Length of mel chunks: {len(mel_chunks)}, length of frames {len(full_frames)}")

    full_frames = full_frames[:len(mel_chunks)]
    face_det_results = face_det_results[:len(mel_chunks)]
    gen = datagen(args, full_frames.copy(), mel_chunks, face_det_results.copy())

    for i, rez in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        (face_img_batch, mel_batch, frames, coords) = rez

        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(f'{dirname}/result.avi',
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        face_img_batch = torch.FloatTensor(np.transpose(face_img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                pred = model(mel_batch, face_img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    return out_path


def process_video(question_data, gpt_response):
    dirname = os.environ.get("OUT_VIDEO_FOLDER", "./out_video_folder")
    os.makedirs(dirname, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)
    
    # Extract data from question
    question_text = question_data.get("question", "")
    character_name = question_data.get("character_name", "")
    subtitle_mode = question_data.get("subtitle_mode", "NONE")
    name = question_data.get("name", "User")
    
    # Get the audio for the response
    audio_outpath = txt_to_speech_call(gpt_response, "male-pt-3%0A", f"./temp/response_{hash(gpt_response) % 100000}.wav")
    
    # Get the face detection group for the character
    if character_name not in face_detection_groups:
        # Handle case where character is not precomputed
        if character_name in CHARACTERS:
            character = CHARACTERS[character_name]
            face_det_group = preprocess(args, character.video_clip_path, character_name, detector, None)
        else:
            # Default to first character if not found
            default_char = next(iter(face_detection_groups.values()))
            face_det_group = default_char
    else:
        face_det_group = face_detection_groups[character_name]
    
    # Wav2lip video generation
    s = time.time()
    outfile_path = wav2lip_exec(dirname, audio_outpath, question_text, face_det_group)
    print("wav2lip prediction time:", time.time() - s)
    
    # Recombination and add subtitles
    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(question_text[:30])
    outfile_combined_path = f'./temp/result_combined_{tag}.mp4'
    combine_video_and_audio(outfile_path, audio_outpath, outfile_combined_path)
    
    # Add subtitles if needed
    if subtitle_mode == "QUESTION":
        subtitle = f"Question de {name} : {question_text}"
        outfile_titled_path = f'./temp/result_titled_{tag}.mp4'
        add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
    elif subtitle_mode == "VOICE_SUBTITLE":
        subtitle = gpt_response
        outfile_titled_path = f'./temp/result_titled_{tag}.mp4'
        add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
    else:
        outfile_titled_path = outfile_combined_path
    
    # Move the file to final location
    final_outfile_path = os.path.join(dirname, f"result_{tag}.mp4")
    shutil.copyfile(outfile_titled_path, final_outfile_path)
    
    # Upload to S3 if needed
    s3_path = s3_upload(final_outfile_path, UPLOAD_BUCKET) if UPLOAD_BUCKET != "default-bucket" else final_outfile_path
    
    return {
        "video_path": final_outfile_path,
        "s3_path": s3_path,
        "response_text": gpt_response
    }


@app.route('/process', methods=['POST'])
def process():
    data = request.json
    
    # Extract data from request
    question_data = data.get("question", {})
    gpt_response = data.get("gpt_response", "")
    
    # Process the video
    result = process_video(question_data, gpt_response)
    
    return jsonify(result)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "device": device})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
