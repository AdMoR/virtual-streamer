import dataclasses
import enum
import os
from typing import Callable, Dict, Any
from collections import defaultdict
from character_setup import CHARACTERS
import datetime
import cv2
import torch
import json
from tqdm import tqdm
import audio
from batch_face import RetinaFace
import time
from textwrap import wrap
from utils import *
from prompts import PROMPT, PROMPT_FR, PROMPT_FR_3, PROMPT_FR_2, SARCASTIC_PROMPT_FR, \
    STAND_UP_PROMPT, SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT
from model_utils import face_detect, load_model
import subprocess
import numpy as np
import random
import shutil


@dataclasses.dataclass()
class Config:
    checkpoint_path: str = "./checkpoints/Wav2Lip.pth"
    resolution: int = 720
    wav2lip_batch_size: int = 8
    face_batch_size: int = 64
    img_size: int = 96

    static: bool = False
    pads: List[int] = (0, 10, 0, 0)
    nosmooth: bool = False
    rotate: bool = False
    box: List[int] = (-1, -1, -1, -1)
    crop: List[int] = (0, -1, 0, -1)
    resize_factor: float = 1.
    fps: int = 25
    outfile: str = 'results/result_voice.mp4'


def datagen(frames, mels, face_det_results):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
UPLOAD_BUCKET = os.environ["S3_BUCKET_URL"]


model = detector = detector_model = None


def do_load(checkpoint_path):
    global model, detector, detector_model

    model = load_model(checkpoint_path, device)

    # SFDDetector.load_model(device)
    detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")

    detector_model = detector.model

    print("Models loaded")


message_directory = "./"
args = Config()
do_load(args.checkpoint_path)

print('Reading video frames...')
full_frames_origin = dict()
face_det_results_origin = dict()


def preprocess(video_path: str, name: str, full_frames: Dict, face_det_results_origin: Dict):
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames[name] = list()

    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break

        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(args.resolution * aspect_ratio), args.resolution))
        # if args.resize_factor > 1:
        #     frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

        y1, y2, x1, x2 = [0, -1, 0, -1]
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames[name].append(frame)
    
    face_det_results_origin[name] = face_detect(detector, full_frames[name], args.face_batch_size)


for k, v in CHARACTERS.items():
    preprocess(v.video_clip_path, k, full_frames_origin, face_det_results_origin)


def wav2lip_exec(dirname, full_frames: List[Any], audio_path: str, question: str, face_det_results: Any):
    fps = 24

    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(question[:30])
    out_path = 'temp/result.avi'
    batch_size = args.wav2lip_batch_size

    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        # command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        # subprocess.call(command, shell=True)
        subprocess.check_call([
            "ffmpeg", "-y",
            "-i", audio_path,
            "temp/temp.wav",
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
    gen = datagen(full_frames.copy(), mel_chunks, face_det_results.copy())

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()
    return out_path


def main(args: Any, question: Question, full_frames: List[Any], face_det_results: Any):
    dirname = os.environ.get("OUT_VIDEO_FOLDER", "./out_video_folder")
    os.makedirs(dirname, exist_ok=True)

    # Step 1 - Get the response from GPT 3.5
    if question.prompt is None:
        question.prompt = random.choice([SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT])
    query = question.render()

    completion = gpt_call(query)
    text = completion
    json.dump({"query": query, "response": text, "question": question.question},
              open(f"prompts/response_{hash(query) % 1000000}.json", "w"))

    # Step 2 - Get the audio for the response
    # prev p317
    #audio_outpath = txt_to_speech_call(text, "male-pt-3%0A", f"./temp/response_{hash(query) % 100000}.wav")
    character = CHARACTERS[question.character_name]
    solero_language_switch(character.language, character.voice)
    os.makedirs("./temp", exist_ok=True)
    audio_outpath = txt_to_speech_call_solero(text, character.language,
                                              character.voice, f"./temp/response_{hash(query) % 100000}.wav")

    # Step 3 - Wav2lip video generation
    s = time.time()
    outfile_path = wav2lip_exec(dirname, full_frames, audio_outpath, question.question, face_det_results)
    print("wav2lip prediction time:", time.time() - s)

    # Step 4 - Recombination and add subtitles
    tag = str(datetime.datetime.now()).replace(" ", "-") + sanitize_str(question.question[:30])
    outfile_combined_path = f'./temp/result_combined_{tag}.mp4'
    combine_video_and_audio(outfile_path, audio_outpath, outfile_combined_path)
    # Opt 4 : add subtitles
    if question.subtitle_mode == SubtitleMode.QUESTION:
        subtitle = f"Question de {question.name} : {question.question}"
    elif question.subtitle_mode == SubtitleMode.VOICE_SUBTITLE:
        subtitle = text
    if question.subtitle_mode != SubtitleMode.NONE:
        outfile_titled_path = f'./temp/result_titled_{tag}.mp4'
        add_subtitle(subtitle, outfile_combined_path, outfile_titled_path)
    else:
        outfile_titled_path = outfile_combined_path
    
    # Step 5 - Move the file
    final_outfile_path = os.path.join(dirname, f"result_{tag}.mp4")
    shutil.copyfile(outfile_titled_path, final_outfile_path)
    print("---> ", final_outfile_path)
    return final_outfile_path, text


def callback(ch, method_frame, properties, body):
    # 1 - Retrieve some params of the processing job
    print("body", body)
    question = question_parser(body)
    character = CHARACTERS[question.character_name].name

    # 2 - main processing : video answer to the question is produced and stored on S3
    media_path, text = main(args, question, full_frames_origin[character], face_det_results_origin[character])
    print(f"New video_path : {media_path}")
    s3_path = s3_upload(media_path, UPLOAD_BUCKET)

    # 3 - A msg is prepared and send to the next user
    msg = VideoResponse(s3_path, text, question.question).serialize()
    routing_key = properties.reply_to or question.routing_queue
    if routing_key:
        print("Reply : ", routing_key)
        ch.basic_publish('', routing_key=routing_key, body=msg)
        ch.basic_ack(delivery_tag=method_frame.delivery_tag)


def main_exec(channel_name="chat_log"):
    channel = get_rmq_channel(channel_name)
    channel.basic_consume(queue=channel_name, on_message_callback=callback)
    print(f"Ready to read {channel_name}")
    channel.start_consuming()


if __name__ == '__main__':
    main_exec()
