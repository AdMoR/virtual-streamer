from typing import Callable, Dict, Any
import argparse
import math
import os
import platform
import subprocess
import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

import audio
# from face_detect import face_rect
from models import Wav2Lip

from batch_face import RetinaFace
import time
from textwrap import wrap
from utils import *
from prompts import PROMPT, PROMPT_FR, PROMPT_FR_3, PROMPT_FR_2, SARCASTIC_PROMPT_FR

import subprocess
import numpy as np
import imageio
import tempfile
from PIL import Image
import dataclasses
import openai


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=8)

# parser.add_argument('--resize_factor', default=1, type=int,
#             help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--out_height', default=1080, type=int,
            help='Output video height. Best results are obtained at 480 or 720')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    results = []
    pady1, pady2, padx1, padx2 = [0, 10, 0, 0]

    s = time.time()

    for image, rect in zip(images, face_rect(images)):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    print('face detect time:', time.time() - s)

    boxes = np.array(results)
    #if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results


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

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()



def face_rect(images):
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size: (i + 1) * face_batch_size]
        all_faces = detector(batch)  # return faces list of all images
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret


model = detector = detector_model = None

def do_load(checkpoint_path):
    global model, detector, detector_model

    model = load_model(checkpoint_path)

    # SFDDetector.load_model(device)
    detector = RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")
    # detector = RetinaFace(gpu_id=0, model_path="checkpoints/resnet50.pth", network="resnet50")

    detector_model = detector.model

    print("Models loaded")

message_directory = "./"
args = parser.parse_args()
args.img_size = 96
do_load(args.checkpoint_path)

face_batch_size = 8 * 8
video_stream = cv2.VideoCapture("/media/amor/Storage/code_dw/cog-Wav2Lip/reference_videos/reference.mp4")
fps = video_stream.get(cv2.CAP_PROP_FPS)

print('Reading video frames...')

full_frames = []
while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break

    aspect_ratio = frame.shape[1] / frame.shape[0]
    frame = cv2.resize(frame, (int(1080 * aspect_ratio), 1080))
    # if args.resize_factor > 1:
    #     frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

    y1, y2, x1, x2 = [0, -1, 0, -1]
    if x2 == -1: x2 = frame.shape[1]
    if y2 == -1: y2 = frame.shape[0]

    frame = frame[y1:y2, x1:x2]

    full_frames.append(frame)
    
face_det_results_origin = face_detect(full_frames) 


def main(args, name, question,audio_path, full_frames, face_det_results_origin):
    dirname = "/media/amor/Storage/Videos/JesusStreamFolder"
    tag=str(datetime.datetime.now()).replace(" ", "-")
    out_path = os.path.join(dirname, f"result_{tag}.mp4")
    batch_size = args.wav2lip_batch_size
    #print ("Number of frames available for inference: "+str(len(full_frames)))

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
    print(mel.shape)

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

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]
    face_det_results = face_det_results_origin[:len(mel_chunks)]


    gen = datagen(full_frames.copy(), mel_chunks, face_det_results.copy())

    s = time.time()

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

    print("wav2lip prediction time:", time.time() - s)
    
    outfile_path = f'./temp/result_{tag}.mp4'
    outfile_titled_path = f'./temp/result_titled_{tag}.mp4'
    #combine_part_in_concat_file([title_path, outfile_path], None, outfile_titled_path)


    subprocess.check_call([
        "ffmpeg", "-y",
        # "-vsync", "0", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", "temp/result.avi",
        "-i", audio_path, 
        "-c:v", "h264_nvenc",
        outfile_path,
    ])
    subtitle = f"Question de {name} : {question}"
    add_subtitle(subtitle, outfile_path, outfile_titled_path)
    
    # Move the file
    final_outfile_path = os.path.join(dirname, f"result_{tag}.mp4")
    shutil.copyfile(outfile_titled_path, final_outfile_path)
    
    return final_outfile_path


def gpt_call_mock(prompt):
    return """
        Bonjour Dany, merci pour cette demande. Les hosties symbolisent mon corps donné pour vous, et leur préparation requiert du pain sans levain. Elles sont un rappel du dernier repas que j'ai partagé avec mes disciples avant ma crucifixion, où j'ai offert le pain comme mon corps sacrifié.
Dans la Bible, lors de la Cène, je partageais le pain avec mes disciples en disant : "Prenez, mangez, ceci est mon corps" (Matthieu 26:26). Cette action symbolique représente l'unité entre les croyants et moi-même, ainsi que le sacrifice ultime que j'ai fait pour l'humanité.
        """


def gpt_call(prompt):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0.3, 
                                              messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content



def fn(args, name, question):
    template = random.choice([PROMPT_FR_2, PROMPT_FR_3, SARCASTIC_PROMPT_FR])
    query = template.format(name=name, question=question)

    completion = gpt_call(query)
    text = completion
    print("response ===> ", text)

    # prev p317
    audio_outpath = txt_to_speech_call(text, "male-pt-3%0A", f"./temp/response_{hash(query) % 100000}.wav")
    # Create response
    return main(args, name, question, audio_outpath, full_frames, face_det_results_origin)


def update_next_qestion_file(question: Question):
    with open("./next_qestion.txt", "wb") as f:
        if question is not None:
            f.write(f"Question de {question.name} : {question.question[:500]}".encode("utf-8"))
        else:
            f.write(f"Pas de question en cours".encode("utf-8"))


def main_exec():
    for question in read_from_queue("chat_log", question_parser):
        update_next_qestion_file(question)
        if question is not None:
            #question = Question(random.choice(default_names), random.choice(default_questions))
            print("-->", question)
            video_path = fn(args, question.name, question.question)
            print(f"New video_path : {video_path}")
            #update_obs_source(video_path)
            add_to_queue("video_response_queue", video_path)
        time.sleep(1)
        print("sleeping")


if __name__ == '__main__':
    main_exec()
