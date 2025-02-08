#!/usr/bin/env python
# coding: utf-8


#!pip install pip3-autoremove
#!pip-autoremove torch torchvision torchaudio -y
#!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
#!pip install --upgrade --no-cache-dir git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/trl.git 


import face_recognition
import os
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration, TextIteratorStreamer
import torch
import av
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
import cv2
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import pipeline
import json
import os
import subprocess
import argparse



model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
video_processor = LlavaOnevisionProcessor.from_pretrained(model_id)
video_model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
video_model.to("cuda")


def sample_frames(video_file, num_frames):
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def video_process_inference(my_video_path):
    container = av.open(my_video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)
    
    # For videos we have to feed a "video" type instead of "image"
    conversation = [
        {
    
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Explain the action and background of this video in a very detailled manner."},
                ],
        },
    ]
    
    prompt = video_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = video_processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda:0", torch.float16)
    out = video_model.generate(**inputs, max_new_tokens=100)
    rez = video_processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return rez[0].split("assistant")[1].strip()



fred1_image = face_recognition.load_image_file("/home/amor/Documents/code_dw/virtual-streamer/assets/fred_1.jpeg")
fred2_image = face_recognition.load_image_file("/home/amor/Documents/code_dw/virtual-streamer/assets/fred_2.jpeg")
fred3_image = face_recognition.load_image_file("/home/amor/Documents/code_dw/virtual-streamer/assets/fred_3.jpeg")

fred_embedding = [face_recognition.face_encodings(
    face_recognition.load_image_file(f"/home/amor/Documents/code_dw/virtual-streamer/assets/fred_{i}.jpeg"))[0] 
                  for i in range(1, 4)]

jamy_embedding = [face_recognition.face_encodings(
    face_recognition.load_image_file(f"/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_{i}.jpeg"))[0] 
                  for i in range(1, 4)]

jamy1_image = face_recognition.load_image_file("/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_1.jpeg")
jamy2_image = face_recognition.load_image_file("/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_2.jpeg")
jamy_1_image = face_recognition.load_image_file("/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_3.jpeg")



results = face_recognition.compare_faces(fred_embedding[1:] + jamy_embedding, fred_embedding[0])





def process(video_path, known_face_encodings, known_face_names):
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    index = 0
    resolution = 480
    speed_up_factor = 2
    results = list()

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    while True:
        # Grab a single frame of video
        index += 1
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if index % (fps * speed_up_factor) != 1:
            continue
        #print("face index frame", index)

        # This is the function block such that frame => result_dict
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        for i, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                results.append((name, index, face_locations[i]))

    return results


def build_personality_embedding(filepath):
    pers_dict: dict[str, list] = json.load(open(filepath))
    results = list()
    for k, v in pers_dict.items():
        results.append(([face_recognition.face_encodings(
            face_recognition.load_image_file(img_path)[0]) for img_path in v], [k] * len(v)))
    return [x[0] for x in results], [x[1] for x in results]


video_path = "/media/amor/data/Downloads/CPS/clips/Alerte au froid - C'est pas sorcier [Occ18RLg1XM]-Scene-085.mp4"

rez = process(video_path, fred_embedding + jamy_embedding, ["fred"] * 3 + ["jamy"] * 3)





torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="french", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch_dtype)
forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="french", task="transcribe")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    chunk_length_s=30,
    stride_length_s=(4, 2)
)


# In[13]:


filename = '/home/amor/Downloads/audio(1).wav'


# In[14]:



def trascribe(filename, outdir = "/media/amor/data/Downloads/CPS/clip_infos"):
    return asr_pipe(filename)["text"]

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return float(result.stdout)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='video_understanding',
        description='From a folder of video, create a description for each clip',
        epilog='Text at the bottom of help')
    parser.add_argument('-d', '--dirname', default="/media/amor/data/Downloads/CPS/clips")
    parser.add_argument('-o', '--outdirname', default="/media/amor/data/Downloads/CPS/clip_infos_test")
    parser.add_argument('-p', '--personality', default="./assets/personality_cps.json")
    args = parser.parse_args()

    too_short = list()
    too_short_file = "/media/amor/data/Downloads/CPS/too_short_clips.txt"

    if os.path.exists(too_short_file):
        with open(too_short_file, "r") as f:
            too_short = f.readlines()

    for i, f in enumerate(os.listdir(args.dirname)):
        # 0-a) exit because not a video
        if not f.endswith("mp4"):
            continue
        # 0-b) exit because of already known too short clip
        path = os.path.join(args.dirname, f)
        if path in too_short:
            continue
        duration = get_length(path)
        # 0-c) exit because of clip is computed as too small
        if duration < 6:
            too_short.append(path)
            with open(too_short_file, "a") as f:
                f.write(path + "\n")
            continue
        #  Already done
        name = os.path.basename(f).split(".")[0]
        filename_out = f"{args.outdirname}/{name}.json"
        if os.path.exists(filename_out):
            continue

        # Find faces
        people_emb, people_names = build_personality_embedding(args.personality)
        faces = process(path, fred_embedding + jamy_embedding, ["fred"] * 3 + ["jamy"] * 3)
        # Audio transcription
        mp3_path = os.path.join(args.outdirname, f.split(".")[0] + ".mp3")
        args = [
            "ffmpeg", "-y",  "-i",
            path, "-b:a",
            "192K", "-vn", mp3_path]
        rez = subprocess.run(args, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        transcription = trascribe(mp3_path, outdir=args.outdirname)
        # Image transcription
        visual_transcript = video_process_inference(path)

        data_dict = {
                "path": path,
                "who": list(faces),
                "transcription": transcription,
                "description": visual_transcript,
                "duration":  duration
            }
        # Save it
        json.dump(data_dict, open(filename_out, "w"))

