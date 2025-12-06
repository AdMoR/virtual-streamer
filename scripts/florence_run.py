#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pip3-autoremove
#!pip-autoremove torch torchvision torchaudio -y
#!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
#!pip install --upgrade --no-cache-dir git+https://github.com/huggingface/transformers.git git+https://github.com/huggingface/trl.git
# get_ipython().system('pip install timm')
# get_ipython().system('pip install face_recognition')


# In[2]:


import sys

sys.executable
# get_ipython().system('python --version')


# In[3]:


import os


from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import torch
# get_ipython().run_line_magic('matplotlib', 'inline')


model_id = "microsoft/Florence-2-large"
flo_model = (
    AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype="auto"
    )
    .eval()
    .cuda()
)
flo_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# In[4]:


import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def run_example(task_prompt, image_, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = flo_processor(text=prompt, images=image_, return_tensors="pt").to(
        "cuda", torch.float16
    )
    generated_ids = flo_model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = flo_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed_answer = flo_processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image_.shape[1], image_.shape[0])
    )

    return parsed_answer


# In[ ]:


# In[ ]:


# In[5]:


import cv2


def florence_process(my_video_path, task_prompt):
    video_stream = cv2.VideoCapture(my_video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    index = 0
    resolution = 480
    speed_up_factor = 5
    results = list()
    while True:
        index += 1
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if index % (fps * speed_up_factor) != 0 and len(results) > 0:
            continue
        # print("Florence index frame : ", index)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(resolution * aspect_ratio), resolution))
        result = run_example(task_prompt, frame)
        results.append(result)

    return results


# In[6]:


video_path = "/media/amor/data/Downloads/CPS/clips/Alerte au froid - C'est pas sorcier [Occ18RLg1XM]-Scene-085.mp4"
rez = florence_process(video_path, "<MORE_DETAILED_CAPTION>")

rez


# In[7]:


video_path = "/media/amor/data/Downloads/CPS/clips/Qu'est-ce qu'un lien hypertexte ？ - C'est pas sorcier [PWnWeG1uiMw]-Scene-007.mp4"
rez = florence_process(video_path, "<MORE_DETAILED_CAPTION>")

rez


# In[8]:


import face_recognition

fred1_image = face_recognition.load_image_file(
    "/home/amor/Documents/code_dw/virtual-streamer/assets/fred_1.jpeg"
)
fred2_image = face_recognition.load_image_file(
    "/home/amor/Documents/code_dw/virtual-streamer/assets/fred_2.jpeg"
)
fred3_image = face_recognition.load_image_file(
    "/home/amor/Documents/code_dw/virtual-streamer/assets/fred_3.jpeg"
)

fred_embedding = [
    face_recognition.face_encodings(
        face_recognition.load_image_file(
            f"/home/amor/Documents/code_dw/virtual-streamer/assets/fred_{i}.jpeg"
        )
    )[0]
    for i in range(1, 4)
]

jamy_embedding = [
    face_recognition.face_encodings(
        face_recognition.load_image_file(
            f"/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_{i}.jpeg"
        )
    )[0]
    for i in range(1, 4)
]

jamy1_image = face_recognition.load_image_file(
    "/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_1.jpeg"
)
jamy2_image = face_recognition.load_image_file(
    "/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_2.jpeg"
)
jamy_1_image = face_recognition.load_image_file(
    "/home/amor/Documents/code_dw/virtual-streamer/assets/jamy_3.jpeg"
)


results = face_recognition.compare_faces(
    fred_embedding[1:] + jamy_embedding, fred_embedding[0]
)


# In[9]:


results


# In[10]:


import numpy as np
import cv2


def process(video_path, known_face_encodings, known_face_names):
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    index = 0
    resolution = 480
    speed_up_factor = 4
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
        # print("face index frame", index)

        # This is the function block such that frame => result_dict
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            name = "Unknown"
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                results.append((name, index))

    return results


# In[11]:


video_path = "/media/amor/data/Downloads/CPS/clips/Alerte au froid - C'est pas sorcier [Occ18RLg1XM]-Scene-085.mp4"

process(video_path, fred_embedding + jamy_embedding, ["fred"] * 3 + ["jamy"] * 3)


# In[ ]:


# In[ ]:


# In[12]:


import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import pipeline


torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-large-v3", language="french", task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch_dtype
)
forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
    language="french", task="transcribe"
)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    chunk_length_s=30,
    stride_length_s=(4, 2),
)


# In[13]:


filename = "/home/amor/Downloads/audio(1).wav"


# In[14]:


import os
import subprocess


def trascribe(filename, outdir="/media/amor/data/Downloads/CPS/clip_infos"):
    return asr_pipe(filename)["text"]


def get_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return float(result.stdout)


print(get_length(filename))
trascribe(filename)


# In[ ]:


import subprocess
import json
import tqdm

dirname = "/media/amor/data/Downloads/CPS/clips"
outdirname = "/media/amor/data/Downloads/CPS/clip_infos"
too_short = list()


for i, f in tqdm.tqdm(enumerate(os.listdir(dirname))):
    if not f.endswith("mp4"):
        continue
    path = os.path.join(dirname, f)
    duration = get_length(path)
    if duration < 6:
        too_short.append(path)
        continue
    #  Already done
    name = os.path.basename(f).split(".")[0]
    filename_out = f"{outdirname}/{name}.json"
    if os.path.exists(filename_out):
        continue

    # Find faces
    faces = process(path, fred_embedding + jamy_embedding, ["fred"] * 3 + ["jamy"] * 3)
    # Audio transcription
    mp3_path = os.path.join(outdirname, f.split(".")[0] + ".mp3")
    args = ["ffmpeg", "-y", "-i", path, "-b:a", "192K", "-vn", mp3_path]
    rez = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    transcription = trascribe(mp3_path, outdir=outdirname)
    # Image transcription
    visual_transcript = florence_process(path, "<MORE_DETAILED_CAPTION>")

    data_dict = {
        "path": path,
        "who": list(faces),
        "transcription": transcription,
        "description": visual_transcript,
        "duration": duration,
    }
    # Save it
    json.dump(data_dict, open(filename_out, "w"))
