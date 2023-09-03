from typing import List, Dict, Optional, Callable, Any

import shutil
import urllib

import random
import os

import openai

import warnings
import requests



openai.api_key = os.environ["OPENAI_TOKEN"]


import os
from textwrap import wrap
import subprocess
import numpy as np
import imageio
import tempfile
from PIL import Image
import dataclasses
import openai
import os


def create_video_from_image(image_path, output_path, duration):
    args = [
        "ffmpeg", "-y",  "-i",
        image_path, "-loop",
        "1", "-t", str(duration),  output_path]
    rez = subprocess.run(args)
    if rez.returncode != 0:
        raise Exception(rez)
    return output_path


def add_subtitle(subtitle, video_path, output_path, min_duration=None) -> str:
    # 2 - Add the title
    for char in ":?/!":
        subtitle = subtitle.replace(char, "\\" + char)
    subtitle = subtitle.replace("'", "").replace('"', "")
    subtitle = "\n".join(wrap(subtitle, 50))
    args = ['ffmpeg',
            "-y", '-i',
            video_path,
            '-filter_complex',
            f"drawtext=fontfile=/home/amor/Downloads/croissant-one/CroissantOne-Regular.ttf:text='{subtitle}':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=7*(h-text_h)/8:box=1:boxcolor=black@0.5:boxborderw=1",
             '-c:v', 'h264_nvenc', "-qp", "32", "-b:v", "3000k", "-minrate", "3000k", "-maxrate", "10000k",
            '-codec:a', 'copy',
            output_path]
    if min_duration is not None:
        supp_args = ["-loop", "1", "-t", str(min_duration), ]
        for i, t in zip(range(2, 2 + len(supp_args)), supp_args):
            args.insert(i, t)

    print("------> ", args)
    rez = subprocess.run(args)
    if rez.returncode != 0:
        raise Exception(rez)
    return output_path


def build_concat_file(path_list, concat_file_path=None, duration_per_block=None):
    if concat_file_path is None:
        concat_file_path = tempfile.mktemp()
    with open(concat_file_path, "w") as f:
        for path in map(lambda x: os.path.abspath(x), path_list):
            if duration_per_block is None:
                text = f"file '{path}'\n"
            else:
                text = f"file '{path}'\nduration {duration_per_block}\n"
            f.write(text)
    return concat_file_path

def combine_part_in_concat_file(video_path_list, concat_file_path, out_path, duration_per_block=None):
    concat_file_path = build_concat_file(video_path_list, concat_file_path, duration_per_block)
    rez = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file_path, '-c:v', 'h264_nvenc', "-qp", "32", "-b:v", "3000k", "-minrate", "3000k", "-maxrate", "10000k",
                          "-c", "copy", out_path])
    if rez.returncode == 1:
        raise Exception("ffmpeg concat failed")
    return out_path


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def txt_to_speech_call(speech_lines, speaker, outpath):
    safe_string = urllib.parse.quote_plus(speech_lines)
    url = f"http://localhost:5002/api/tts?text={safe_string}&speaker_id={speaker}&style_wav=&language_id=fr-fr"

    rez = subprocess.run(["curl", "-L", "-X", "GET", url, "--output", outpath])
    if rez.returncode == 1:
        raise Exception("Txt to speech call failed")

    return outpath


@dataclasses.dataclass
class Question:
    name: str
    question: str


def question_parser(line: bytes) -> Question:
    tokens = line.decode("utf-8").split("|")
    return Question(tokens[0], tokens[1])


def read_from_question_queue(fileptah: str ="./chat_log.txt",
                             parser: Callable[[bytes], Any] = question_parser) -> Optional[Any]:
    offset = 0
    while True:
        infile = open(fileptah, "rb")
        infile.seek(offset)
        for line in infile:
            yield parser(line)
        infile.close()
        with open(fileptah, "wb") as f:
            f.write(b"")
        yield None
        print(offset)
