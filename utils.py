from typing import List, Dict, Optional, Callable, Any
import re
import num2words
import shutil
import urllib
import random
import os
import openai
import warnings
import requests
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


openai.api_key = os.environ["OPENAI_TOKEN"]
queue_directory = "./"


def combine_video_and_audio(vfile_path, afile_path, out_path):
    return subprocess.check_call([
        "ffmpeg", "-y",
        "-i", vfile_path,
        "-i", afile_path,
        "-c:v", "h264_nvenc",
        out_path,
    ])


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
            f"drawtext=fontfile=/home/amor/Downloads/croissant-one/CroissantOne-Regular.ttf:text='{subtitle}':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=7*(h-text_h)/8:box=1:boxcolor=black@0.5:boxborderw=1",
             '-c:v', 'h264_nvenc', "-qp", "32", "-b:v", "3000k", "-minrate", "3000k", "-maxrate", "10000k",
            '-codec:a', 'copy',
            output_path]
    if min_duration is not None:
        supp_args = ["-loop", "1", "-t", str(min_duration), ]
        for i, t in zip(range(2, 2 + len(supp_args)), supp_args):
            args.insert(i, t)

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
    rez = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file_path, '-c:v', 'h264_nvenc',
                          "-qp", "32", "-b:v", "3000k", "-minrate", "3000k", "-maxrate", "10000k",
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


def read_from_queue(queue_name: str,
                    parser: Callable[[bytes], Any] = question_parser) -> Optional[Any]:
    filepath = f"{queue_directory}/{queue_name}.txt"
    while True:
        infile = open(filepath, "rb")
        for line in infile:
            yield parser(line)
        infile.close()
        with open(filepath, "wb") as f:
            f.write(b"")
        yield None


def add_to_queue(queue_name, message):
    with open(f"{queue_directory}/{queue_name}.txt", "a") as f:
        if not message.endswith("\n"):
            message += "\n"
        f.write(f"{message}")


def replace_number_to_text(str_):
    """
    str_ = "souvenez-vous toujours de ce sage conseil du 'Livre des Saveurs Éternelles', chapitre trois, verset 10.13"
    >>> def replace_number_to_text(str_):
    ...     pattern = "\d+\.?\d{0,2}"
    ...     for p in sorted(re.findall(pattern, str_), key=lambda x: len(x), reverse=True):
    ...         str_ = str_.replace(p, num2words.num2words(p, lang="fr"))
    ...     return str_
    ...
    >>> replace_number_to_text(str_)
    "souvenez-vous toujours de ce sage conseil du 'Livre des Saveurs Éternelles', chapitre trois,
    verset dix virgule un trois"
    """
    pattern = "\d+\.?\d{0,2}"
    for p in sorted(re.findall(pattern, str_), key=lambda x: len(x), reverse=True):
        str_ = str_.replace(p, num2words.num2words(p, lang="fr"))
    return str_


def gpt_call_mock(_prompt):
    return """
        Bonjour Dany, merci pour cette demande. Les hosties symbolisent mon corps donné pour vous, et leur préparation
        requiert du pain sans levain. Elles sont un rappel du dernier repas que j'ai partagé avec mes disciples 
        avant ma crucifixion, où j'ai offert le pain comme mon corps sacrifié.
        Dans la Bible, lors de la Cène, je partageais le pain avec mes disciples en disant : 
        "Prenez, mangez, ceci est mon corps" (Matthieu 26:26). Cette action symbolique 
        représente l'unité entre les croyants 
        et moi-même, ainsi que le sacrifice ultime que j'ai fait pour l'humanité.
        """


def gpt_call(prompt):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0.3,
                                              messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content


def sanitize_str(str_):
    for c in ["/", ",", ";", ":", "!", "?", ".", "\n", "\t", "\r"]:
        str_ = str_.replace(c, "")
    for c in [" "]:
        str_ = str_.replace(c, "_")
    return str_


def update_next_qestion_file(question: Question):
    with open("./next_question.txt", "wb") as f:
        if question is not None:
            f.write(f"Question de {question.name} : {question.question[:500]}".encode("utf-8"))
        else:
            f.write(f"Pas de question en cours".encode("utf-8"))