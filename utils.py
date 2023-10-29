import enum
import json
import serde
from serde.json import from_json, to_json
from typing import List, Dict, Optional, Callable, Any
import re
import num2words
from urllib import parse
from textwrap import wrap
import subprocess
import tempfile
import dataclasses
import openai
import os
import pika
import requests
import abc
from abc import ABC


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


def speech_to_text_call(audio_path, prompt=""):
    with open(audio_path, 'rb') as audio_data:
        transcription = openai.Audio.transcribe("whisper-1", audio_data, prompt=prompt)
        return transcription['text']


def txt_to_speech_call(speech_lines, speaker, outpath):
    host = os.environ.get("TTS_HOST", "localhost")
    safe_string = parse.quote_plus(speech_lines)
    url = f"http://{host}:5002/api/tts?text={safe_string}&speaker_id={speaker}&style_wav=&language_id=fr-fr"

    rez = subprocess.run(["curl", "-L", "-X", "GET", url, "--output", outpath])
    if rez.returncode == 1:
        raise Exception("Txt to speech call failed")

    return outpath


def txt_to_speech_call_bis(speech_lines, speaker, outpath):
    safe_string = parse.quote_plus(speech_lines)
    url = f"http://0.0.0.0:59125/api/tts?text={safe_string}&voice=fr_FR/tom_low&noiseScale=0.9&noiseW=0.1&lengthScale=0.8&ssml=false&audioTarget=client"
    rez = subprocess.run(["curl", "-L", "-X", "GET", url, "--output", outpath])
    if rez.returncode == 1:
        raise Exception("Txt to speech call failed")
    return outpath


def solero_language_switch(language, speaker):
    url = f"http://0.0.0.0:8001/tts/speakers"
    response = requests.get(url)
    rez = response.json()

    if speaker not in {e["name"] for e in rez}:
        url = f"http://0.0.0.0:8001/tts/language"
        response = requests.post(url, data=json.dumps({
          "id": f"v3_{language}.pt"
        }))
        return response.ok


def txt_to_speech_call_solero(speech_lines, language, speaker, outpath):
    try:
        url = f"http://0.0.0.0:8001/tts/session"
        data = {
            "path": "pouet"
        }
        response = requests.post(url, data=json.dumps(data))
    except Exception as e:
        print(e)

    url = f"http://0.0.0.0:8001/tts/generate"
    data = {
        "text": speech_lines,
        "speaker": speaker,
        "session": "pouet"
    }
    response = requests.post(url, data=json.dumps(data))
    if not response.ok:
        raise Exception("Txt to speech call failed")
    with open(outpath, "wb") as f:
        f.write(response.content)
    return outpath


class AbstractPromptQuery(ABC):
    @abc.abstractmethod
    def render(self) -> str:
        pass
    def serialize(self) -> str:
        pass


@serde.deserialize
@serde.serialize
@dataclasses.dataclass
class Question(AbstractPromptQuery):
    name: str
    question: str
    routing_queue: str = "video_response_queue"
    prompt: str = None

    def serialize(self):
        return to_json(self)

    def render(self) -> str:
        return self.prompt.format(name=self.name, question=self.question)


class SubtitleMode(enum.Enum):
    NONE = "none"
    QUESTION = "question"
    VOICE_SUBTITLE = "subtitle"

@serde.deserialize
@serde.serialize
@dataclasses.dataclass
class ChatQuestion(AbstractPromptQuery):
    name: str
    question: str
    routing_queue: str = "video_response_queue"
    prompt: str = None
    history: List[tuple[str, str]] = dataclasses.field(default_factory=list)
    character_name: str = "Jesus"
    subtitle_mode: SubtitleMode = SubtitleMode.QUESTION

    def serialize(self):
        return to_json(self)

    def render(self):
        question = self.question.replace("!allo", "")
        history = self.history
        history_str = "\n".join(f"{self.name}: {e[0]}\nJesus: {e[1]}\n"
                                for e in history if e[0] is not None and e[1] is not None)
        return self.prompt.format(name=self.name, question=question, history=history_str)


@serde.deserialize
@serde.serialize
@dataclasses.dataclass
class VideoResponse:
    video_path: str
    text_response: str
    request: str

    def serialize(self):
        return to_json(self)


def question_parser(line: bytes) -> AbstractPromptQuery:
    text = line.decode("utf-8")
    for class_ in [ChatQuestion, Question]:
        try:
            return from_json(class_, text)
        except Exception as e:
            pass
    raise Exception("Incorrect message format")


def read_from_queue_old(queue_name: str,
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


def add_to_queue_old(queue_name, message):
    with open(f"{queue_directory}/{queue_name}.txt", "a") as f:
        if not message.endswith("\n"):
            message += "\n"
        f.write(f"{message}")


def get_rmq_channel(queue_name):
    # Define the connection parameters
    host = os.environ.get("RMQ_HOST", "localhost")
    connection_params = pika.ConnectionParameters(host=host)
    connection = pika.BlockingConnection(connection_params)

    # Create a channel
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    return channel


def add_to_queue(queue_name, message):
    # Define the connection parameters
    channel = get_rmq_channel(queue_name)
    # Send the message
    channel.basic_publish(exchange='', routing_key=queue_name, body=message)


def read_from_queue(queue_name: str,
                    parser: Callable[[bytes], Any]) -> Optional[Any]:
    channel = get_rmq_channel(queue_name)
    channel.consume()
    while True:
        method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
        if method_frame:
            yield parser(body)
        else:
            break
    yield None


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