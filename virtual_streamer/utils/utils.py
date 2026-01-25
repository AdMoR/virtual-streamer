import enum
import json
import logging

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
import boto3
from botocore.exceptions import NoCredentialsError
import ormsgpack
import httpx
import aiofiles


queue_directory = "./"


def combine_video_and_audio(vfile_path, afile_path, out_path):
    return subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            vfile_path,
            "-i",
            afile_path,
            "-c:v",
            "h264_nvenc",
            out_path,
        ]
    )


def combine_video_and_short_audio(vfile_path, afile_path, out_path, target_fps=30):
    duration = get_length(afile_path)
    return subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            vfile_path,
            "-i",
            afile_path,
            "-vf",
            f"fps={target_fps},scale=720:480",  # Normalize fps and resolution
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            str(duration),
            "-c:v",
            "h264_nvenc",
            "-c:a",
            "aac",           # Explicit audio encoding for consistency
            "-ar",
            "44100",         # Normalize audio sample rate
            "-vsync",
            "cfr",           # Force constant frame rate output
            out_path,
        ]
    )


def create_video_from_image(image_path, output_path, duration):
    args = [
        "ffmpeg",
        "-y",
        "-i",
        image_path,
        "-loop",
        "1",
        "-t",
        str(duration),
        output_path,
    ]
    rez = subprocess.run(args)
    if rez.returncode != 0:
        raise Exception(rez)
    return output_path


def add_subtitle(
    subtitle, video_path, output_path, min_duration=None, fontsize=28
) -> str:
    # 2 - Add the title
    for char in ":?/!":
        subtitle = subtitle.replace(char, "\\" + char)
    subtitle = subtitle.replace("'", "`").replace('"', "")
    subtitle = "\n".join(wrap(subtitle, 50))
    args = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-filter_complex",
        f"drawtext=fontfile=/home/amor/Downloads/croissant-one/CroissantOne-Regular.ttf:text='{subtitle}':fontcolor=white:fontsize={fontsize}:x=(w-text_w)/2:y=7*(h-text_h)/8:box=1:boxcolor=black@0.5:boxborderw=1",
        "-c:v",
        "h264_nvenc",
        "-qp",
        "32",
        "-b:v",
        "3000k",
        "-minrate",
        "3000k",
        "-maxrate",
        "10000k",
        "-codec:a",
        "copy",
        output_path,
    ]
    if min_duration is not None:
        supp_args = [
            "-loop",
            "1",
            "-t",
            str(min_duration),
        ]
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


def combine_part_in_concat_file(
    video_path_list, concat_file_path, out_path, duration_per_block=None
):
    concat_file_path = build_concat_file(
        video_path_list, concat_file_path, duration_per_block
    )
    rez = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file_path,
            "-c:v",
            "h264_nvenc",
            "-qp",
            "32",
            "-b:v",
            "3000k",
            "-minrate",
            "3000k",
            "-maxrate",
            "10000k",
            "-c",
            "copy",
            out_path,
        ]
    )
    if rez.returncode == 1:
        raise Exception("ffmpeg concat failed")
    return out_path


def add_subtitle_from_srt(video_path, srt_path, output_path, fontsize=14, target_fps=30) -> str:
    # 2 - Add the title
    args = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={target_fps},subtitles={srt_path}:force_style='fontcolor=white,fontsize={fontsize},x=(w-text_w)/2,y=7*(h-text_h)/8,box=1,boxcolor=black@0.5,boxborderw=1'",
        "-c:v",
        "h264_nvenc",
        "-qp",
        "32",
        "-b:v",
        "3000k",
        "-minrate",
        "3000k",
        "-maxrate",
        "10000k",
        "-vsync",
        "cfr",           # Force constant frame rate output
        "-codec:a",
        "copy",
        output_path,
    ]

    rez = subprocess.run(args)
    if rez.returncode != 0:
        raise Exception(rez)
    return output_path


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
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


def speech_to_text_call(audio_path, prompt=""):
    with open(audio_path, "rb") as audio_data:
        transcription = openai.Audio.transcribe("whisper-1", audio_data, prompt=prompt)
        return transcription["text"]


def txt_to_speech_call(speech_lines, speaker, outpath):
    host = os.environ.get("TTS_HOST", "localhost")
    safe_string = parse.quote_plus(speech_lines)
    speaker = "male-pt-3%0A"
    url = f"http://{host}:5002/api/tts?text={safe_string}&speaker_id={speaker}&style_wav=&language_id=fr-fr"

    rez = subprocess.run(["curl", "-L", "-X", "GET", url, "--output", outpath])
    if rez.returncode == 1:
        raise Exception("Txt to speech call failed")

    return outpath


def solero_language_switch(language, speaker):
    host = os.environ.get("TTS_HOST", "0.0.0.0")
    url = f"http://{host}:8001/tts/speakers"
    response = requests.get(url)
    rez = response.json()

    if speaker not in {e["name"] for e in rez}:
        url = f"http://{host}:8001/tts/language"
        response = requests.post(url, data=json.dumps({"id": f"v3_{language}.pt"}))
        return response.ok


def txt_to_speech_call_solero(speech_lines, language, speaker, outpath):
    host = os.environ.get("TTS_HOST", "0.0.0.0")
    try:
        url = f"http://{host}:8001/tts/session"
        data = {"path": "pouet"}
        response = requests.post(url, data=json.dumps(data))
    except Exception as e:
        print(e)

    url = f"http://{host}:8001/tts/generate"
    data = {"text": speech_lines, "speaker": speaker, "session": "pouet"}
    response = requests.post(url, data=json.dumps(data))
    if not response.ok:
        raise Exception("Txt to speech call failed")
    with open(outpath, "wb") as f:
        f.write(response.content)
    return outpath


def txt_to_speech_call_fish(
    speech_lines: str,
    reference_audio=None,
    reference_text=None,
    reference_id=None,
    outpath=None,
    format="wav",
    max_new_tokens=1024,
    chunk_length=300,
    top_p=0.8,
    repetition_penalty=1.1,
    temperature=0.8,
    use_memory_cache="off",
    seed=None,
    host: str = None,
    port: int = None,
):
    """
    Call fish-speech TTS API to generate speech.

    Args:
        speech_lines: Text to be synthesized
        reference_audio: Path to reference audio file (optional)
        reference_text: Reference text for voice cloning (optional)
        reference_id: ID of pre-configured reference (optional)
        outpath: Output file path
        format: Output format (wav, mp3, flac)
        max_new_tokens: Maximum new tokens to generate (0 means no limit)
        chunk_length: Chunk length for synthesis
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        temperature: Temperature for sampling
        use_memory_cache: Cache encoded references in memory ("on" or "off")
        seed: Random seed for deterministic generation (None for random)

    Returns:
        Path to generated audio file
    """
    host = host or os.environ.get("FISH_TTS_HOST", "tts")
    port = port or os.environ.get("FISH_TTS_PORT", "8003")
    api_key = os.environ.get("FISH_TTS_API_KEY", "YOUR_API_KEY")
    url = f"http://{host}:{port}/v1/tts"
    if outpath is None:
        outpath = tempfile.mktemp(suffix=f".{format}")
    # Prepare reference audio if provided
    references = []
    if reference_audio is not None and reference_text is not None:
        with open(reference_audio, "rb") as f:
            audio_bytes = f.read()
        references.append({"audio": audio_bytes, "text": reference_text})
    # Prepare request data
    data = {
        "text": speech_lines,
        "references": references,
        "reference_id": reference_id,
        "format": format,
        "max_new_tokens": max_new_tokens,
        "chunk_length": chunk_length,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "streaming": False,
        "use_memory_cache": use_memory_cache,
        "seed": seed,
    }
    # Send request
    try:
        response = requests.post(
            url,
            #params={"format": "msgpack"},
            data=ormsgpack.packb(data),
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/msgpack",
            },
            timeout=15*60,
        )
        if response.status_code == 200:
            with open(outpath, "wb") as f:
                f.write(response.content)
            return outpath
        else:
            logging.error(f"Failed to synthesize speech : {response.status_code}, {response.text}")
            raise Exception(
                f"Fish TTS request failed with status code {response.status_code}: {response.text}"
            )
    except Exception as e:
        logging.error(f"Failed to synthesize speech  with unknown error: {e}", exc_info=True)
        raise Exception(f"Fish TTS call failed: {str(e)}")


async def txt_to_speech_call_fish_async(
    speech_lines: str,
    reference_audio=None,
    reference_text=None,
    reference_id=None,
    outpath=None,
    format="wav",
    max_new_tokens=1024,
    chunk_length=300,
    top_p=0.8,
    repetition_penalty=1.1,
    temperature=0.8,
    use_memory_cache="off",
    seed=None,
    host: str = None,
    port: int = None,
):
    """
    Async version of Fish-Speech TTS API call.

    Uses httpx.AsyncClient for non-blocking HTTP requests, allowing the
    event loop to handle other tasks while waiting for the TTS response.

    Args:
        speech_lines: Text to be synthesized
        reference_audio: Path to reference audio file (optional)
        reference_text: Reference text for voice cloning (optional)
        reference_id: ID of pre-configured reference (optional)
        outpath: Output file path
        format: Output format (wav, mp3, flac)
        max_new_tokens: Maximum new tokens to generate (0 means no limit)
        chunk_length: Chunk length for synthesis
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        temperature: Temperature for sampling
        use_memory_cache: Cache encoded references in memory ("on" or "off")
        seed: Random seed for deterministic generation (None for random)
        host: TTS service host
        port: TTS service port

    Returns:
        Path to generated audio file
    """
    host = host or os.environ.get("FISH_TTS_HOST", "tts")
    port = port or os.environ.get("FISH_TTS_PORT", "8003")
    api_key = os.environ.get("FISH_TTS_API_KEY", "YOUR_API_KEY")
    url = f"http://{host}:{port}/v1/tts"
    
    if outpath is None:
        outpath = tempfile.mktemp(suffix=f".{format}")
    
    # Prepare reference audio if provided (async file read)
    references = []
    if reference_audio is not None and reference_text is not None:
        async with aiofiles.open(reference_audio, "rb") as f:
            audio_bytes = await f.read()
        references.append({"audio": audio_bytes, "text": reference_text})
    
    # Prepare request data
    data = {
        "text": speech_lines,
        "references": references,
        "reference_id": reference_id,
        "format": format,
        "max_new_tokens": max_new_tokens,
        "chunk_length": chunk_length,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "streaming": False,
        "use_memory_cache": use_memory_cache,
        "seed": seed,
    }
    
    # Send async request
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                content=ormsgpack.packb(data),
                headers={
                    "authorization": f"Bearer {api_key}",
                    "content-type": "application/msgpack",
                },
            )
            
            if response.status_code == 200:
                async with aiofiles.open(outpath, "wb") as f:
                    await f.write(response.content)
                return outpath
            else:
                raise Exception(
                    f"Fish TTS request failed with status code {response.status_code}: {response.text}"
                )
    except httpx.TimeoutException:
        raise Exception("Fish TTS call timed out after 60 seconds")
    except Exception as e:
        raise Exception(f"Fish TTS async call failed: {str(e)}")


class SubtitleMode(enum.Enum):
    NONE = "none"
    QUESTION = "question"
    VOICE_SUBTITLE = "subtitle"


@serde.deserialize
@serde.serialize
@dataclasses.dataclass
class Question:
    name: str
    question: str
    routing_queue: Optional[str] = "video_response_queue"
    next_queue: Optional[str] = None
    prompt: Optional[str] = None

    def serialize(self):
        return to_json(self)

    def render(self) -> str:
        return self.prompt.format(name=self.name, question=self.question)


@serde.deserialize
@serde.serialize
@dataclasses.dataclass
class ChatQuestion(Question):
    history: List[tuple[str, str]] = dataclasses.field(default_factory=list)
    character_name: str = "Jesus"
    subtitle_mode: SubtitleMode = SubtitleMode.QUESTION

    def render(self):
        question = self.question.replace("!allo", "")
        history = self.history
        history_str = "\n".join(
            f"{self.name}: {e[0]}\n{self.character_name}: {e[1]}\n"
            for e in history
            if e[0] is not None and e[1] is not None
        )
        return self.prompt.format(
            name=self.name, question=question, history=history_str
        )


@serde.deserialize
@serde.serialize
@dataclasses.dataclass
class VideoResponse:
    video_path: str
    text_response: str
    request: str

    def serialize(self):
        return to_json(self)


def question_parser(line: bytes) -> Question:
    text = line.decode("utf-8")
    for class_ in [ChatQuestion, Question]:
        try:
            return from_json(class_, text)
        except Exception as e:
            pass
    raise Exception("Incorrect message format")


def read_from_queue_old(
    queue_name: str, parser: Callable[[bytes], Any] = question_parser
) -> Optional[Any]:
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
    if "RMQ_URL" not in os.environ:
        print("Host Rabbit mode", os.environ.get("RMQ_HOST", "localhost"))
        host = os.environ.get("RMQ_HOST", "localhost")
        parameters = pika.ConnectionParameters(
            host=host, port=5672, retry_delay=10, connection_attempts=3
        )
    else:
        print("Remote Rabbit mode", os.environ["RMQ_URL"])
        parameters = pika.URLParameters(os.environ["RMQ_URL"])
    connection = pika.BlockingConnection(parameters)

    # Create a channel
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    return channel


def add_to_queue(queue_name, message):
    # Define the connection parameters
    channel = get_rmq_channel(queue_name)
    # Send the message
    channel.basic_publish(exchange="", routing_key=queue_name, body=message)


def read_from_queue(queue_name: str, parser: Callable[[bytes], Any]) -> Optional[Any]:
    channel = get_rmq_channel(queue_name)
    channel.consume()
    while True:
        method_frame, header_frame, body = channel.basic_get(
            queue=queue_name, auto_ack=True
        )
        if method_frame:
            yield parser(body)
        else:
            break
    yield None


def replace_number_to_text(str_):
    """
    str_ = "souvenez-vous toujours de ce sage conseil du 'Livre des Saveurs Éternelles', chapitre trois, verset 10.13"
    >>> def replace_number_to_text(str_):
    ...     pattern = r"\d+\.?\d{0,2}"
    ...     for p in sorted(re.findall(pattern, str_), key=lambda x: len(x), reverse=True):
    ...         str_ = str_.replace(p, num2words.num2words(p, lang="fr"))
    ...     return str_
    ...
    >>> replace_number_to_text(str_)
    "souvenez-vous toujours de ce sage conseil du 'Livre des Saveurs Éternelles', chapitre trois,
    verset dix virgule un trois"
    """
    pattern = r"\d+\.?\d{0,2}"
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
    openai.api_key = os.environ["OPENAI_TOKEN"]
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
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
            f.write(
                f"Question de {question.name} : {question.question[:500]}".encode(
                    "utf-8"
                )
            )
        else:
            f.write(f"Pas de question en cours".encode("utf-8"))


def s3_upload(local_file_path, bucket_name):
    # AWS credentials and S3 bucket information

    try:
        aws_access_key_id = os.environ["AWS_ACCESS_KEY"]
        aws_secret_access_key = os.environ["AWS_SECRET_KEY"]
    except KeyError:
        session = boto3.Session()
        credentials = session.get_credentials()
        credentials = credentials.get_frozen_credentials()
        aws_access_key_id = credentials.access_key
        aws_secret_access_key = credentials.secret_key
    object_name = os.path.basename(local_file_path)

    # Initialize the S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    # Upload the file to the S3 bucket
    s3.upload_file(local_file_path, bucket_name, object_name)
    object_dst_url = f"{bucket_name}/{object_name}"
    print(f"Successfully uploaded {local_file_path} to {bucket_name}/{object_name}")
    """
    except NoCredentialsError:
        print('AWS credentials not available or incorrect. Please configure your credentials.')
    except Exception as e:
        print(f'An error occurred: {str(e)}')
    """
    return object_dst_url


def s3_download(s3_path: str, output_dir: Optional[str] = None):
    try:
        aws_access_key_id = os.environ["AWS_ACCESS_KEY"]
        aws_secret_access_key = os.environ["AWS_SECRET_KEY"]
    except KeyError:
        session = boto3.Session()
        credentials = session.get_credentials()
        credentials = credentials.get_frozen_credentials()
        aws_access_key_id = credentials.access_key
        aws_secret_access_key = credentials.secret_key

    # Initialize the S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    # Upload the file to the S3 bucket
    bucket, *others = s3_path.split("/")
    filename = others[-1]
    if output_dir is None:
        output_dir = "../.."
    full_local_path = os.path.join(output_dir, filename)
    s3.download_file(bucket, "/".join(others), full_local_path)
    return full_local_path
