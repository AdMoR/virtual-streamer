import numpy as np
from transformers import pipeline
import gradio as gr
import os
import time
from utils import add_to_queue, read_from_queue, speech_to_text_call
from pydub import AudioSegment

transcriber = pipeline("automatic-speech-recognition", model="qanastek/whisper-base-french-cased")
com_channel = "jesus_chat_123"

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y[:, 0]})["text"]


def openai_transcribe(audio):
    sr, y = audio
    audio = AudioSegment(
        data=y.tobytes(),
        sample_width=2,
        frame_rate=sr,
        channels=y.shape[1]
    )
    # Save the audio as an MP3 file
    temp_path = "microphone_input.mp3"
    audio.export(temp_path, format="mp3")
    return speech_to_text_call(temp_path)


def add_text(history, audio):
    text = openai_transcribe(audio)
    print("Transcribed : ", text)
    add_to_queue("chat_log", f"GentilUtilisateur|{text}|{com_channel}")
    response = None
    counter = 0
    while response is None and counter < 120:
        response = next(read_from_queue(com_channel, lambda x: x.decode("utf-8")))
        time.sleep(1)
        counter += 1
    history = history + [(text, None), (None, (response,))]
    return history, None


def bot(history):
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        height=800
    )

    with gr.Row():
        audio = gr.Audio(source="microphone")
        txt_msg = audio.stop_recording(add_text, [chatbot, audio], [chatbot, audio], queue=False).then(
            bot, chatbot, chatbot
        )



demo.queue()
if __name__ == "__main__":
    demo.launch()

