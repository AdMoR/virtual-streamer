import numpy as np
from transformers import pipeline
import gradio as gr
import os
import time
import json
from utils import add_to_queue, read_from_queue, speech_to_text_call

com_channel = "jesus_chat_123456"

"""
def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y[:, 0]})["text"]
"""

PROMPT = """
    You are a german teacher. You are roleplaying with your student as Jesus to help him learn. 
    The level of your student is B1 so you have to make simple sentence but you should create some content to keep the conversation going. 
    You can also precise some grammatical or lexical points.
    ```
    {name}: {question}
    Jesus: 
    ```
    Only generate what Jesus would say.
    """

HIST_PROMPT = """
    You are a german teacher. You are roleplaying with your student as Jesus to help him learn. 
    The level of your student is B1 so you have to make simple sentence but you should create some content to keep the conversation going. 
    You can also precise some grammatical or lexical points.
    ```
    {history}
    {name}: {question}
    Jesus: 
    ```
    Only generate what Jesus would say.
    """


def add_text(history, audio):
    tt = time.time()
    query_text = speech_to_text_call(audio, "Ich spreche Deutsch.")
    print("Transcribed : ", query_text, f" in {time.time() - tt} secs.")
    add_to_queue("chat_log", f"GentilUtilisateur|{query_text}|{com_channel}|{HIST_PROMPT}|{json.dumps(history)}")
    response = None
    counter = 0
    while response is None and counter < 120:
        response = next(read_from_queue(com_channel, lambda x: x.decode("utf-8")))
        time.sleep(1)
        counter += 1
    print("Total video time : ", counter)
    video_path, text_response, *_ = response.split("|")
    history = history + [(query_text, text_response), (None, (video_path,))]
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
        audio = gr.Audio(source="microphone", type="filepath")
        txt_msg = audio.stop_recording(add_text, [chatbot, audio], [chatbot, audio], queue=False)


demo.queue()
if __name__ == "__main__":
    demo.launch()

