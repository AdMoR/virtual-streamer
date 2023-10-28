import numpy as np
from transformers import pipeline
import gradio as gr
import os
import time
import json
import pika
from serde.json import from_json
from utils import add_to_queue, read_from_queue, speech_to_text_call, get_rmq_channel, ChatQuestion, VideoResponse
from character_setup import CHARACTERS

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

prompt_dict = {
    "de": "Ich spreche Deutch",
    "fr": "Je parle français",
    "en": "I speak english"
}



def build_callback(server_queue="chat_log", prompt=HIST_PROMPT):
    # 1 - Prepare the com channel
    channel = get_rmq_channel(server_queue)
    # The answer channel must be prepared
    next(channel.consume(queue="amq.rabbitmq.reply-to", auto_ack=True, inactivity_timeout=0.1))

    def aaaa(chatbot, audio, character):
        # 2 - Get the query and send it
        language = CHARACTERS[character].language
        query_text = speech_to_text_call(audio, prompt_dict.get(language, ""))
        q = ChatQuestion("GentilUtilisateur", query_text, None, prompt, chatbot,
                         character_name=character)
        text = q.serialize()
        print("Text content : ", text)
        channel.basic_publish(
            exchange="", routing_key=server_queue, body=text.encode(),
            properties=pika.BasicProperties(reply_to="amq.rabbitmq.reply-to"))
        print("sent:", text)

        # 3 - Wait for the response from server and update history
        def response_parser(history, text_response):
            response = from_json(VideoResponse, text_response.decode())
            history += [(response.request, response.text_response), (None, (response.video_path,))]
        (method, properties, body) = next(channel.consume(queue="amq.rabbitmq.reply-to", auto_ack=True,
                                                          inactivity_timeout=2*60))
        response_parser(chatbot, body)
        return chatbot, None, language

    return aaaa


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        height=800
    )
    language = gr.Dropdown(choices=CHARACTERS.keys(), value="Jesus_de", label="Character selection")
    callback = build_callback()

    with gr.Row():
        audio = gr.Audio(source="microphone", type="filepath")
        txt_msg = audio.stop_recording(callback, [chatbot, audio, language], [chatbot, audio, language], queue=False)


demo.queue()
if __name__ == "__main__":
    demo.launch()

