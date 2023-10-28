import numpy as np
from transformers import pipeline
import gradio as gr
import os
import time
import json
import pika
from utils import add_to_queue, read_from_queue, speech_to_text_call, get_rmq_channel

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


def bind_callback_to_history(chatbot, server_queue="chat_log", prompt=HIST_PROMPT):

    def response_parser(text):
        response = text.decode("utf-8")
        video_path, text_response, query_text, *_ = response.split("|")
        #chatbot.value.extend()
        chatbot.update(value=chatbot.value + [(query_text, text_response), (None, (video_path,))])

    def on_client_rx_reply_from_server(ch, method_frame, properties, body):
        response_parser(body)
        # NOTE A real client might want to make additional RPC requests, but in this
        # simple example we're closing the channel after getting our first reply
        # to force control to return from channel.start_consuming()
        print('RPC Client says bye')
        ch.close()

    channel = get_rmq_channel(server_queue)
    channel.basic_consume('amq.rabbitmq.reply-to',
                          on_client_rx_reply_from_server,
                          auto_ack=True)

    def query_send(audio):
        query_text = speech_to_text_call(audio, "Ich spreche Deutsch.")
        text = f"GentilUtilisateur|{query_text}||{prompt}|{json.dumps(chatbot.value)}"
        print("Swoosh !")
        channel.basic_publish(
            exchange='',
            routing_key=server_queue,
            body=text,
            properties=pika.BasicProperties(reply_to='amq.rabbitmq.reply-to')
        )

    # channel.start_consuming()
    return channel, query_send


def build_callback(server_queue="chat_log", prompt=HIST_PROMPT):
    # 1 - Prepare the com channel
    channel = get_rmq_channel(server_queue)
    # The answer channel must be prepared
    next(channel.consume(queue="amq.rabbitmq.reply-to", auto_ack=True, inactivity_timeout=0.1))

    def aaaa(chatbot, audio):
        # 2 - Get the query and send it
        query_text = speech_to_text_call(audio, "Ich spreche Deutsch.")
        text = f"GentilUtilisateur|{query_text}||{prompt}|{json.dumps(chatbot)}"
        channel.basic_publish(
            exchange="", routing_key=server_queue, body=text.encode(),
            properties=pika.BasicProperties(reply_to="amq.rabbitmq.reply-to"))
        print("sent:", text)

        # 3 - Wait for the response from server and update history
        def response_parser(chatbot, text):
            response = text.decode("utf-8")
            video_path, text_response, query_text, *_ = response.split("|")
            chatbot += [(query_text, text_response), (None, (video_path,))]
        (method, properties, body) = next(channel.consume(queue="amq.rabbitmq.reply-to", auto_ack=True,
                                                          inactivity_timeout=2*60))
        response_parser(chatbot, body)
        return chatbot, None

    return aaaa


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
        height=800
    )

    callback = build_callback()

    with gr.Row():
        audio = gr.Audio(source="microphone", type="filepath")
        txt_msg = audio.stop_recording(callback, [chatbot, audio], [chatbot, audio], queue=False)


demo.queue()
if __name__ == "__main__":
    demo.launch()

