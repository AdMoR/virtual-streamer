import gradio as gr
import os
import pika
from serde.json import from_json
from utils import speech_to_text_call, get_rmq_channel, ChatQuestion, \
    VideoResponse, SubtitleMode, s3_download
from character_setup import CHARACTERS


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
    The level of your student is B1 so you have to make simple sentence but 
    you should create some content to keep the conversation going. 
    You can also precise some grammatical or lexical points.
    ```
    {history}
    {name}: {question}
    Jesus: 
    ```
    Only generate what Jesus would say.
    """


prompt_dict = {
    "de": "Ich spreche Deutsch",
    "fr": "Je parle français",
    "en": "I speak english"
}


def build_callback(server_queue="chat_log", prompt=HIST_PROMPT):
    # 1 - Prepare the com channel


    def aaaa(chatbot, audio, character):
        channel = get_rmq_channel(server_queue)
        # The answer channel must be prepared
        next(channel.consume(queue="amq.rabbitmq.reply-to", auto_ack=True, inactivity_timeout=0.1))
        # 2 - Get the query and send it
        language = CHARACTERS[character].language
        query_text = speech_to_text_call(audio, prompt_dict[language])
        q = ChatQuestion("GentilUtilisateur", query_text, None, prompt, chatbot,
                         character_name=character, subtitle_mode=SubtitleMode.NONE)
        text = q.serialize()
        print("Text content : ", text)
        try:
            channel.basic_publish(
                exchange="", routing_key=server_queue, body=text.encode(),
                properties=pika.BasicProperties(reply_to="amq.rabbitmq.reply-to"))
        except pika.exceptions.ChannelWrongStateError:
            raise Exception("Connection closed")
        print("sent:", text)

        # 3 - Wait for the response from server and update history
        def response_parser(history, text_response):
            response = from_json(VideoResponse, text_response.decode())
            video_path = s3_download(response.video_path)
            history += [(response.request, response.text_response), (None, (video_path,))]
        (method, properties, body) = next(channel.consume(queue="amq.rabbitmq.reply-to", auto_ack=True,
                                                          inactivity_timeout=2*60))
        response_parser(chatbot, body)
        channel.close()
        return chatbot, audio, character

    return aaaa


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, None),
        height=800
    )
    language = gr.Dropdown(choices=CHARACTERS.keys(), value="Jesus_fr", label="Character selection")
    callback = build_callback()

    with gr.Row():
        audio = gr.Microphone(type="filepath")
        txt_msg = audio.stop_recording(callback, [chatbot, audio, language], [chatbot, audio, language], queue=False)


demo.queue()
if __name__ == "__main__":
    demo.launch(share=True)

