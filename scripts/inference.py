import os
import json
import random
import requests
from virtual_streamer.workflows.character_setup import CHARACTERS
from virtual_streamer.utils.utils import get_rmq_channel, question_parser, gpt_call, VideoResponse, SubtitleMode
from virtual_streamer.workflows.prompts import PROMPT, PROMPT_FR, PROMPT_FR_3, PROMPT_FR_2, SARCASTIC_PROMPT_FR, \
    STAND_UP_PROMPT, SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT

# Configuration
UPLOAD_BUCKET = os.environ.get("S3_BUCKET_URL", "default-bucket")
WEBSERVICE_URL = os.environ.get("WEBSERVICE_URL", "http://localhost:5000")

def main(question):
    """Process a question by calling the web service"""
    # Step 1 - Get the response from GPT
    if question.prompt is None:
        question.prompt = random.choice([SARCASTIC_STANDUP, VERY_SARCASTIC_STANDUP_PROMPT, VERY_SARCASTIC_PROMPT])
    query = question.render()
    text = gpt_call(query)
    
    # Save the query and response for debugging
    os.makedirs("prompts", exist_ok=True)
    json.dump({"query": query, "response": text, "question": question.question},
              open(f"prompts/response_{hash(query) % 1000000}.json", "w"))
    
    # Step 2 - Call the web service to process the video
    question_data = {
        "question": question.question,
        "character_name": question.character_name,
        "subtitle_mode": question.subtitle_mode.name if hasattr(question.subtitle_mode, 'name') else question.subtitle_mode,
        "name": question.name
    }
    
    payload = {
        "question": question_data,
        "gpt_response": text
    }
    
    try:
        response = requests.post(f"{WEBSERVICE_URL}/process", json=payload)
        response.raise_for_status()
        result = response.json()
        
        return result["video_path"], result["response_text"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling web service: {e}")
        # Fallback to local file if web service fails
        return f"./error_video_{hash(query) % 100000}.mp4", text


def callback(ch, method_frame, properties, body):
    # 1 - Retrieve some params of the processing job
    print("body", body)
    question = question_parser(body)
    
    # 2 - main processing: video answer to the question is produced and stored on S3
    media_path, text = main(question)
    print(f"New video_path : {media_path}")
    
    # 3 - A msg is prepared and send to the next user
    # If the media_path is already an S3 path (from the web service), use it directly
    if media_path.startswith("http"):
        s3_path = media_path
    else:
        # For local files, upload to S3 if needed
        from virtual_streamer.utils.utils import s3_upload
        s3_path = s3_upload(media_path, UPLOAD_BUCKET) if UPLOAD_BUCKET != "default-bucket" else media_path
    
    msg = VideoResponse(s3_path, text, question.question).serialize()
    routing_key = properties.reply_to or question.routing_queue
    if routing_key:
        print("Reply : ", routing_key)
        ch.basic_publish('', routing_key=routing_key, body=msg)
        ch.basic_ack(delivery_tag=method_frame.delivery_tag)


def main_exec(channel_name="chat_log"):
    # Check if web service is available
    try:
        response = requests.get(f"{WEBSERVICE_URL}/health")
        if response.status_code == 200:
            print(f"Web service is available at {WEBSERVICE_URL}")
        else:
            print(f"Warning: Web service returned status code {response.status_code}")
    except requests.exceptions.RequestException:
        print(f"Warning: Could not connect to web service at {WEBSERVICE_URL}")
    
    # Start consuming from RMQ
    channel = get_rmq_channel(channel_name)
    channel.basic_consume(queue=channel_name, on_message_callback=callback)
    print(f"Ready to read {channel_name}")
    channel.start_consuming()


if __name__ == '__main__':
    main_exec()
