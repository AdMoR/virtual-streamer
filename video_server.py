import os
import random
import threading

import flask
from serde.json import from_json
from flask import Flask, request, jsonify
import json
from flask_cors import CORS
from utils import get_rmq_channel, s3_download, VideoResponse


# RabbitMQ connection parameters
#rmq_url = os.environ["RMQ_URL"]
queue_name = os.environ.get("VIDEO_QUEUE", "obs")
video_folder = os.environ.get("VIDEO_FOLDER", "assets")

app = Flask(__name__, static_folder=video_folder)
CORS(app)

old_videos = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
new_videos = list()


@app.route('/videos')
def videos():
    if len(new_videos) > 0:
        new_vid = new_videos.pop(0)
        random.shuffle(old_videos)
        old_videos.append(new_vid)
    videos = [f"http://127.0.0.1:5000/video/{f}" for f in old_videos]
    response = app.response_class(
        response=json.dumps(videos),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/video/<path:filename>')
def video_server(filename: str):
    return app.send_static_file(filename)


@app.route('/done/<path:filename>')
def video_completed(filename: str):
    response = app.response_class(
        response=json.dumps("ok"),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/')
def home():
    return app.send_static_file("test.html")


def start_flask():
    app.run(debug=True, use_reloader=False)


def start_rabbitmq_consumer():
    def callback(ch, method, properties, body):
        response = from_json(VideoResponse, body.decode())
        s3_path = response.video_path
        file = s3_download(s3_path, output_dir=video_folder)
        new_videos.append(file)

    channel = get_rmq_channel(queue_name)
    channel.queue_declare(queue=queue_name)
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    # Start Flask web server in a separate thread
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()

    # Start RabbitMQ consumer in the main thread
    #start_rabbitmq_consumer()

