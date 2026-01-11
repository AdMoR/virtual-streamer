"""
DEPRECATED: This file is deprecated and will be removed in a future version.

The functionality has been migrated to:
- virtual_streamer/streaming/video_server/app.py (FastAPI-based video server)
- virtual_streamer/api/low_level/playlist.py (Playlist management API)

The new architecture:
- Uses database-driven playlists instead of RabbitMQ
- Uses MinIO presigned URLs instead of local file serving
- Uses the main Virtual Streamer API for all logic

To use the new video server:
    docker compose -f compose_streaming.yml up video_server

Or run locally:
    python -m uvicorn virtual_streamer.streaming.video_server.app:app --port 5000
"""
import warnings
warnings.warn(
    "obs_video_server.py is deprecated. Use virtual_streamer.streaming.video_server instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import random
import threading
from multiprocessing.pool import ThreadPool
import flask
from serde.json import from_json
from flask import Flask, request, jsonify
import json
from flask_cors import CORS
from virtual_streamer.utils.utils import get_rmq_channel, s3_download, VideoResponse


# RabbitMQ connection parameters
# rmq_url = os.environ["RMQ_URL"]
queue_name = os.environ.get("VIDEO_QUEUE", "obs")
video_folder = os.environ.get("VIDEO_FOLDER", "assets")
HOST_NAME = os.environ.get("HOST_NAME", "127.0.0.1")
REMOTE_VIDEO = False

app = Flask(__name__, static_folder=video_folder)
CORS(app)

old_videos = [
    os.path.join(video_folder, f)
    for f in os.listdir(video_folder)
    if f.endswith(".mp4")
]
new_videos = list()


@app.route("/videos")
def videos():
    old_videos = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.endswith(".mp4")
    ]
    if len(new_videos) > 0:
        new_vid = new_videos.pop(0)
        random.shuffle(old_videos)
        old_videos.insert(0, new_vid)
    videos = (
        [f"http://{HOST_NAME}:5000/video/{f}" for f in old_videos]
        if REMOTE_VIDEO
        else [f"{f}" for f in old_videos]
    )
    response = app.response_class(
        response=json.dumps(videos), status=200, mimetype="application/json"
    )
    return response


@app.route("/video/<path:filename>")
def video_server(filename: str):
    return app.send_static_file(filename)


@app.route("/hasNewVideo")
def hasNewVideos():
    has_video = len(new_videos) > 0
    response = app.response_class(
        status=200 if has_video else 404, mimetype="application/json"
    )
    return response


@app.route("/")
def home():
    return app.send_static_file("test.html")


def start_flask():
    app.run(host="0.0.0.0", debug=True, use_reloader=False)


def start_rabbitmq_consumer():
    def callback(ch, method, properties, body):
        response = from_json(VideoResponse, body.decode())
        s3_path = response.video_path
        try:
            file = s3_download(s3_path, output_dir=video_folder)
        except Exception as e:
            print("Exception encountered in S3 download")
            raise e
        new_videos.append(file)
        print("New video added ---->", new_videos)

    channel = get_rmq_channel(queue_name)
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

    print("Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()
    print("exiting")


if __name__ == "__main__":
    # Start Flask web server in a separate thread
    pool = ThreadPool(processes=2)

    pool.apply_async(start_flask)
    pool.apply_async(start_rabbitmq_consumer)

    print("Ready")
    pool.close()
    pool.join()
