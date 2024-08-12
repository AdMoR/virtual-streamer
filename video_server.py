import os
import threading
from serde.json import from_json
from flask import Flask, request, jsonify
from utils import get_rmq_channel, s3_download, VideoResponse

app = Flask(__name__)

# RabbitMQ connection parameters
rmq_url = os.environ["RMQ_URL"]
queue_name = os.environ.get("VIDEO_QUEUE", "obs")
video_folder = os.environ.get("VIDEO_FOLDER", ".")


@app.route('/videos')
def videos():
    return [os.path.join(video_folder, f) for f in os.listdir(video_folder)]


def start_flask():
    app.run(debug=True, use_reloader=False)


def start_rabbitmq_consumer():
    def callback(ch, method, properties, body):
        response = from_json(VideoResponse, body.decode())
        s3_path = response.video_path
        s3_download(s3_path, output_dir=video_folder)

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
    start_rabbitmq_consumer()

