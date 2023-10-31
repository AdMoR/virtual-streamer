FROM python:3.9.18:latest

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt-get update && apt-get install -y libgirepository1.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad

COPY . .

RUN pip install -r requirements_cpu.txt