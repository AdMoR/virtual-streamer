FROM python:3.9

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt update && apt install ffmpeg -y

COPY ./requirements_cpu.txt .
RUN pip install -r requirements_cpu.txt

COPY ./video_server.py .
COPY ./utils.py .
COPY ./prompts.py .

ENTRYPOINT ["python3", "video_server.py"]