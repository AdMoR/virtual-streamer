FROM python:3.9

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt update && apt install ffmpeg -y

COPY ./requirements_cpu.txt .
RUN pip install -r requirements_cpu.txt

COPY ./virtual_streamer .
COPY ./setup.py .
RUN pip install -e .

COPY ./apps/obs_video_server.py .

ENTRYPOINT ["python3", "obs_video_server.py"]