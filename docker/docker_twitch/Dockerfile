FROM python:3.9

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

COPY ./requirements_cpu.txt .
RUN pip install -r requirements_cpu.txt

COPY ./virtual_streamer .
COPY ./setup.py .
RUN pip install -e .

ENTRYPOINT ["python3", "virtual_streamer/twitch/chat_reader.py"]