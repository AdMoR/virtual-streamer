FROM python:3.9

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

RUN apt update && apt install ffmpeg -y

COPY ./requirements_demo.txt .
RUN pip install -r requirements_demo.txt

COPY ./demo.py .
COPY ./character_setup.py .
COPY ./utils.py .

ENTRYPOINT ["python3", "apps/demo.py"]