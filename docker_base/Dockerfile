FROM python:3.9.18:latest

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

COPY . .

RUN pip install -r requirements_cpu.txt