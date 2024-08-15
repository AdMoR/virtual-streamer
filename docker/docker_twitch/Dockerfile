FROM python:3.9

ENV WORKDIR /src
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

COPY ./requirements_cpu.txt .
RUN pip install -r requirements_cpu.txt

COPY ./prompts.py .
COPY ./utils.py .
COPY ./chat_reader.py .

ENTRYPOINT ["python3", "chat_reader.py"]