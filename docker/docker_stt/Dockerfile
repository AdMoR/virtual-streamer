FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# install python via pyenv
RUN apt-get update && apt-get install -y --no-install-recommends \
	make \
	build-essential \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	wget \
	curl \
	llvm \
	libncurses5-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libffi-dev \
	liblzma-dev \
	git \
	ca-certificates \
    libgl1 \
    cython3 \
    ffmpeg \
    libgirepository1.0-dev \
	&& rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
ARG PYTHON_VERSION=3.9
RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
	pyenv install $PYTHON_VERSION && \
	pyenv global $PYTHON_VERSION

WORKDIR /src
RUN pip install cython numpy typing-extensions
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
RUN cd whisper-diarization.git && pip install -r requirements.txt
RUN pip install huggingface-hub==0.20.3

RUN python3 /src/whisper-diarization/diarize.py -a test.mp3