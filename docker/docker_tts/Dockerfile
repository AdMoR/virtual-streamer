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
    libgirepository1.0-dev \
	&& rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
ARG PYTHON_VERSION=3.9
RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
	pyenv install $PYTHON_VERSION && \
	pyenv global $PYTHON_VERSION

# install deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libsndfile1 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/admor/silero-api-server.git && cd silero-api-server && git pull origin master && pip install -e . && pip install -r requirements.txt

CMD ["python3", "-m", "silero_api_server"]