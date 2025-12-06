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


# install deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libsndfile1 \
	&& rm -rf /var/lib/apt/lists/*

#ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
#ARG PYTHON_VERSION=3.9
#RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
#	pyenv install $PYTHON_VERSION && \
#	pyenv global $PYTHON_VERSION

ENV WORKDIR /src
RUN mkdir -p $WORKDIR && mkdir /root/.aws
WORKDIR $WORKDIR
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ./uv.lock .
COPY ./pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable



# install requirements
#COPY requirements.txt .
#RUN pip install -r requirements.txt && pip install git+https://github.com/elliottzheng/batch-face.git@master
#RUN pip install fastapi[standard]==0.113.0
#RUN pip install ormsgpack
# Create necessary directories
RUN mkdir -p /data/out_video_folder /data/temp

# copy sources
COPY ./face_detection ./face_detection
COPY ./models ./models
COPY ./checkpoints ./checkpoints
COPY ./setup.py .
COPY ./webservice.py .
COPY ./webservice_prod.py .
COPY ./README.md .
COPY ./virtual_streamer ./virtual_streamer

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PYTHONUNBUFFERED=1

# Default command to run the webservice with Gunicorn and the production config
CMD ["uv", "run", "fastapi", "run", "/src/webservice.py"]
