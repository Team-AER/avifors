# syntax=docker/dockerfile:1
# Qwen3-ASR speech worker (CUDA). Build from the repository root:
#   docker build -f deploy/docker/avifors-stt-qwen.Dockerfile -t avifors-stt-qwen:local .
# Dependencies come from deploy/docker/requirements/stt-qwen.txt (hash-pinned; torch/torchaudio
# CUDA 12.8 wheels from https://download.pytorch.org/whl/cu128). The host driver is injected by
# the NVIDIA container runtime; CUDA user-space libraries ship inside the torch wheels.
ARG PYTHON_IMAGE=python:3.14-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.19

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN uv venv --python /usr/local/bin/python3 /opt/avifors-stt
COPY deploy/docker/requirements/stt-qwen.txt /tmp/stt-qwen.txt
# sox 1.5.0 is published only as a pure-Python sdist; everything else installs from wheels.
RUN uv pip install --python /opt/avifors-stt/bin/python --torch-backend cu128 --require-hashes \
        --no-deps -r /tmp/stt-qwen.txt
COPY pyproject.toml README.md LICENSE /src/
COPY avifors /src/avifors
RUN uv pip install --python /opt/avifors-stt/bin/python --no-deps /src

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 sox ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 990 avifors \
    && useradd --system --uid 996 --gid 990 --home-dir /var/lib/avifors --no-create-home \
       --shell /usr/sbin/nologin avifors
COPY --from=build /opt/avifors-stt /opt/avifors-stt
# librosa's numba kernels cache next to their sources unless told otherwise; site-packages is
# read-only for the avifors user, so cache under the writable /tmp instead.
ENV PATH=/opt/avifors-stt/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=6 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 996:990
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "avifors.stt_worker", "--engine", "qwen"]
