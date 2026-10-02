# syntax=docker/dockerfile:1
# Omnilingual ASR speech worker (CUDA). Build from the repository root:
#   docker build -f deploy/docker/avifors-stt-omni.Dockerfile -t avifors-stt-omni:local .
#
# ############################################################################################
# TIME-LIMITED EXCEPTION TO THE LATEST-PYTHON RULE: this image stays on Python 3.12.
# omnilingual-asr declares requires-python <=3.12 and fairseq2n (0.6 through 0.8.1) publishes
# wheels only up to cp312, so no newer interpreter can run this engine. Every other Avifors image
# is on Python 3.14. Remove this exception (move to python:3.14-slim, refresh the pins) as soon
# as fairseq2/fairseq2n and omnilingual-asr ship Python 3.14 support. See STT.md.
# ############################################################################################
#
# The pins are production's: deploy/stt/requirements.lock (torch 2.8.0 + CUDA 12.8 wheels,
# fairseq2/fairseq2n 0.6, omnilingual-asr at a fixed commit). The Avifors package requires
# Python 3.14, so it is installed here with --ignore-requires-python and only its stt_worker
# module runs; tests/test_controller.py keeps that module parseable as Python 3.12.
ARG PYTHON_IMAGE=python:3.12-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.19

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
# kenlm and sox are sdist-only; omnilingual-asr is pinned to a git commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
       zlib1g-dev libbz2-dev liblzma-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/avifors-stt
COPY deploy/stt/requirements.lock /tmp/requirements.lock
RUN uv pip install --python /opt/avifors-stt/bin/python -r /tmp/requirements.lock
COPY pyproject.toml README.md LICENSE /src/
COPY avifors /src/avifors
RUN /opt/avifors-stt/bin/pip install --no-deps --ignore-requires-python /src

FROM ${PYTHON_IMAGE} AS runtime
# fairseq2n links the system libsndfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 990 avifors \
    && useradd --system --uid 996 --gid 990 --home-dir /var/lib/avifors --no-create-home \
       --shell /usr/sbin/nologin avifors
COPY --from=build /opt/avifors-stt /opt/avifors-stt
ENV PATH=/opt/avifors-stt/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    FAIRSEQ2_ASSET_DIR=/etc/avifors/stt-assets \
    FAIRSEQ2_CACHE_DIR=/var/lib/avifors/fairseq2 \
    OMP_NUM_THREADS=6 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 996:990
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "avifors.stt_worker"]
