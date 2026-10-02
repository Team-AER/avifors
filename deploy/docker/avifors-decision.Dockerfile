# syntax=docker/dockerfile:1
# System One decision worker (Laya on CPU torch). Build from the repository root:
#   docker build -f deploy/docker/avifors-decision.Dockerfile -t avifors-decision:local .
# Dependencies come from deploy/docker/requirements/decision.txt (hash-pinned, CPU torch from
# https://download.pytorch.org/whl/cpu); regenerate it with the command in its header.
ARG PYTHON_IMAGE=python:3.14-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.19

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
RUN uv venv --python /usr/local/bin/python3 /opt/avifors-decision
COPY deploy/docker/requirements/decision.txt /tmp/decision.txt
RUN uv pip install --python /opt/avifors-decision/bin/python --torch-backend cpu --require-hashes \
        --no-deps -r /tmp/decision.txt
COPY pyproject.toml README.md LICENSE /src/
COPY avifors /src/avifors
RUN uv pip install --python /opt/avifors-decision/bin/python --no-deps /src

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 990 avifors \
    && useradd --system --uid 996 --gid 990 --home-dir /var/lib/avifors --no-create-home \
       --shell /usr/sbin/nologin avifors
COPY --from=build /opt/avifors-decision /opt/avifors-decision
# CPU lane: never touch the managed GPU, offline Hugging Face, three threads (as the systemd units).
# TOKENIZERS_PARALLELISM is not a secret.
# hadolint ignore=DL3064
ENV PATH=/opt/avifors-decision/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    CUDA_VISIBLE_DEVICES= \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=3
USER 996:990
WORKDIR /tmp
ENTRYPOINT ["python", "-m", "avifors.decision_worker"]
