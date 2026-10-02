# syntax=docker/dockerfile:1
# Broker image: avifors (broker), avifors-controller (root sidecar) and avifors-workerctl.
# Build from the repository root:
#   docker build -f deploy/docker/avifors.Dockerfile -t avifors:local .
# Base images are pinned by tag; record their digests at deploy time.
ARG PYTHON_IMAGE=python:3.14-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.19

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/avifors
WORKDIR /src
# Dependencies first (cached unless the lock changes), exactly as locked in uv.lock.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project --extra audio
COPY avifors ./avifors
RUN uv sync --locked --no-dev --no-editable --extra audio

FROM ${PYTHON_IMAGE} AS runtime
# ffmpeg decodes uploaded audio (avifors/audio.py); tini reaps ffmpeg and workerctl children and
# forwards SIGTERM, so the broker drains and stops its workers on `docker stop`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tini \
    && rm -rf /var/lib/apt/lists/*
# Same IDs as the production host's avifors user (996:990), so bind-mounted stores keep their owner.
RUN groupadd --system --gid 990 avifors \
    && useradd --system --uid 996 --gid 990 --home-dir /var/lib/avifors --no-create-home \
       --shell /usr/sbin/nologin avifors \
    && install -d -m 0750 -o root -g 990 /run/avifors \
    && install -d -m 0750 -o avifors -g avifors /var/lib/avifors /var/lib/avifors/images /var/lib/avifors/audio
COPY --from=build /opt/avifors /opt/avifors
ENV PATH=/opt/avifors/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 996:990
WORKDIR /var/lib/avifors
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["avifors", "--config", "/etc/avifors/config.yaml"]
