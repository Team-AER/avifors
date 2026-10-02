# syntax=docker/dockerfile:1
# FLUX.2 image worker: stable-diffusion.cpp sd-server with CUDA for the RTX 4060 Ti (SM 89).
# Build from the repository root (compiles CUDA kernels; build on an x86-64 host):
#   docker build -f deploy/docker/avifors-image.Dockerfile -t avifors-image:local .
# Same engine revision and CMake flags as production (FLUX2.md); only sd-server reaches the
# final image. The host driver is injected by the NVIDIA container runtime.
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.9.1-runtime-ubuntu24.04

FROM ${CUDA_DEVEL_IMAGE} AS build
ARG SD_CPP_REPO=https://github.com/leejet/stable-diffusion.cpp.git
ARG SD_CPP_COMMIT=36746936c054d889e9b3c0e6ee490a9e362a0808
ARG CUDA_ARCHITECTURES=89
RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q . \
    && git remote add origin "${SD_CPP_REPO}" \
    && git fetch -q --depth 1 origin "${SD_CPP_COMMIT}" \
    && git checkout -q FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${SD_CPP_COMMIT}" \
    && git submodule update -q --init --recursive
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DGGML_CUDA_FA=ON \
    && cmake --build build --config Release -j"$(nproc)" \
    && install -m 0755 build/bin/sd-server /usr/local/bin/sd-server

FROM ${CUDA_RUNTIME_IMAGE} AS runtime
# ggml's CPU backend (used by --offload-to-cpu) links OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 990 avifors \
    && useradd --system --uid 996 --gid 990 --home-dir /var/lib/avifors --no-create-home \
       --shell /usr/sbin/nologin avifors
COPY --from=build /usr/local/bin/sd-server /usr/local/bin/sd-server
# Fail the build, not the first request, if sd-server needs a library the runtime image lacks
# (libcuda comes from the host driver at run time).
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN missing="$(ldd /usr/local/bin/sd-server | grep 'not found' | grep -v 'libcuda\.so' || true)" \
    && if [ -n "$missing" ]; then echo "$missing" >&2; exit 1; fi
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 996:990
WORKDIR /tmp
ENTRYPOINT ["/usr/local/bin/sd-server"]
