# syntax=docker/dockerfile:1
#
# Multi-stage build for the Prism challenge with two independently buildable
# targets (mirrors the agent-challenge pattern):
#
#   docker build --target service   -t prism-svc  .   # uvicorn API on :8080
#   docker build --target evaluator -t prism-eval .   # CUDA cu128 torchrun runner
#
# A plain `docker build .` (no --target) yields the `service` image, preserving
# the previous single-image consumer (uvicorn app on port 8080).
#
# NOTE: every stage that runs `pip install .` MUST have `git` installed, because
# pyproject.toml pulls `platform-network @ git+https://github.com/PlatformNetwork/platform.git`.

############################################################
# evaluator target — CUDA-capable image (cu128 series) that
# runs the torchrun runner. Matches the proven local image
# prism-evaluator:smoke-local-cu128-nonroot (cu128, non-root).
############################################################
ARG CUDA_BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_BASE} AS evaluator

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# torch CUDA channel: cu128 wheels carry their own CUDA 12.8 runtime libs.
ARG TORCH_CUDA_CHANNEL=cu128

WORKDIR /workspace

# python3.12 ships with ubuntu24.04; git is required for the git+https dep clone.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (ubuntu24.04 python is externally-managed).
RUN python3 -m venv "$VIRTUAL_ENV"

COPY pyproject.toml ./
COPY src ./src

# Install the CUDA-enabled torch from the cu128 channel FIRST, so the package
# install below resolves `torch>=2.3` against the GPU build (not the CPU wheel),
# then install the package (brings numpy, platform-network and runner deps).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
    && pip install --no-cache-dir .

# Non-root runtime user.
RUN useradd --create-home --shell /usr/sbin/nologin prism \
    && mkdir -p /workspace /artifacts \
    && chown -R prism:prism /workspace /artifacts /opt/venv

USER prism
ENV HOME=/home/prism

# The runner.py + payload.json are mounted into /workspace at run time and driven
# with: torchrun --standalone --nnodes=1 --nproc-per-node=N \
#                 /workspace/runner.py /workspace/payload.json
# (see src/prism_challenge/evaluator/container.py::_runner_launch_command).
CMD ["python", "-c", "import torch; print('prism-evaluator ready: torch', torch.__version__, 'cuda', torch.version.cuda)"]


############################################################
# service target — uvicorn API on :8080 (non-root prism).
# This is the final stage, so `docker build .` (no --target)
# reproduces the previous single-image service behavior.
############################################################
FROM python:3.12-slim AS service

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# git: required for the `platform-network @ git+https://...` dependency clone.
# docker-cli: the service shells out to the docker broker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git docker-cli \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --create-home --shell /usr/sbin/nologin prism \
    && mkdir -p /data \
    && chown -R prism:prism /app /data

USER prism
ENV HOME=/home/prism

EXPOSE 8080

CMD ["uvicorn", "prism_challenge.app:app", "--host", "0.0.0.0", "--port", "8080"]
