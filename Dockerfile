# Base image for autoresearch agent containers.
# Includes: Python, CUDA, common ML packages, and CLI agent tools.
#
# Build:  docker build -t autoresearch/agent:latest .
# Build with specific CUDA version:
#   docker build --build-arg CUDA_VERSION=12.4.0 -t autoresearch/agent:latest .

ARG CUDA_VERSION=12.4.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl wget unzip \
    build-essential cmake \
    texlive-latex-base texlive-latex-extra texlive-fonts-recommended \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ── Base Python packages ──
RUN pip install --no-cache-dir --break-system-packages --upgrade pip && \
    pip install --no-cache-dir --break-system-packages \
    numpy pandas matplotlib scipy scikit-learn \
    torch torchvision torchaudio \
    transformers datasets accelerate \
    huggingface_hub wandb \
    jupyter seaborn plotly

# ── CLI agent tools ──
# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Codex (OpenAI CLI)
RUN npm install -g @openai/codex

# Aider
RUN pip install --no-cache-dir --break-system-packages aider-chat

# ── Workspace setup ──
RUN mkdir -p /workspace
WORKDIR /workspace

# Non-interactive defaults
ENV NONINTERACTIVE=1
ENV CI=1
