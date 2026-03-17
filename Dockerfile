# Base image for researcharena agent containers.
# Uses the official PyTorch image (has Python, CUDA, pip pre-installed).
# Only pip installs — no apt-get — so it works with rootless podman.
#
# CLI agent tools (claude, codex, etc.) are mounted from the host at runtime
# by agent_runner.py, so they don't need to be installed here.
#
# Build:
#   podman build --userns=host -t researcharena/agent:latest .
#   # or: docker build -t researcharena/agent:latest .

FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1

# ── Python ML packages (beyond what pytorch image provides) ──
RUN pip install --no-cache-dir \
    torchvision torchaudio \
    transformers datasets accelerate \
    huggingface_hub wandb \
    scikit-learn scipy pandas matplotlib seaborn plotly

# ── Workspace setup ──
RUN mkdir -p /workspace
WORKDIR /workspace

# Non-interactive defaults
ENV NONINTERACTIVE=1
ENV CI=1
