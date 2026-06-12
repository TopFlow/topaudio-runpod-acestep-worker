FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_BUILD_ISOLATION=1

ENV HF_HOME=/runpod-volume/topaudio_ai/acestep_hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/topaudio_ai/acestep_hf_cache
ENV TORCH_HOME=/runpod-volume/topaudio_ai/acestep_torch_cache
ENV TOPAUDIO_BASE_DIR=/runpod-volume/topaudio_ai
ENV ACESTEP_REPO_DIR=/opt/ACE-Step-1.5
ENV ACESTEP_HOST=127.0.0.1
ENV ACESTEP_PORT=8001

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git ffmpeg libsndfile1 curl wget build-essential \
    python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN python3.11 -m pip install --upgrade pip setuptools wheel packaging ninja hatchling

# CUDA PyTorch first
RUN python3.11 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Base deps for ACE-Step repo + worker.
# We intentionally skip flash-attn and pinned torch cu128/cu130.
RUN python3.11 -m pip install --no-cache-dir \
    runpod==1.7.13 \
    requests \
    pandas \
    soundfile \
    librosa \
    tqdm \
    huggingface_hub \
    accelerate \
    "transformers>=4.51.0,<4.58.0" \
    "diffusers>=0.37.0" \
    peft \
    safetensors \
    einops \
    numpy \
    scipy \
    gradio==6.2.0 \
    fastapi \
    "uvicorn[standard]" \
    loguru \
    toml \
    lightning \
    tensorboard \
    vector-quantize-pytorch \
    lycoris-lora \
    modelscope \
    diskcache \
    typer-slim \
    pywavelets \
    pytorch-wavelets

WORKDIR /opt

RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5

WORKDIR /opt/ACE-Step-1.5

# Install ACE-Step package entrypoints without pulling its heavy/pinned deps.
RUN python3.11 -m pip install --no-cache-dir -e . --no-deps

WORKDIR /app

COPY handler.py /app/handler.py

CMD ["python3.11", "-u", "/app/handler.py"]
