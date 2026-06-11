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
    python3 python3-pip python3-venv git ffmpeg libsndfile1 curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel packaging ninja

# CUDA PyTorch first
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Base deps for ACE-Step repo inspection + worker.
# Do NOT install ACE-Step requirements.txt yet because it pulls flash-attn and newer CUDA packages.
RUN pip3 install --no-cache-dir \
    runpod==1.7.13 \
    requests \
    pandas \
    soundfile \
    librosa \
    tqdm \
    huggingface_hub \
    accelerate \
    transformers \
    diffusers \
    peft \
    safetensors \
    einops \
    numpy \
    scipy \
    gradio \
    fastapi \
    uvicorn \
    loguru \
    toml \
    lightning \
    tensorboard \
    vector-quantize-pytorch \
    lycoris-lora \
    modelscope

WORKDIR /opt

RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5

WORKDIR /opt/ACE-Step-1.5

# Install repo package without dependency resolution.
# We will add missing deps from runtime logs, not blindly from requirements.txt.
RUN pip3 install --no-cache-dir -e . --no-deps || true

WORKDIR /app

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
