FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
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

WORKDIR /opt

RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5

WORKDIR /opt/ACE-Step-1.5

RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch CUDA wheels first.
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Try project install. If repo packaging changes, this layer is the first thing we'll adjust.
RUN if [ -f requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt; fi
RUN pip3 install --no-cache-dir -e .

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
