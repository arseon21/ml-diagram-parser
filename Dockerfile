FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/dataset/train/images data/dataset/train/texts \
    data/dataset/val/images data/dataset/val/texts \
    data/dataset/test/images data/dataset/test/texts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
