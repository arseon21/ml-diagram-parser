FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/dataset/train/images data/dataset/train/texts \
    data/dataset/val/images data/dataset/val/texts \
    data/dataset/test/images data/dataset/test/texts

EXPOSE 8000

CMD python -m scripts.split_data && python -m scripts.build_index && uvicorn app.main:app --host 0.0.0.0 --port 8000
