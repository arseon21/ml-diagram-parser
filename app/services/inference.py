from __future__ import annotations

import asyncio
import io
from datetime import datetime
from functools import lru_cache
from typing import Dict

from PIL import Image

from app.core.model import QwenEngine
from app.services.retrieval import VectorStore


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


def _resize_if_needed(image: Image.Image, max_size: int = 768) -> Image.Image:
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    scale = min(max_size / width, max_size / height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStore:
    return VectorStore(device="cpu")


@lru_cache(maxsize=1)
def _get_qwen_engine() -> QwenEngine:
    return QwenEngine()


async def process_diagram(image_bytes: bytes) -> Dict[str, object]:
    _log(f"Старт обработки диаграммы: bytes={len(image_bytes)}")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = _resize_if_needed(image)
    _log(f"Изображение подготовлено: size={image.size}")

    vector_store = _get_vector_store()
    examples = vector_store.search(image, k=2)
    _log(f"Найдено примеров: {len(examples)}")

    qwen = _get_qwen_engine()
    description = await asyncio.to_thread(qwen.generate_description, image, examples)
    _log("Генерация описания завершена")

    return {"description": description, "examples_used": len(examples)}
