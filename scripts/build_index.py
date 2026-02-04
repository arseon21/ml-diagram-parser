import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from app.services.retrieval import VectorStore

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "data" / "dataset" / "train"
IMAGES_DIR = DATASET_DIR / "images"
TEXTS_DIR = DATASET_DIR / "texts"

INDEX_PATH = BASE_DIR / "data" / "index.npy"
PATHS_PATH = BASE_DIR / "data" / "paths.json"


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def main() -> None:
    _log("Загрузка VectorStore...")
    vector_store = VectorStore(device="cpu")
    model = vector_store.model
    _log("Модель загружена")

    _log(f"Сканирование изображения {IMAGES_DIR}...")
    image_paths = list(IMAGES_DIR.rglob("*.jpg")) + list(IMAGES_DIR.rglob("*.png"))
    _log(f"Найдено {len(image_paths)} изображения")
    
    embeddings: List[torch.Tensor] = []
    text_paths: List[str] = []

    for idx, img_path in enumerate(image_paths, 1):
        txt_path = TEXTS_DIR / f"{img_path.stem}.txt"
        if not txt_path.exists():
            _log(f"Нет текста для {img_path.name}")
            continue
        
        try:
            image = Image.open(img_path).convert("RGB")
            emb = model.encode(
                sentences=[image],
                convert_to_tensor=True,
                batch_size=1
            )
            
            embeddings.append(emb)
            text_paths.append(txt_path.as_posix())
            
            if idx % 10 == 0:
                _log(f"{idx}/{len(image_paths)}")
                
        except Exception as exc:
            _log(f"Ошибка кодирования {img_path.name}: {exc}")

    if not embeddings:
        _log("Индекс не сохранён.")
        return

    emb_tensor = torch.cat(embeddings, dim=0)
    emb_tensor = _normalize(emb_tensor)

    _log("Сохранение индексов.")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_PATH, emb_tensor.cpu().numpy())
    PATHS_PATH.write_text(json.dumps(text_paths, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(f"Сохранено {len(text_paths)} индексов")


if __name__ == "__main__":
    main()