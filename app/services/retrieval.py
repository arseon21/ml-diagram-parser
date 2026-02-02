from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = BASE_DIR / "data" / "index.npy"
DEFAULT_PATHS_PATH = BASE_DIR / "data" / "paths.json"


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


class VectorStore:
    def __init__(self, device: str = "cpu", index_path: Path = DEFAULT_INDEX_PATH, paths_path: Path = DEFAULT_PATHS_PATH) -> None:
        self.device = device
        self.model = SentenceTransformer("sentence-transformers/clip-ViT-B-32", device=device)
        self.index_path = index_path
        self.paths_path = paths_path
        self._embeddings: torch.Tensor | None = None
        self._paths: List[str] = []
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists() or not self.paths_path.exists():
            _log("Index файл отсутствует")
            return

        try:
            embeddings_np = np.load(self.index_path)
            self._embeddings = torch.from_numpy(embeddings_np).to(self.device)
            with self.paths_path.open("r", encoding="utf-8") as f:
                self._paths = json.load(f)
            _log(f"Загружено индексов: {len(self._paths)}")
        except Exception as exc:
            _log(f"Ошибка загрузки: {exc}")
            self._embeddings = None
            self._paths = []

    def search(self, query_image: Image.Image, k: int = 2) -> List[str]:
        if self._embeddings is None or not self._paths:
            _log("Пустой индекс")
            return []

        image = query_image.convert("RGB")
        query_emb = self.model.encode(images=[image], convert_to_tensor=True)
        query_emb = _normalize(query_emb)

        embeddings = self._embeddings
        if embeddings.device != query_emb.device:
            embeddings = embeddings.to(query_emb.device)
        embeddings = _normalize(embeddings)

        scores = util.cos_sim(query_emb, embeddings)
        top_k = min(k, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)

        results: List[str] = []
        for idx in top_indices[0].tolist():
            try:
                txt_path = Path(self._paths[idx])
                text = txt_path.read_text(encoding="utf-8")
                results.append(text)
            except Exception as exc:
                _log(f"Ошибка чтения {self._paths[idx]}: {exc}")

        return results
