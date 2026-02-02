import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "dataset"


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


def _collect_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    images = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))
    pairs: List[Tuple[Path, Path]] = []
    for img_path in images:
        txt_path = img_path.with_suffix(".txt")
        if img_path.stem != txt_path.stem or not txt_path.exists():
            _log(f"Нет описания для {img_path}")
            continue
        pairs.append((img_path, txt_path))
    return pairs


def _split_pairs(pairs: List[Tuple[Path, Path]]) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    pairs.sort (key=lambda p: p[0].name)
    random.seed(42)
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * 0.6)
    val_end = train_end + int(total * 0.2)
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    return train_pairs, val_pairs, test_pairs


def _copy_split(split_name: str, pairs: List[Tuple[Path, Path]]) -> None:
    img_dir = OUT_DIR / split_name / "images"
    txt_dir = OUT_DIR / split_name / "texts"
    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    for img_path, txt_path in pairs:
        try:
            shutil.copy(img_path, img_dir / img_path.name)
            shutil.copy(txt_path, txt_dir / txt_path.name)
        except Exception as exc:
            _log(f"Ошибка копирования {img_path}: {exc}")


def main() -> None:
    if OUT_DIR.exists():
        _log(f"Очистка:{OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    pairs = _collect_pairs(RAW_DIR)
    train_pairs, val_pairs, test_pairs = _split_pairs(pairs)

    _copy_split("train", train_pairs)
    _copy_split("val", val_pairs)
    _copy_split("test", test_pairs)

    _log(f"Пары примеров: {len(train_pairs)}")
    _log(f"Пары для валидации: {len(val_pairs)}")
    _log(f"Пары для тестов: {len(test_pairs)}")


if __name__ == "__main__":
    main()
