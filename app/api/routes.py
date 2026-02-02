import time
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile

from app.services.inference import process_diagram

router = APIRouter()


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


@router.post("/analyze")
async def analyze(file: UploadFile) -> dict:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Не поддерживаемый формат файла")

    start = time.time()
    image_bytes = await file.read()
    result = await process_diagram(image_bytes)
    elapsed = time.time() - start
    _log(f"Анализ завершен за: {elapsed:.3f}s")
    return result
