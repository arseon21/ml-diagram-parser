from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI

from app.api.routes import router
from app.services.inference import _get_qwen_engine, _get_vector_store


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log("Инициализация VectorStore и QwenEngine.")
    _get_vector_store()
    _get_qwen_engine()
    _log("Завершено.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
