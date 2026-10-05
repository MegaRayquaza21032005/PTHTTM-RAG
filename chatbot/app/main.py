from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline import get_pipeline


logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Khởi tạo và giải phóng các tài nguyên dùng chung của ứng dụng."""
    logger.info("Đang khởi tạo chatbot pipeline")
    pipeline = await asyncio.to_thread(get_pipeline)
    app.state.pipeline = pipeline
    logger.info("Chatbot API đã sẵn sàng")

    try:
        yield
    finally:
        await pipeline.retriever.hybrid_searcher.vector_db.close()
        logger.info("Chatbot API đã dừng")


def create_app() -> FastAPI:
    """Tạo FastAPI application."""
    application = FastAPI(
        title="Vietnamese Legal RAG Chatbot",
        version="1.0.0",
        debug=settings.app_env == "development",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router, prefix="/api")
    return application


app = create_app()
