"""
app/core/config.py
------------------
Cấu hình ứng dụng sử dụng Pydantic Settings.
Tất cả biến môi trường được load từ file .env.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pathlib import Path

# Xác định đường dẫn đến file .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"



class Settings(BaseSettings):
    """Toàn bộ cấu hình ứng dụng.

    Các giá trị được đọc từ biến môi trường (hoặc file .env).
    Sử dụng ``get_settings()`` thay vì khởi tạo trực tiếp để tận dụng cache.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    secret_key: str = Field(default="", min_length=32)
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]
    api_rate_limit: str = "60/minute"

    # ── LLM ──────────────────────────────────────────────────
    llm_provider: Literal["openai", "gemini"] = "gemini"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # ── Embedding ────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 128

    # ── Vector DB ────────────────────────────────────────────
    vector_db_provider: Literal["qdrant"] = "qdrant"

    ## Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "TYP-legal-rag-chatbot"

    # ── BM25 ─────────────────────────────────────────────────
    bm25_store_path: str = "./data/bm25_index"

    # ── Reranker ─────────────────────────────────────────────
    reranker_provider: Literal["cohere", "bge"] = "bge"
    cohere_api_key: str = Field(default="") # KHÔNG paste key vào đây rồi đẩy lên github
    cohere_reranker_model: str = "rerank-v3.5"
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 20
    rerank_top_n: int = 5

    # ── MongoDB ──────────────────────────────────────────────
    # mongodb_url: str = "mongodb://localhost:27017"
    # mongodb_db_name: str = "rag_chatbot"
    # sessions_collection: str = "sessions"
    # history_max_messages: int = Field(default=20, ge=1) # ghi là Field(20, ge=1) cũng được nhưng IDE sẽ gạch đỏ setting bên dưới dùng không lỗi
    # history_max_tokens: int = Field(default=4096, ge=256)

    # ── Chunking ─────────────────────────────────────────────
    default_chunk_size: int = Field(default=512, ge=64)
    default_chunk_overlap: int = Field(default=50, ge=0)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list[str]) -> list[str]:
        """Chuyển đổi chuỗi comma-separated thành list nếu cần."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Trả về instance Settings được cache (singleton).

    Returns:
        Settings: Instance cấu hình duy nhất cho toàn bộ ứng dụng.
    """
    return Settings()
