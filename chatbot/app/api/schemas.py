"""Pydantic schemas used by the chatbot API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """Shared validation rules for public API schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ChatRequest(APIModel):
    """Payload accepted by the chat endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4_096,
        description="Cau hoi cua nguoi dung.",
        examples=["Người lao động được nghỉ phép năm bao nhiêu ngày?"],
    )


class MetaData(APIModel):
    """Information that identifies the source of a document chunk."""

    document_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    source: str | None = Field(
        default=None,
        description="Ten hoac duong dan tai lieu nguon.",
    )
    chuong: str | None = None
    muc: str | None = None
    dieu: str | None = None


class Chunk(APIModel):
    """A text chunk created from a source document."""

    chunk_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1, description="Noi dung doan tai lieu.")
    chunk_index: int = Field(..., ge=0)
    metadata: MetaData

class SearchResult(Chunk):
    """Kết quả sau khi hợp nhất dense search và sparse search."""

    score: float
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)


class RankedResult(SearchResult):
    """Kết quả sau khi rerank các kết quả hybrid search."""

    rerank_score: float
    original_rank: int = Field(ge=1)


class SourceResponse(Chunk):
    """A retrieved chunk used to support the generated answer."""

    score: float | None = Field(
        default=None,
        description="Diem lien quan sau retrieval hoac reranking.",
    )


class ChatResponse(APIModel):
    """Response returned after the RAG pipeline answers a question."""

    answer: str = Field(..., min_length=1, description="Cau tra loi cua chatbot.")
    sources: list[SourceResponse] = Field(default_factory=list)


class HealthResponse(APIModel):
    """Trạng thái hoạt động của API."""

    status: Literal["ok"] = "ok"


class ErrorResponse(APIModel):
    """Consistent error body for documented API failures."""

    detail: str = Field(..., min_length=1)
