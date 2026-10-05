from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from threading import Lock

from sentence_transformers import CrossEncoder

from app.api.schemas import RankedResult, SearchResult
from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class BaseReranker(ABC):
    """Interface chung cho các reranker."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[RankedResult]:
        """Sắp xếp lại kết quả retrieval theo độ liên quan."""
        ...

    @staticmethod
    def _to_ranked_result(
        result: SearchResult,
        rerank_score: float,
        original_rank: int,
    ) -> RankedResult:
        return RankedResult(
            **result.model_dump(),
            rerank_score=rerank_score,
            original_rank=original_rank,
        )


class CohereReranker(BaseReranker):
    """Rerank kết quả bằng Cohere Rerank API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        import cohere

        self.model = model or settings.cohere_reranker_model
        self.client = cohere.ClientV2(api_key=api_key or settings.cohere_api_key)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[RankedResult]:
        if not results:
            return []

        limit = min(top_n or settings.rerank_top_n, len(results))
        try:
            response = await asyncio.to_thread(
                self.client.rerank,
                model=self.model,
                query=query,
                documents=[result.content for result in results],
                top_n=limit,
            )
            ranked_results = [
                self._to_ranked_result(
                    results[item.index],
                    float(item.relevance_score),
                    item.index + 1,
                )
                for item in response.results
            ]
            logger.info("Cohere đã rerank %d kết quả", len(ranked_results))
            return ranked_results
        except Exception:
            logger.exception("Lỗi khi rerank bằng Cohere")
            raise


class BGEReranker(BaseReranker):
    """Rerank cục bộ bằng BGE M3 CrossEncoder."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model or settings.bge_reranker_model
        self.batch_size = batch_size
        self.model = CrossEncoder(self.model_name)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[RankedResult]:
        if not results:
            return []

        limit = min(top_n or settings.rerank_top_n, len(results))
        pairs = [(query, result.content) for result in results]

        try:
            scores = await asyncio.to_thread(
                self.model.predict,
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            ranked_indices = sorted(
                range(len(results)),
                key=lambda index: float(scores[index]),
                reverse=True,
            )[:limit]
            ranked_results = [
                self._to_ranked_result(
                    results[index],
                    float(scores[index]),
                    index + 1,
                )
                for index in ranked_indices
            ]
            logger.info("BGE đã rerank %d kết quả", len(ranked_results))
            return ranked_results
        except Exception:
            logger.exception("Lỗi khi rerank bằng BGE")
            raise


# Singleton Pattern
_reranker: BaseReranker | None = None
_reranker_lock = Lock()


def get_reranker() -> BaseReranker:
    """Trả về singleton reranker theo provider trong cấu hình."""
    global _reranker

    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                try:
                    if settings.reranker_provider == "cohere":
                        _reranker = CohereReranker()
                    else:
                        _reranker = BGEReranker()
                    logger.info(
                        "Đã khởi tạo reranker | provider=%s",
                        settings.reranker_provider,
                    )
                except Exception:
                    logger.exception("Lỗi khi khởi tạo reranker")
                    raise

    return _reranker
