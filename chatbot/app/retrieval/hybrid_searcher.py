from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any

from app.api.schemas import MetaData, SearchResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.database.bm25_store import BM25Store, get_bm25_store
from app.database.vector_db import VectorDBClient, get_vector_db
from app.indexing.embedding import Embedder, get_embedder


logger = get_logger(__name__)
settings = get_settings()


class HybridSearcher:
    """Kết hợp dense search và BM25 bằng Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_db: VectorDBClient | None = None,
        bm25_store: BM25Store | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.vector_db = vector_db or get_vector_db()
        self.bm25_store = bm25_store or get_bm25_store()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Tìm kiếm đồng thời bằng dense vector và BM25 rồi hợp nhất."""
        result_count = top_k or settings.rerank_top_k

        dense_results, sparse_results = await asyncio.gather(
            self._dense_search(query, result_count),
            self._sparse_search(query, result_count),
        )

        try:
            results = self._fuse(
                dense_results,
                sparse_results,
                top_k=result_count,
            )

            logger.info(
                "Hybrid search trả về %d kết quả cho truy vấn: %s",
                len(results),
                query,
            )
            return results
        except Exception:
            logger.exception("Lỗi khi hợp nhất kết quả dense và sparse search")
            raise

    async def _dense_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Tạo query embedding và tìm kiếm vector trên Qdrant."""
        try:
            query_vector = await self.embedder.embed_text(query)
            results = await self.vector_db.query(query_vector, top_k=top_k)
            logger.debug("Dense search trả về %d kết quả", len(results))
            return results
        except Exception:
            logger.exception("Lỗi tại nhánh dense search trên Qdrant")
            raise

    async def _sparse_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, object]]:
        """Tìm kiếm từ khóa trên BM25."""
        try:
            results = await self.bm25_store.search(query, top_k=top_k)
            logger.debug("Sparse search trả về %d kết quả", len(results))
            return results
        except Exception:
            logger.exception("Lỗi tại nhánh sparse search bằng BM25")
            raise

    def _fuse(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, object]],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        merged: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk_id = str(result["id"])
            merged[chunk_id] = {
                "payload": result["payload"],
                "score": self.dense_weight / (self.rrf_k + rank),
                "dense_rank": rank,
                "sparse_rank": None,
            }

        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = str(result["id"])
            rrf_score = self.sparse_weight / (self.rrf_k + rank)

            if chunk_id in merged:
                merged[chunk_id]["score"] += rrf_score
                merged[chunk_id]["sparse_rank"] = rank
            else:
                merged[chunk_id] = {
                    "payload": result["payload"],
                    "score": rrf_score,
                    "dense_rank": None,
                    "sparse_rank": rank,
                }

        ranked_items = sorted(
            merged.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:top_k]

        return [
            self._to_search_result(chunk_id, data)
            for chunk_id, data in ranked_items
        ]

    @staticmethod
    def _to_search_result(
        chunk_id: str,
        data: dict[str, Any],
    ) -> SearchResult:
        payload = data["payload"]
        return SearchResult(
            chunk_id=chunk_id,
            content=payload["content"],
            chunk_index=payload["chunk_index"],
            metadata=MetaData.model_validate(payload["metadata"]),
            score=data["score"],
            dense_rank=data["dense_rank"],
            sparse_rank=data["sparse_rank"],
        )


# Singleton Pattern
_hybrid_searcher: HybridSearcher | None = None
_hybrid_searcher_lock = Lock()


def get_hybrid_searcher() -> HybridSearcher:
    """Trả về singleton hybrid searcher."""
    global _hybrid_searcher

    if _hybrid_searcher is None:
        with _hybrid_searcher_lock:
            if _hybrid_searcher is None:
                try:
                    _hybrid_searcher = HybridSearcher()
                    logger.info("Đã khởi tạo HybridSearcher")
                except Exception:
                    logger.exception("Lỗi khi khởi tạo HybridSearcher")
                    raise

    return _hybrid_searcher
