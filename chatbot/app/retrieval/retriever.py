from __future__ import annotations

import asyncio
from threading import Lock

from app.api.schemas import SearchResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.hybrid_searcher import HybridSearcher, get_hybrid_searcher
from app.retrieval.pre_retrieval import PreRetriever, StrategyName


logger = get_logger(__name__)
settings = get_settings()


class Retriever:
    """Điều phối pre-retrieval và hybrid search."""

    def __init__(
        self,
        hybrid_searcher: HybridSearcher | None = None,
        pre_retriever: PreRetriever | None = None,
    ) -> None:
        self.hybrid_searcher = hybrid_searcher or get_hybrid_searcher()
        self.pre_retriever = pre_retriever

    async def retrieve(
        self,
        query: str,
        *,
        use_pre_retrieval: bool = False,
        strategy: StrategyName = "rewrite",
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Truy xuất các chunk liên quan nhất cho một câu hỏi."""
        result_count = top_k or settings.rerank_top_k

        try:
            queries = await self._prepare_queries(
                query,
                use_pre_retrieval,
                strategy,
            )
            result_groups = await asyncio.gather(
                *(
                    self.hybrid_searcher.search(item, top_k=result_count)
                    for item in queries
                )
            )
            results = self._merge_results(result_groups, result_count)

            logger.info(
                "Retriever trả về %d kết quả từ %d truy vấn",
                len(results),
                len(queries),
            )
            return results
        except Exception:
            logger.exception("Lỗi khi truy xuất tài liệu cho câu hỏi: %s", query)
            raise

    async def _prepare_queries(
        self,
        query: str,
        use_pre_retrieval: bool,
        strategy: StrategyName,
    ) -> list[str]:
        if not use_pre_retrieval:
            return [query]
        if self.pre_retriever is None:
            raise RuntimeError("PreRetriever chưa được khởi tạo")
        return await self.pre_retriever.process(query, strategy)

    @staticmethod
    def _merge_results(
        result_groups: list[list[SearchResult]],
        top_k: int,
    ) -> list[SearchResult]:
        best_results: dict[str, SearchResult] = {}

        for results in result_groups:
            for result in results:
                current = best_results.get(result.chunk_id)
                if current is None or result.score > current.score:
                    best_results[result.chunk_id] = result

        return sorted(
            best_results.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]


# Singleton Pattern 
_retriever: Retriever | None = None
_retriever_lock = Lock()


def get_retriever(
    pre_retriever: PreRetriever | None = None,
) -> Retriever:
    """Trả về singleton retriever."""
    global _retriever

    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever(pre_retriever=pre_retriever)
                logger.info("Đã khởi tạo Retriever")
    elif pre_retriever is not None and _retriever.pre_retriever is None:
        _retriever.pre_retriever = pre_retriever

    return _retriever


if __name__ == "__main__":
    import asyncio
    async def main():
        retriever = Retriever()
        results = await retriever.retrieve(query="Luật Lao động là gì?", 
                                           use_pre_retrieval=False)
        for result in results:
            print(result)
            print("\n\n")
    asyncio.run(main())
    
    