from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from app.api.schemas import (
    ChatResponse,
    RankedResult,
    SearchResult,
    SourceResponse,
)
from app.core.logging import get_logger
from app.generation.generator import BaseGenerator, get_generator
from app.retrieval.pre_retrieval import StrategyName, get_pre_retriever
from app.retrieval.reranker import BaseReranker, get_reranker
from app.retrieval.retriever import Retriever, get_retriever

logger = get_logger(__name__)


class ChatbotPipeline:
    """Điều phối toàn bộ luồng RAG của chatbot."""

    def __init__(
        self,
        generator: BaseGenerator | None = None,
        retriever: Retriever | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.generator = generator or get_generator()
        pre_retriever = get_pre_retriever()
        self.retriever = retriever or get_retriever(pre_retriever)
        if self.retriever.pre_retriever is None:
            self.retriever.pre_retriever = pre_retriever
        self.reranker = reranker or get_reranker()

    async def ask(
        self,
        question: str,
        *,
        use_pre_retrieval: bool = True,
        strategy: StrategyName = "expansion",
        use_rerank: bool = True,
        retrieval_top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> ChatResponse:
        """Trả lời câu hỏi bằng retrieval, rerank và generation."""
        try:
            retrieved_results = await self.retriever.retrieve(
                question,
                use_pre_retrieval=use_pre_retrieval,
                strategy=strategy,
                top_k=retrieval_top_k,
            )

            final_results: Sequence[SearchResult] = retrieved_results
            if use_rerank:
                logger.info(
                    "RERANK | enabled=true | provider=%s | query=%r | "
                    "input_count=%d | requested_top_n=%s",
                    self.reranker.__class__.__name__,
                    question,
                    len(retrieved_results),
                    rerank_top_n if rerank_top_n is not None else "default",
                )
                self._log_ranked_results("RERANK BEFORE", retrieved_results)
                final_results = await self.reranker.rerank(
                    question,
                    retrieved_results,
                    top_n=rerank_top_n,
                )
                self._log_ranked_results("RERANK AFTER", final_results)
            else:
                logger.info(
                    "RERANK | enabled=false | query=%r | input_count=%d | "
                    "retrieval_order_retained=true",
                    question,
                    len(retrieved_results),
                )
                self._log_ranked_results(
                    "RERANK OUTPUT (disabled)",
                    retrieved_results,
                )

            answer = await self.generator.generate(question, final_results)
            response = ChatResponse(
                answer=answer,
                sources=self._build_sources(final_results),
            )
            logger.info(
                "Pipeline đã xử lý câu hỏi với %d nguồn",
                len(response.sources),
            )
            return response
        except Exception:
            logger.exception("Lỗi khi xử lý câu hỏi trong pipeline")
            raise

    @staticmethod
    def _log_ranked_results(
        stage: str,
        results: Sequence[SearchResult],
    ) -> None:
        """Log thứ tự, điểm và nội dung rút gọn để so sánh reranking."""
        if not results:
            logger.info("%s | result_count=0", stage)
            return

        for position, result in enumerate(results, start=1):
            metadata = result.metadata
            preview = " ".join(result.content.split())
            if len(preview) > 180:
                preview = f"{preview[:177]}..."

            if isinstance(result, RankedResult):
                logger.info(
                    "%s [%02d] | chunk_id=%s | title=%r | dieu=%r | "
                    "retrieval_score=%.6f | rerank_score=%.6f | "
                    "original_rank=%d | preview=%r",
                    stage,
                    position,
                    result.chunk_id,
                    metadata.title,
                    metadata.dieu,
                    result.score,
                    result.rerank_score,
                    result.original_rank,
                    preview,
                )
            else:
                logger.info(
                    "%s [%02d] | chunk_id=%s | title=%r | dieu=%r | "
                    "retrieval_score=%.6f | preview=%r",
                    stage,
                    position,
                    result.chunk_id,
                    metadata.title,
                    metadata.dieu,
                    result.score,
                    preview,
                )

    @staticmethod
    def _build_sources(
        results: Sequence[SearchResult],
    ) -> list[SourceResponse]:
        return [
            SourceResponse(
                chunk_id=result.chunk_id,
                content=result.content,
                chunk_index=result.chunk_index,
                metadata=result.metadata,
                score=(
                    result.rerank_score
                    if isinstance(result, RankedResult)
                    else result.score
                ),
                score_type=("rerank" if isinstance(result, RankedResult) else "rrf"),
            )
            for result in results
        ]


# Singleton Pattern
_pipeline: ChatbotPipeline | None = None
_pipeline_lock = Lock()


def get_pipeline() -> ChatbotPipeline:
    """Trả về singleton chatbot pipeline."""
    global _pipeline

    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                try:
                    _pipeline = ChatbotPipeline()
                    logger.info("Đã khởi tạo ChatbotPipeline")
                except Exception:
                    logger.exception("Lỗi khi khởi tạo ChatbotPipeline")
                    raise

    return _pipeline


if __name__ == "__main__":
    import asyncio

    async def main():
        pipeline = ChatbotPipeline()
        response = await pipeline.ask("Luật Lao động là gì?")
        print(response.answer)

    asyncio.run(main())
