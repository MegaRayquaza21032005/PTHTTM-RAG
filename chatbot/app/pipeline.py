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
        pre_retriever = get_pre_retriever(self.generator.llm)
        self.retriever = retriever or get_retriever(pre_retriever)
        if self.retriever.pre_retriever is None:
            self.retriever.pre_retriever = pre_retriever
        self.reranker = reranker or get_reranker()

    async def ask(
        self,
        question: str,
        *,
        use_pre_retrieval: bool = False,
        strategy: StrategyName = "rewrite",
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
                final_results = await self.reranker.rerank(
                    question,
                    retrieved_results,
                    top_n=rerank_top_n,
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
