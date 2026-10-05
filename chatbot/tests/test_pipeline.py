from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from app.api.schemas import MetaData, RankedResult, SearchResult
from app.pipeline import ChatbotPipeline


class ChatbotPipelineTest(IsolatedAsyncioTestCase):
    async def test_ask_runs_the_complete_pipeline(self) -> None:
        question = "Người lao động được nghỉ phép năm bao nhiêu ngày?"
        metadata = MetaData(
            document_id="bo-luat-lao-dong",
            title="Bộ luật Lao động",
            chuong="Chương VII",
            dieu="Điều 113",
        )
        search_result = SearchResult(
            chunk_id="chunk-1",
            content="Người lao động được nghỉ hằng năm theo quy định.",
            chunk_index=0,
            metadata=metadata,
            score=0.02,
            dense_rank=1,
            sparse_rank=2,
        )
        ranked_result = RankedResult(
            **search_result.model_dump(),
            rerank_score=0.95,
            original_rank=1,
        )

        retriever = SimpleNamespace(
            pre_retriever=object(),
            retrieve=AsyncMock(return_value=[search_result]),
        )
        reranker = SimpleNamespace(
            rerank=AsyncMock(return_value=[ranked_result]),
        )
        generator = SimpleNamespace(
            generate=AsyncMock(return_value="Người lao động được nghỉ phép năm."),
        )

        with patch("app.pipeline.get_pre_retriever"):
            pipeline = ChatbotPipeline(
                generator=generator,
                retriever=retriever,
                reranker=reranker,
            )

        response = await pipeline.ask(
            question,
            use_pre_retrieval=True,
            strategy="rewrite",
            retrieval_top_k=20,
            rerank_top_n=5,
        )

        retriever.retrieve.assert_awaited_once_with(
            question,
            use_pre_retrieval=True,
            strategy="rewrite",
            top_k=20,
        )
        reranker.rerank.assert_awaited_once_with(
            question,
            [search_result],
            top_n=5,
        )
        generator.generate.assert_awaited_once_with(question, [ranked_result])

        self.assertEqual(response.answer, "Người lao động được nghỉ phép năm.")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].chunk_id, "chunk-1")
        self.assertEqual(response.sources[0].score, 0.95)


if __name__ == "__main__":
    import unittest

    unittest.main()
