from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from app.api.schemas import MetaData, RankedResult, SearchResult
from app.pipeline import ChatbotPipeline
from app.retrieval.pre_retrieval import PreRetriever
from app.retrieval.retriever import Retriever


def rendered_log_messages(mock_logger: object) -> str:
    messages: list[str] = []
    for call in mock_logger.info.call_args_list:  # type: ignore[attr-defined]
        template, *args = call.args
        messages.append(template % tuple(args) if args else template)
    return "\n".join(messages)


def make_results() -> tuple[SearchResult, RankedResult]:
    metadata = MetaData(
        document_id="bo-luat-lao-dong",
        title="Bộ luật Lao động",
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
    return search_result, ranked_result


class PreRetrievalObservabilityTest(IsolatedAsyncioTestCase):
    async def test_all_strategies_log_original_and_transformed_queries(self) -> None:
        cases = {
            "rewrite": ["Quy định nghỉ hằng năm của người lao động"],
            "decomposition": ["Có bao nhiêu ngày nghỉ?", "Điều kiện hưởng là gì?"],
            "expansion": ["Nghỉ phép năm", "Nghỉ hằng năm", "Ngày phép lao động"],
        }

        for strategy, transformed in cases.items():
            with self.subTest(strategy=strategy):
                pre_retriever = PreRetriever.__new__(PreRetriever)
                pre_retriever.strategies = {
                    strategy: SimpleNamespace(
                        transform=AsyncMock(return_value=transformed),
                    )
                }

                with patch("app.retrieval.pre_retrieval.logger") as logger:
                    result = await pre_retriever.process(
                        "Người lao động được nghỉ phép thế nào?",
                        strategy=strategy,
                    )

                logs = rendered_log_messages(logger)
                self.assertEqual(result, transformed)
                self.assertIn(
                    f"PRE-RETRIEVAL | enabled=true | strategy={strategy}",
                    logs,
                )
                self.assertIn("Người lao động được nghỉ phép thế nào?", logs)
                for query in transformed:
                    self.assertIn(query, logs)

    async def test_disabled_pre_retrieval_logs_passthrough_query(self) -> None:
        retriever = Retriever.__new__(Retriever)
        retriever.pre_retriever = None

        with patch("app.retrieval.retriever.logger") as logger:
            result = await retriever._prepare_queries(
                "Câu hỏi gốc",
                use_pre_retrieval=False,
                strategy="rewrite",
            )

        logs = rendered_log_messages(logger)
        self.assertEqual(result, ["Câu hỏi gốc"])
        self.assertIn("PRE-RETRIEVAL | enabled=false", logs)
        self.assertIn("Câu hỏi gốc", logs)


class RerankObservabilityTest(IsolatedAsyncioTestCase):
    async def test_rerank_enabled_logs_before_and_after_order(self) -> None:
        search_result, ranked_result = make_results()
        retriever = SimpleNamespace(
            pre_retriever=object(),
            retrieve=AsyncMock(return_value=[search_result]),
        )
        reranker = SimpleNamespace(
            rerank=AsyncMock(return_value=[ranked_result]),
        )
        generator = SimpleNamespace(generate=AsyncMock(return_value="Câu trả lời"))

        with patch("app.pipeline.get_pre_retriever"):
            pipeline = ChatbotPipeline(generator, retriever, reranker)

        with patch("app.pipeline.logger") as logger:
            await pipeline.ask("Câu hỏi", use_rerank=True, rerank_top_n=5)

        logs = rendered_log_messages(logger)
        self.assertIn("RERANK | enabled=true", logs)
        self.assertIn("RERANK BEFORE [01]", logs)
        self.assertIn("retrieval_score=0.020000", logs)
        self.assertIn("RERANK AFTER [01]", logs)
        self.assertIn("rerank_score=0.950000", logs)
        self.assertIn("original_rank=1", logs)

    async def test_rerank_disabled_logs_retained_retrieval_order(self) -> None:
        search_result, _ = make_results()
        retriever = SimpleNamespace(
            pre_retriever=object(),
            retrieve=AsyncMock(return_value=[search_result]),
        )
        reranker = SimpleNamespace(rerank=AsyncMock())
        generator = SimpleNamespace(generate=AsyncMock(return_value="Câu trả lời"))

        with patch("app.pipeline.get_pre_retriever"):
            pipeline = ChatbotPipeline(generator, retriever, reranker)

        with patch("app.pipeline.logger") as logger:
            response = await pipeline.ask("Câu hỏi", use_rerank=False)

        logs = rendered_log_messages(logger)
        reranker.rerank.assert_not_awaited()
        self.assertIn("RERANK | enabled=false", logs)
        self.assertIn("RERANK OUTPUT (disabled) [01]", logs)
        self.assertIn("retrieval_score=0.020000", logs)
        self.assertEqual(response.sources[0].score_type, "rrf")


if __name__ == "__main__":
    import unittest

    unittest.main()
