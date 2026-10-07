from __future__ import annotations

import re
from abc import ABC, abstractmethod
from threading import Lock
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()
StrategyName = Literal["rewrite", "decomposition", "expansion"]


class QueryStrategy(ABC):
    """Lớp cơ sở cho các chiến lược xử lý truy vấn."""

    system_prompt: str

    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", "{query}")]
        )
        self.chain = prompt | llm | StrOutputParser()

    async def _invoke(self, query: str) -> str:
        return (await self.chain.ainvoke({"query": query})).strip()

    @abstractmethod
    async def transform(self, query: str) -> list[str]:
        """Biến đổi truy vấn thành một hoặc nhiều truy vấn tìm kiếm."""
        ...

    @staticmethod
    def _parse_lines(text: str) -> list[str]:
        queries = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                queries.append(cleaned)
        return queries


class QueryRewriting(QueryStrategy):
    system_prompt = """
Bạn viết lại câu hỏi để tìm kiếm trong kho văn bản pháp luật Việt Nam.
Giữ nguyên ý định, bổ sung ngữ cảnh còn thiếu và dùng thuật ngữ pháp lý rõ ràng.
Chỉ trả về một câu truy vấn, không giải thích và không trả lời câu hỏi.
"""

    async def transform(self, query: str) -> list[str]:
        rewritten_query = await self._invoke(query)
        return [rewritten_query or query]


class QueryDecomposition(QueryStrategy):
    system_prompt = """
Phân rã câu hỏi pháp luật phức tạp thành tối đa 4 câu hỏi con độc lập.
Nếu câu hỏi đã đơn giản, giữ nguyên câu hỏi đó.
Mỗi dòng chỉ chứa một câu hỏi, không đánh số, không giải thích và không trả lời.
"""

    async def transform(self, query: str) -> list[str]:
        queries = self._parse_lines(await self._invoke(query))
        return queries or [query]


class QueryExpansion(QueryStrategy):
    system_prompt = """
Tạo 3 biến thể tìm kiếm cho câu hỏi pháp luật bằng từ đồng nghĩa và thuật ngữ liên quan.
Không thay đổi ý định của câu hỏi.
Mỗi dòng chỉ chứa một truy vấn, không đánh số, không giải thích và không trả lời.
"""

    async def transform(self, query: str) -> list[str]:
        expanded_queries = self._parse_lines(await self._invoke(query))
        return self._unique([query, *expanded_queries])

    @staticmethod
    def _unique(queries: list[str]) -> list[str]:
        unique_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            key = query.casefold()
            if key not in seen:
                seen.add(key)
                unique_queries.append(query)
        return unique_queries


class PreRetriever:
    """Xử lý truy vấn trước retrieval bằng OpenAI."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = ChatOpenAI(
            model=settings.pre_retrieval_model,
            api_key=settings.openai_api_key or None,
            temperature=0,
        )
        self.strategies: dict[StrategyName, QueryStrategy] = {
            "rewrite": QueryRewriting(self.llm),
            "decomposition": QueryDecomposition(self.llm),
            "expansion": QueryExpansion(self.llm),
        }

    async def process(
        self,
        query: str,
        strategy: StrategyName = "rewrite",
    ) -> list[str]:
        try:
            queries = await self.strategies[strategy].transform(query)
            logger.info(
                "PRE-RETRIEVAL | enabled=true | strategy=%s | "
                "original_query=%r | output_count=%d",
                strategy,
                query,
                len(queries),
            )
            for index, transformed_query in enumerate(queries, start=1):
                logger.info(
                    "PRE-RETRIEVAL OUTPUT [%02d] | strategy=%s | query=%r",
                    index,
                    strategy,
                    transformed_query,
                )
            return queries
        except Exception:
            logger.exception(
                "Lỗi khi xử lý truy vấn bằng chiến lược %s",
                strategy,
            )
            raise


# Singleton Pattern
_pre_retriever: PreRetriever | None = None
_pre_retriever_lock = Lock()


def get_pre_retriever(llm: BaseChatModel | None = None) -> PreRetriever:
    """Trả về singleton pre-retriever."""
    global _pre_retriever

    if _pre_retriever is None:
        with _pre_retriever_lock:
            if _pre_retriever is None:
                _pre_retriever = PreRetriever(llm)
                logger.info(
                    "Đã khởi tạo PreRetriever | provider=openai | model=%s",
                    settings.pre_retrieval_model,
                )

    return _pre_retriever


if __name__ == "__main__":
    import pprint
    import asyncio

    async def main():
        pre_retriever = PreRetriever()
        rewrite_query = await pre_retriever.process(query="Luật lao động là gì?", strategy = "rewrite")
        query_expansion = await pre_retriever.process(query="Luật lao động là gì?", strategy = "expansion")
        query_decomposition = await pre_retriever.process(query="Luật lao động là gì?", strategy = "decomposition")
        print("========Viết lại truy vấn========\n")
        print(rewrite_query)

        print("========Mở rộng truy vấn========\n")
        print(query_expansion)

        print("========Phân rã truy vấn========\n")
        print(query_decomposition)

    asyncio.run(main())
