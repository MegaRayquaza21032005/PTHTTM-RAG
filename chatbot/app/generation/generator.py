from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from threading import Lock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.api.schemas import SearchResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.generation.prompt import PromptBuilder, get_prompt_builder


logger = get_logger(__name__)
settings = get_settings()


class BaseGenerator(ABC):
    """Lớp cơ sở cho các provider sinh câu trả lời."""

    def __init__(self, prompt_builder: PromptBuilder | None = None) -> None:
        self.prompt_builder = prompt_builder or get_prompt_builder()
        self.llm = self._create_llm()

    @abstractmethod
    def _create_llm(self) -> BaseChatModel:
        """Khởi tạo chat model của provider."""
        ...

    async def generate(
        self,
        question: str,
        results: Sequence[SearchResult],
    ) -> str:
        """Sinh câu trả lời từ câu hỏi và các tài liệu đã retrieval."""
        try:
            messages = self.prompt_builder.build_messages(question, results)
            response = await self.llm.ainvoke(messages)
            answer = self._extract_text(response)
            logger.info("Đã sinh câu trả lời bằng %s", self.__class__.__name__)
            return answer
        except Exception:
            logger.exception("Lỗi khi sinh câu trả lời")
            raise

    @staticmethod
    def _extract_text(message: BaseMessage) -> str:
        if isinstance(message.content, str):
            return message.content.strip()

        texts = [
            block.get("text", "")
            for block in message.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(texts).strip()


class OpenAIGenerator(BaseGenerator):
    """Sinh câu trả lời bằng OpenAI."""

    def _create_llm(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            temperature=0,
        )


class GeminiGenerator(BaseGenerator):
    """Sinh câu trả lời bằng Google Gemini."""

    def _create_llm(self) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key or None,
            temperature=0,
        )


# Singleton Pattern
_generator: BaseGenerator | None = None
_generator_lock = Lock()


def get_generator() -> BaseGenerator:
    """Trả về singleton generator theo provider trong cấu hình."""
    global _generator

    if _generator is None:
        with _generator_lock:
            if _generator is None:
                try:
                    if settings.llm_provider == "openai":
                        _generator = OpenAIGenerator()
                    else:
                        _generator = GeminiGenerator()
                    logger.info(
                        "Đã khởi tạo generator | provider=%s",
                        settings.llm_provider,
                    )
                except Exception:
                    logger.exception("Lỗi khi khởi tạo generator")
                    raise

    return _generator

if __name__ == "__main__":
    import asyncio

    from app.retrieval.retriever import Retriever
    from app.retrieval.reranker import BGEReranker

    async def main():
        question = "Luật Lao động là gì?"
        retriever = Retriever()
        results = await retriever.retrieve(
            query=question,
            use_pre_retrieval=False,
        )
        reranker = BGEReranker()
        ranked_results = await reranker.rerank(
            query=question,
            results=results,
            top_n=5,
        )
        # prompt_builder = PromptBuilder()
        # messages = prompt_builder.build_messages(question, ranked_results)
        # for message in messages:
        #     print(message.pretty_repr())
            
        generator = GeminiGenerator()
        response = await generator.generate(question=question, results=ranked_results)
        print(response)
        print("\n\n")

    asyncio.run(main())
    