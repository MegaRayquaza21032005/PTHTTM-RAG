from __future__ import annotations

import asyncio
from collections.abc import Sequence
from threading import Lock

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class Embedder:
    def __init__(
        self,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self.dimension = (
            settings.embedding_dimension if dimension is None else dimension
        )
        self.batch_size = (
            settings.embedding_batch_size if batch_size is None else batch_size
        )
        self.encoder = SentenceTransformer(self.model)

    async def embed_text(self, text: str) -> list[float]:
        """Tạo vector embedding cho một văn bản."""
        clean_text = text.replace("\n", " ")

        try:
            vector = await asyncio.to_thread(
                self.encoder.encode,
                clean_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vector.tolist()
        except Exception:
            logger.exception("Lỗi khi tạo embedding cho văn bản")
            raise

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Tạo vector embedding cho nhiều văn bản."""
        clean_texts = [text.replace("\n", " ") for text in texts]

        try:
            embeddings = await asyncio.to_thread(
                self.encoder.encode,
                clean_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = embeddings.tolist()
            logger.debug("Đã tạo embedding cho %d văn bản", len(vectors))
            return vectors
        except Exception:
            logger.exception(
                "Lỗi khi tạo embedding theo batch gồm %d văn bản",
                len(clean_texts),
            )
            raise


# Singleton Pattern
_embedder: Embedder | None = None
_embedder_lock = Lock()


def get_embedder() -> Embedder:
    """Return the shared embedder instance."""
    global _embedder

    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                try:
                    _embedder = Embedder()
                    logger.info(
                        "Đã khởi tạo Embedder | model=%s | dimension=%d",
                        _embedder.model,
                        _embedder.dimension,
                    )
                except Exception:
                    logger.exception("Lỗi khi khởi tạo Embedder")
                    raise

    return _embedder


if __name__ == "__main__":
    from app.indexing.chunking import DocumentChunker
    import asyncio
    document_chunker = DocumentChunker()
    text = document_chunker.split_file(file_path="/home/nguyen-thanh-dat/Documents/PTHTTM_PTIT/HTTM_Representation/chatbot/data/de_muc_20_2_lao-dong.md")[0].content
    
    async def main():
        embedder = Embedder()
        vector = await embedder.embed_text(text = text)
        print(len(vector))
    asyncio.run(main())