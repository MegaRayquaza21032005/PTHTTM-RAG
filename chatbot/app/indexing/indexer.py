from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock

from app.api.schemas import Chunk
from app.core.logging import get_logger
from app.database.bm25_store import BM25Store, get_bm25_store
from app.database.vector_db import VectorDBClient, get_vector_db
from app.indexing.chunking import DocumentChunker, get_chunker
from app.indexing.embedding import Embedder, get_embedder


logger = get_logger(__name__)


class DocumentIndexer:
    """Điều phối chunking, embedding, Qdrant và BM25 indexing."""

    def __init__(
        self,
        chunker: DocumentChunker | None = None,
        embedder: Embedder | None = None,
        vector_db: VectorDBClient | None = None,
        bm25_store: BM25Store | None = None,
    ) -> None:
        self.chunker = chunker or get_chunker()
        self.embedder = embedder or get_embedder()
        self.vector_db = vector_db or get_vector_db()
        self.bm25_store = bm25_store or get_bm25_store()
        self._index_lock = asyncio.Lock()

    async def index_file(
        self,
        file_path: str | Path,
        *,
        document_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, object]:
        """Index một file Markdown vào cả Qdrant và BM25."""
        path = Path(file_path)
        resolved_document_id = document_id or path.stem

        async with self._index_lock:
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"Không tìm thấy tài liệu: {path}")
                if path.suffix.lower() != ".md":
                    raise ValueError(f"Chỉ hỗ trợ file Markdown: {path}")

                dense_exists, bm25_exists = await asyncio.gather(
                    self.vector_db.check_exists(resolved_document_id),
                    self.bm25_store.check_exists(resolved_document_id),
                )
                if dense_exists and bm25_exists:
                    logger.info(
                        "Bỏ qua tài liệu đã được index: %s",
                        resolved_document_id,
                    )
                    return {
                        "status": "skipped",
                        "document_id": resolved_document_id,
                        "chunk_count": 0,
                    }

                chunks = await self.chunker.asplit_file(
                    path,
                    document_id=resolved_document_id,
                    title=title,
                )
                dense_count = await self._index_dense_chunks(chunks)
                await self.bm25_store.upsert(chunks)

                logger.info(
                    "Đã index tài liệu %s với %d chunk",
                    resolved_document_id,
                    len(chunks),
                )
                return {
                    "status": "indexed",
                    "document_id": resolved_document_id,
                    "chunk_count": len(chunks),
                    "dense_count": dense_count,
                    "dense_existed": dense_exists,
                    "bm25_existed": bm25_exists,
                }
            except Exception:
                logger.exception(
                    "Lỗi khi index tài liệu %s",
                    resolved_document_id,
                )
                raise

    async def index_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = False,
    ) -> list[dict[str, object]]:
        """Index lần lượt tất cả file Markdown trong một thư mục."""
        directory_path = Path(directory)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Không tìm thấy thư mục: {directory_path}")

        pattern = "**/*.md" if recursive else "*.md"
        file_paths = sorted(directory_path.glob(pattern))
        results: list[dict[str, object]] = []

        logger.info(
            "Bắt đầu index %d file Markdown trong %s",
            len(file_paths),
            directory_path,
        )
        for file_path in file_paths:
            results.append(await self.index_file(file_path))

        logger.info("Đã hoàn thành index %d file Markdown", len(file_paths))
        return results

    async def _index_dense_chunks(self, chunks: list[Chunk]) -> int:
        indexed_count = 0
        batch_size = self.embedder.batch_size

        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            texts = [self._text_for_embedding(chunk) for chunk in chunk_batch]
            embeddings = await self.embedder.embed_batch(texts)
            points = [
                {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "payload": self._chunk_payload(chunk),
                }
                for chunk, embedding in zip(
                    chunk_batch,
                    embeddings,
                    strict=True,
                )
            ]
            await self.vector_db.upsert(points)
            indexed_count += len(points)

        return indexed_count

    @staticmethod
    def _text_for_embedding(chunk: Chunk) -> str:
        metadata = chunk.metadata
        context = filter(
            None,
            [
                metadata.title,
                metadata.chuong,
                metadata.muc,
                metadata.dieu,
                chunk.content,
            ],
        )
        return "\n".join(context)

    @staticmethod
    def _chunk_payload(chunk: Chunk) -> dict[str, object]:
        return {
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.metadata.model_dump(mode="json"),
        }


# Singleton Pattern
_indexer: DocumentIndexer | None = None
_indexer_lock = Lock()


def get_indexer() -> DocumentIndexer:
    """Trả về singleton document indexer."""
    global _indexer

    if _indexer is None:
        with _indexer_lock:
            if _indexer is None:
                try:
                    _indexer = DocumentIndexer()
                    logger.info("Đã khởi tạo DocumentIndexer")
                except Exception:
                    logger.exception("Lỗi khi khởi tạo DocumentIndexer")
                    raise

    return _indexer
