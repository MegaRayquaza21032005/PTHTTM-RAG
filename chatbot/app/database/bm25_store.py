from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from threading import Lock

from rank_bm25 import BM25Okapi

from app.api.schemas import Chunk
from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class BM25Store:
    """Quản lý BM25 index và corpus chunk được lưu trên ổ đĩa."""

    INDEX_FILE_NAME = "chunks.json"

    def __init__(self, store_path: str | Path | None = None) -> None:
        configured_path = Path(store_path or settings.bm25_store_path)
        if not configured_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            configured_path = project_root / configured_path

        self.store_path = configured_path.resolve()
        self.index_file = self.store_path / self.INDEX_FILE_NAME
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._loaded = False
        self._lock = asyncio.Lock()

    async def build(self, chunks: list[Chunk]) -> None:
        """Dựng lại BM25 index từ toàn bộ danh sách chunk."""
        async with self._lock:
            try:
                await asyncio.to_thread(self._build_index, list(chunks))
                self._loaded = True
                logger.info("Đã tạo BM25 index từ %d chunk", len(chunks))
            except Exception:
                logger.exception("Lỗi khi tạo BM25 index")
                raise

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Thêm mới hoặc cập nhật chunk theo chunk_id rồi dựng lại index."""
        if not chunks:
            return

        await self._ensure_loaded()
        async with self._lock:
            try:
                chunk_map = {chunk.chunk_id: chunk for chunk in self._chunks}
                chunk_map.update({chunk.chunk_id: chunk for chunk in chunks})
                await asyncio.to_thread(
                    self._build_index,
                    list(chunk_map.values()),
                )
                await asyncio.to_thread(self._save_to_disk)
                logger.info("Đã upsert %d chunk vào BM25", len(chunks))
            except Exception:
                logger.exception("Lỗi khi upsert chunk vào BM25")
                raise

    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """Tìm kiếm chunk bằng điểm BM25."""
        await self._ensure_loaded()

        async with self._lock:
            if self._bm25 is None:
                return []

            try:
                results = await asyncio.to_thread(self._search, query, top_k)
                logger.debug("Tìm thấy %d kết quả từ BM25", len(results))
                return results
            except Exception:
                logger.exception("Lỗi khi tìm kiếm bằng BM25")
                raise

    async def save(self) -> None:
        """Lưu corpus hiện tại xuống ổ đĩa."""
        async with self._lock:
            try:
                await asyncio.to_thread(self._save_to_disk)
                logger.info("Đã lưu BM25 corpus tại %s", self.index_file)
            except Exception:
                logger.exception("Lỗi khi lưu BM25 corpus")
                raise

    async def load(self) -> None:
        """Tải corpus từ ổ đĩa và dựng lại BM25 index."""
        async with self._lock:
            try:
                await asyncio.to_thread(self._load_from_disk)
                self._loaded = True
                logger.info("Đã tải BM25 index gồm %d chunk", len(self._chunks))
            except Exception:
                logger.exception("Lỗi khi tải BM25 index")
                raise

    async def check_exists(self, document_id: str) -> bool:
        """Kiểm tra document_id đã có trong BM25 corpus hay chưa."""
        await self._ensure_loaded()
        async with self._lock:
            return any(
                chunk.metadata.document_id == document_id
                for chunk in self._chunks
            )

    async def delete_document(self, document_id: str) -> int:
        """Xóa toàn bộ chunk thuộc một tài liệu và trả về số chunk đã xóa."""
        await self._ensure_loaded()
        async with self._lock:
            try:
                remaining_chunks = [
                    chunk
                    for chunk in self._chunks
                    if chunk.metadata.document_id != document_id
                ]
                deleted_count = len(self._chunks) - len(remaining_chunks)
                if deleted_count:
                    await asyncio.to_thread(self._build_index, remaining_chunks)
                    await asyncio.to_thread(self._save_to_disk)

                logger.info(
                    "Đã xóa %d chunk của tài liệu %s khỏi BM25",
                    deleted_count,
                    document_id,
                )
                return deleted_count
            except Exception:
                logger.exception(
                    "Lỗi khi xóa tài liệu %s khỏi BM25",
                    document_id,
                )
                raise

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if self.index_file.is_file():
            await self.load()
            return

        async with self._lock:
            if not self._loaded:
                self._chunks = []
                self._bm25 = None
                self._loaded = True

    def _build_index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        if not chunks:
            self._bm25 = None
            return

        tokenized_corpus = [
            self._tokenize(self._text_for_index(chunk)) for chunk in chunks
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _search(self, query: str, top_k: int) -> list[dict[str, object]]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[dict[str, object]] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue

            chunk = self._chunks[index]
            results.append(
                {
                    "id": chunk.chunk_id,
                    "score": score,
                    "payload": {
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata.model_dump(),
                    },
                }
            )

        return results

    def _save_to_disk(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        temporary_file = self.index_file.with_suffix(".tmp")
        data = [chunk.model_dump(mode="json") for chunk in self._chunks]
        temporary_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_file.replace(self.index_file)

    def _load_from_disk(self) -> None:
        data = json.loads(self.index_file.read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(item) for item in data]
        self._build_index(chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _text_for_index(chunk: Chunk) -> str:
        metadata = chunk.metadata
        context = filter(
            None,
            [metadata.chuong, metadata.muc, metadata.dieu, chunk.content],
        )
        return "\n".join(context)


# Singleton Pattern 
_bm25_store: BM25Store | None = None
_bm25_store_lock = Lock()


def get_bm25_store() -> BM25Store:
    """Trả về singleton BM25 store."""
    global _bm25_store

    if _bm25_store is None:
        with _bm25_store_lock:
            if _bm25_store is None:
                try:
                    _bm25_store = BM25Store()
                    logger.info(
                        "Đã khởi tạo BM25 store | path=%s",
                        _bm25_store.store_path,
                    )
                except Exception:
                    logger.exception("Lỗi khi khởi tạo BM25 store")
                    raise

    return _bm25_store
