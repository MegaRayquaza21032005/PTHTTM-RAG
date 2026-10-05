from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class VectorDBClient(ABC):
    """Interface chung cho các vector database."""

    @abstractmethod
    async def upsert(self, vectors: list[dict[str, Any]]) -> dict[str, Any]:
        """Thêm mới hoặc cập nhật danh sách vector."""
        ...

    @abstractmethod
    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Tìm các vector gần nhất."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Xóa vector theo danh sách ID."""
        ...

    @abstractmethod
    async def check_exists(self, document_id: str) -> bool:
        """Kiểm tra tài liệu đã được index hay chưa."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Đóng kết nối tới vector database."""
        ...


class QdrantClient(VectorDBClient):
    """Async client quản lý collection vector trên Qdrant."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.dimension = dimension or settings.embedding_dimension
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key or None,
        )
        self._collection_ready = False
        self._collection_lock = asyncio.Lock()

        logger.info(
            "Đã khởi tạo Qdrant client | url=%s | collection=%s",
            self.url,
            self.collection_name,
        )

    async def upsert(self, vectors: list[dict[str, Any]]) -> dict[str, Any]:
        if not vectors:
            return {"status": "skipped", "count": 0}

        try:
            await self._ensure_collection()
            points = [
                PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item.get("payload", {}),
                )
                for item in vectors
            ]
            result = await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            status = getattr(result.status, "value", str(result.status))

            logger.info("Đã upsert %d vector vào Qdrant", len(points))
            return {
                "status": status,
                "count": len(points),
                "operation_id": result.operation_id,
            }
        except Exception:
            logger.exception("Lỗi khi upsert %d vector vào Qdrant", len(vectors))
            raise

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            await self._ensure_collection()
            result = await self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=self._build_filter(filters),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            points = [
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": dict(point.payload or {}),
                }
                for point in result.points
            ]

            logger.debug("Tìm thấy %d kết quả từ Qdrant", len(points))
            return points
        except Exception:
            logger.exception("Lỗi khi truy vấn vector từ Qdrant")
            raise

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        try:
            await self._ensure_collection()
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
                wait=True,
            )
            logger.info("Đã xóa %d vector khỏi Qdrant", len(ids))
        except Exception:
            logger.exception("Lỗi khi xóa %d vector khỏi Qdrant", len(ids))
            raise

    async def check_exists(self, document_id: str) -> bool:
        try:
            await self._ensure_collection()
            points, _ = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return bool(points)
        except Exception:
            logger.exception(
                "Lỗi khi kiểm tra tài liệu %s trong Qdrant",
                document_id,
            )
            raise

    async def close(self) -> None:
        try:
            await self.client.close()
            self._collection_ready = False
            logger.info("Đã đóng kết nối Qdrant")
        except Exception:
            logger.exception("Lỗi khi đóng kết nối Qdrant")
            raise

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return

        async with self._collection_lock:
            if self._collection_ready:
                return

            exists = await self.client.collection_exists(self.collection_name)
            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Đã tạo Qdrant collection | name=%s | dimension=%d",
                    self.collection_name,
                    self.dimension,
                )

            self._collection_ready = True

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None

        return Filter(
            must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
        )


# Singleton Pattern
_vector_db: VectorDBClient | None = None
_vector_db_lock = Lock()


def get_vector_db() -> VectorDBClient:
    """Trả về singleton vector database client."""
    global _vector_db

    if _vector_db is None:
        with _vector_db_lock:
            if _vector_db is None:
                try:
                    _vector_db = QdrantClient()
                except Exception:
                    logger.exception("Lỗi khi khởi tạo Qdrant client")
                    raise

    return _vector_db
