"""Split legal Markdown documents while preserving their hierarchy."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock
from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.api.schemas import Chunk, MetaData
from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class DocumentChunker:
    """Create chunks after splitting Markdown by chapter, section and article."""

    HEADERS_TO_SPLIT_ON = [
        ("#", "chuong"),
        ("##", "muc"),
        ("###", "dieu"),
    ]
    SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = (
            settings.default_chunk_size if chunk_size is None else chunk_size
        )
        self.chunk_overlap = (
            settings.default_chunk_overlap
            if chunk_overlap is None
            else chunk_overlap
        )
        self._validate_chunk_config()

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
        )

    def split_markdown(
        self,
        markdown_text: str,
        *,
        document_id: str,
        title: str,
        source: str | None = None,
    ) -> list[Chunk]:
        """Split Markdown hierarchy first, then split each section into chunks."""
        if not markdown_text.strip():
            raise ValueError("Markdown content must not be empty")
        if not document_id.strip():
            raise ValueError("document_id must not be empty")
        if not title.strip():
            raise ValueError("title must not be empty")

        sections = self._header_splitter.split_text(markdown_text)
        documents = self._text_splitter.split_documents(sections)
        chunks: list[Chunk] = []

        for chunk_index, document in enumerate(documents):
            content = document.page_content.strip()
            if not content:
                continue

            metadata = MetaData(
                document_id=document_id,
                title=title,
                source=source,
                chuong=document.metadata.get("chuong"),
                muc=document.metadata.get("muc"),
                dieu=document.metadata.get("dieu"),
            )
            chunk_id = str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk_index}"))
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=content,
                    chunk_index=chunk_index,
                    metadata=metadata,
                )
            )

        logger.info("Created %d chunks from document %s", len(chunks), document_id)
        return chunks

    def split_file(
        self,
        file_path: str | Path,
        *,
        document_id: str | None = None,
        title: str | None = None,
    ) -> list[Chunk]:
        """Read and split one UTF-8 Markdown file."""
        path = Path(file_path)
        if path.suffix.lower() != ".md":
            raise ValueError(f"Only Markdown files are supported: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        markdown_text = path.read_text(encoding="utf-8")
        return self.split_markdown(
            markdown_text,
            document_id=document_id or path.stem,
            title=title or path.stem.replace("-", " ").replace("_", " "),
            source=str(path),
        )

    async def asplit_file(
        self,
        file_path: str | Path,
        *,
        document_id: str | None = None,
        title: str | None = None,
    ) -> list[Chunk]:
        """Run file reading and chunking without blocking the event loop."""
        return await asyncio.to_thread(
            self.split_file,
            file_path,
            document_id=document_id,
            title=title,
        )

    def _validate_chunk_config(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")



# Singleton Pattern
_chunker: DocumentChunker | None = None
_chunker_lock = Lock()


def get_chunker() -> DocumentChunker:
    """Return the shared document chunker instance."""
    global _chunker

    if _chunker is None:
        with _chunker_lock:
            if _chunker is None:
                _chunker = DocumentChunker()
                logger.info(
                    "Chunker initialized | chunk_size=%d | chunk_overlap=%d",
                    _chunker.chunk_size,
                    _chunker.chunk_overlap,
                )

    return _chunker


if __name__ == "__main__":
    import pprint
    document = DocumentChunker()
    pprint.pprint(document.split_file(file_path="/home/nguyen-thanh-dat/Documents/PTHTTM_PTIT/HTTM_Representation/chatbot/data/de_muc_20_2_lao-dong.md")[0])