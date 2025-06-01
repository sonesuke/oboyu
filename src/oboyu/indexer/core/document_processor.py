"""Document processing for Oboyu indexer.

This module handles document chunking, text normalization, and preparation
for embedding generation with special handling for Japanese content.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import Japanese text processing from crawler component
from oboyu.crawler.japanese import process_japanese_text


@dataclass
class Chunk:
    """Document chunk for indexing."""

    id: str
    """Unique identifier for the chunk."""

    path: Path
    """Path to the source document."""

    title: str
    """Title of the document or chunk."""

    content: str
    """Chunk text content."""

    chunk_index: int
    """Position of this chunk in the original document."""

    language: str
    """Language code of the content."""

    created_at: datetime
    """Timestamp when the chunk was created."""

    modified_at: datetime
    """Timestamp when the chunk was last modified."""

    metadata: Dict[str, object]
    """Additional metadata about the chunk."""

    prefix_content: Optional[str] = None
    """Content with prefix applied, ready for embedding."""


class DocumentProcessor:
    """Processor for chunking and preparing documents for indexing."""

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        document_prefix: str = "検索文書: ",
    ) -> None:
        """Initialize the document processor.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            document_prefix: Prefix to add to document chunks for embedding

        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_prefix = document_prefix

    def process_document(
        self,
        path: Path,
        content: str,
        title: str,
        language: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> List[Chunk]:
        """Process a document into chunks.

        Args:
            path: Path to the document
            content: Document content
            title: Document title
            language: Language code
            metadata: Additional metadata

        Returns:
            List of document chunks

        """
        # Apply language-specific processing
        processed_content = self._preprocess_content(content, language)

        # Split into chunks
        chunks = self._chunk_text(processed_content)

        # Create chunk objects
        now = datetime.now()

        # Extract dates from metadata if available
        created_at = now
        modified_at = now

        if metadata:
            if "created_at" in metadata and isinstance(metadata["created_at"], datetime):
                created_at = metadata["created_at"]
                # Remove from metadata dict to avoid duplication in JSON field
                metadata = {k: v for k, v in metadata.items() if k != "created_at"}

            if "updated_at" in metadata and isinstance(metadata["updated_at"], datetime):
                modified_at = metadata["updated_at"]
                # Remove from metadata dict to avoid duplication in JSON field
                metadata = {k: v for k, v in metadata.items() if k != "updated_at"}

        result = []

        for i, chunk_text in enumerate(chunks):
            # Generate a unique ID for the chunk
            chunk_id = f"{uuid.uuid4()}"

            # Create chunk object
            chunk = Chunk(
                id=chunk_id,
                path=path,
                title=f"{title} - Part {i + 1}" if len(chunks) > 1 else title,
                content=chunk_text,
                chunk_index=i,
                language=language,
                created_at=created_at,
                modified_at=modified_at,
                metadata=metadata or {},
            )

            # Add prefixed content for embedding
            chunk.prefix_content = self._add_prefix(chunk_text)

            result.append(chunk)

        return result

    def chunk_document(self, content: str, metadata: Dict[str, object]) -> List[Chunk]:
        """Split document into chunks.

        Args:
            content: Document content
            metadata: Document metadata containing path, title, language

        Returns:
            List of document chunks

        """
        path = metadata.get("path", Path("unknown"))
        title = metadata.get("title", "Unknown")
        language = metadata.get("language", "en")

        # Ensure path is properly converted to Path
        if isinstance(path, Path):
            doc_path = path
        elif isinstance(path, str):
            doc_path = Path(path)
        else:
            doc_path = Path("unknown")

        return self.process_document(
            path=doc_path,
            content=content,
            title=str(title),
            language=str(language),
            metadata=metadata,
        )

    def prepare_for_embedding(self, chunks: List[Chunk]) -> List[str]:
        """Prepare chunks for embedding generation.

        Args:
            chunks: List of document chunks

        Returns:
            List of text ready for embedding

        """
        return [chunk.prefix_content or self._add_prefix(chunk.content) for chunk in chunks]

    def prepare_for_bm25(self, chunks: List[Chunk]) -> List[str]:
        """Prepare chunks for BM25 indexing.

        Args:
            chunks: List of document chunks

        Returns:
            List of text content for BM25 tokenization

        """
        return [chunk.content for chunk in chunks]

    def _preprocess_content(self, content: str, language: str) -> str:
        """Preprocess content based on language.

        Args:
            content: Original content
            language: Language code

        Returns:
            Preprocessed content

        """
        # Apply special processing for Japanese text
        if language == "ja":
            # Use the Japanese text processing from the crawler component
            content = process_japanese_text(content, "utf-8")

        # For all languages
        # Remove excessive whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks

        """
        # If text is shorter than chunk size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        iteration_count = 0
        max_iterations = 10000  # Safety limit to prevent infinite loops

        while start < len(text) and iteration_count < max_iterations:
            iteration_count += 1

            # Get chunk of specified size
            end = start + self.chunk_size

            # Adjust chunk boundary to end at a sentence or paragraph if possible
            if end < len(text):
                # Try to find paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break
                else:
                    # Try to find sentence break
                    sentence_breaks = [
                        text.rfind(". ", start, end),
                        text.rfind("。", start, end),
                        text.rfind("! ", start, end),
                        text.rfind("？", start, end),
                        text.rfind("? ", start, end),
                        text.rfind("！", start, end),
                        text.rfind("\n", start, end),
                    ]

                    # Find the latest valid break point
                    valid_breaks = [b for b in sentence_breaks if b != -1 and b > start + self.chunk_size // 2]
                    if valid_breaks:
                        end = max(valid_breaks) + 1  # Include the punctuation

            # Add the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position for next chunk, considering overlap
            old_start = start
            start = end - self.chunk_overlap if end < len(text) else len(text)

            # Safety check to prevent infinite loops
            if start <= old_start and iteration_count > 1:
                import logging

                logging.error(f"_chunk_text: Potential infinite loop detected! start={start}, old_start={old_start}")
                start = old_start + max(1, self.chunk_size // 2)  # Force progress

        if iteration_count >= max_iterations:
            import logging

            logging.error(f"_chunk_text: Hit maximum iteration limit {max_iterations}")
        return chunks

    def _add_prefix(self, text: str) -> str:
        """Add the document prefix to a text for embedding.

        Args:
            text: Original text

        Returns:
            Text with prefix

        """
        return f"{self.document_prefix}{text}"


def chunk_documents(
    documents: List[Tuple[Path, str, str, str, Dict[str, object]]],
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    document_prefix: str = "検索文書: ",
) -> List[Chunk]:
    """Process multiple documents into chunks.

    Args:
        documents: List of (path, content, title, language, metadata) tuples
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks in characters
        document_prefix: Prefix to add to document chunks for embedding

    Returns:
        List of document chunks across all documents

    """
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document_prefix=document_prefix,
    )

    all_chunks = []
    for path, content, title, language, metadata in documents:
        chunks = processor.process_document(path, content, title, language, metadata)
        all_chunks.extend(chunks)

    return all_chunks
