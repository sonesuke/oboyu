"""Document processing for Oboyu indexer.

This module handles document chunking, text normalization, and preparation
for embedding generation with special handling for Japanese content.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from oboyu.common.types import Chunk

from .document_chunker import DocumentChunker
from .embedding_prefix_handler import EmbeddingPrefixHandler
from .language_processor import LanguageProcessor
from .text_normalizer import TextNormalizer


class DocumentProcessor:
    """Processor for chunking and preparing documents for indexing."""

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        document_prefix: str = "検索文書: ",
        text_normalizer: Optional[TextNormalizer] = None,
        document_chunker: Optional[DocumentChunker] = None,
        language_processor: Optional[LanguageProcessor] = None,
        prefix_handler: Optional[EmbeddingPrefixHandler] = None,
    ) -> None:
        """Initialize the document processor.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            document_prefix: Prefix to add to document chunks for embedding
            text_normalizer: Text normalization handler
            document_chunker: Document chunking handler
            language_processor: Language-specific processing handler
            prefix_handler: Embedding prefix handler

        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_prefix = document_prefix
        
        # Initialize processing components with defaults if not provided
        self.text_normalizer = text_normalizer or TextNormalizer()
        self.document_chunker = document_chunker or DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.language_processor = language_processor or LanguageProcessor()
        self.prefix_handler = prefix_handler or EmbeddingPrefixHandler(
            document_prefix=document_prefix
        )

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
        processed_content = self.language_processor.prepare_text(content, language)
        
        # Normalize the text
        normalized_content = self.text_normalizer.normalize(processed_content, language)

        # Split into chunks
        chunks = self.document_chunker.chunk_text(normalized_content)

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
            chunk.prefix_content = self.prefix_handler.add_document_prefix(chunk_text, "ruri")

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
        return [
            chunk.prefix_content or self.prefix_handler.add_document_prefix(chunk.content, "ruri")
            for chunk in chunks
        ]

    def prepare_for_bm25(self, chunks: List[Chunk]) -> List[str]:
        """Prepare chunks for BM25 indexing.

        Args:
            chunks: List of document chunks

        Returns:
            List of text content for BM25 tokenization

        """
        return [chunk.content for chunk in chunks]


def chunk_documents(
    documents: Union[List[Tuple[Path, str, str, str, Dict[str, object]]], List[Dict[str, object]]],
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    document_prefix: str = "検索文書: ",
) -> List[Chunk]:
    """Process multiple documents into chunks.

    Args:
        documents: List of (path, content, title, language, metadata) tuples or list of document dicts
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
    
    for doc in documents:
        if isinstance(doc, dict):
            # Handle dictionary format
            path = Path(str(doc["path"]))
            content = str(doc["content"])
            title = str(doc["title"])
            language = str(doc["language"])
            metadata = doc.get("metadata", {})
            # Ensure metadata is the right type
            if not isinstance(metadata, dict):
                metadata = {}
        else:
            # Handle tuple format
            path, content, title, language, metadata = doc
            
        chunks = processor.process_document(path, content, title, language, metadata)
        all_chunks.extend(chunks)

    return all_chunks
