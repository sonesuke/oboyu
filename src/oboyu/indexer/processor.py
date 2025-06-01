"""Legacy import compatibility for processor module."""

from typing import Any, Dict, List

# Re-export from the new location
from oboyu.indexer.core.document_processor import Chunk, DocumentProcessor


# Legacy function - forward to DocumentProcessor
def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1024,
    chunk_overlap: int = 256,
    document_prefix: str = "検索文書: "
) -> List[Chunk]:
    """Legacy function for chunking documents."""
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document_prefix=document_prefix
    )
    all_chunks = []
    for doc in documents:
        chunks = processor.process_document(
            path=doc.get("path", "unknown"),
            content=doc.get("content", ""),
            title=doc.get("title", ""),
            language=doc.get("language", "en"),
            metadata=doc.get("metadata", {})
        )
        all_chunks.extend(chunks)
    return all_chunks

__all__ = ["Chunk", "DocumentProcessor", "chunk_documents"]
