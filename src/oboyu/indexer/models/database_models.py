"""Pydantic models for database records and operations."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ChunkRecord(BaseModel):
    """Model for chunk database records."""

    id: str = Field(description="Unique chunk identifier")
    path: str = Field(description="Path to source file")
    title: str = Field(description="Document or chunk title")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(ge=0, description="Position in original document")
    language: str = Field(description="Language code (e.g., 'ja', 'en')")
    created_at: datetime = Field(description="Creation timestamp")
    modified_at: datetime = Field(description="Last modification timestamp")
    metadata: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Additional metadata as dictionary"
    )


class EmbeddingRecord(BaseModel):
    """Model for embedding database records."""

    id: str = Field(description="Unique embedding identifier")
    chunk_id: str = Field(description="Related chunk ID")
    model: str = Field(description="Embedding model used")
    vector: List[float] = Field(description="Embedding vector")
    created_at: datetime = Field(description="Embedding generation timestamp")

    @field_validator('vector')
    @classmethod
    def validate_vector(cls, v: List[float]) -> List[float]:
        """Validate vector dimensions and values."""
        if not v:
            raise ValueError("Vector cannot be empty")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("Vector must contain only numeric values")
        return v


class VocabularyRecord(BaseModel):
    """Model for BM25 vocabulary records."""

    term: str = Field(description="Vocabulary term")
    document_frequency: int = Field(ge=0, description="Number of documents containing term")
    collection_frequency: int = Field(ge=0, description="Total occurrences across collection")


class InvertedIndexRecord(BaseModel):
    """Model for BM25 inverted index records."""

    term: str = Field(description="Index term")
    chunk_id: str = Field(description="Chunk ID")
    term_frequency: int = Field(ge=0, description="Frequency of term in chunk")
    positions: List[int] = Field(default_factory=list, description="Token positions for phrase search")


class DocumentStatsRecord(BaseModel):
    """Model for BM25 document statistics records."""

    chunk_id: str = Field(description="Chunk ID")
    total_terms: int = Field(ge=0, description="Total number of terms in document")
    unique_terms: int = Field(ge=0, description="Number of unique terms")
    avg_term_frequency: float = Field(ge=0.0, description="Average term frequency")


class CollectionStatsRecord(BaseModel):
    """Model for BM25 collection statistics records."""

    id: int = Field(default=1, description="Single row table identifier")
    total_documents: int = Field(ge=0, description="Total number of documents")
    total_terms: int = Field(ge=0, description="Total terms across collection")
    avg_document_length: float = Field(ge=0.0, description="Average document length")
    last_updated: datetime = Field(description="Last update timestamp")


class FileMetadataRecord(BaseModel):
    """Model for file metadata tracking records."""

    path: str = Field(description="File path")
    last_processed_at: datetime = Field(description="Last processing timestamp")
    file_modified_at: datetime = Field(description="File modification timestamp")
    file_size: int = Field(ge=0, description="File size in bytes")
    content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of file content")
    chunk_count: int = Field(ge=0, default=0, description="Number of chunks created from file")
    processing_status: str = Field(default="completed", description="Processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Record update timestamp")

    @field_validator('processing_status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate processing status."""
        valid_statuses = {"completed", "error", "in_progress"}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}, got {v}")
        return v


class SchemaVersionRecord(BaseModel):
    """Model for schema version tracking records."""

    version: str = Field(description="Schema version")
    description: str = Field(description="Version description")
    applied_at: datetime = Field(description="Migration application timestamp")
    migration_checksum: Optional[str] = Field(default=None, description="Checksum of migration SQL")


class QueryParameters(BaseModel):
    """Model for search query parameters."""

    query: str = Field(min_length=1, description="Search query text")
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    mode: str = Field(default="hybrid", description="Search mode")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector search in hybrid mode")
    rerank: bool = Field(default=True, description="Enable reranking")
    include_scores: bool = Field(default=False, description="Include similarity scores in results")

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate search mode."""
        valid_modes = {"vector", "bm25", "hybrid"}
        if v not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {v}")
        return v


class IndexingRequest(BaseModel):
    """Model for indexing operation requests."""

    source_path: Path = Field(description="Path to index")
    config_overrides: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Configuration overrides for this indexing operation"
    )
    force_reindex: bool = Field(default=False, description="Force reindexing of existing files")
    dry_run: bool = Field(default=False, description="Perform dry run without actual indexing")


class IndexingResult(BaseModel):
    """Model for indexing operation results."""

    success: bool = Field(description="Whether indexing was successful")
    files_processed: int = Field(ge=0, description="Number of files processed")
    chunks_created: int = Field(ge=0, description="Number of chunks created")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    skipped_files: List[str] = Field(default_factory=list, description="Files that were skipped")
