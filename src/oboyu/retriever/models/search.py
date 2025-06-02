"""Enhanced indexing models with comprehensive validation."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    
    RURI_V3_30M = "cl-nagoya/ruri-v3-30m"
    RURI_RERANKER_SMALL = "cl-nagoya/ruri-reranker-small"
    CUSTOM = "custom"


class IndexingStatus(str, Enum):
    """Indexing operation status."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EmbeddingVector(BaseModel):
    """Model for embedding vectors with validation."""
    
    chunk_id: str = Field(..., description="Associated chunk identifier")
    model: EmbeddingModel = Field(..., description="Embedding model used")
    vector: List[float] = Field(..., description="Embedding vector")
    dimensions: int = Field(..., ge=128, le=2048, description="Vector dimensions")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('vector')
    @classmethod
    def validate_vector_dimension(cls, v: List[float]) -> List[float]:
        """Validate vector dimensions and values."""
        if not v:
            raise ValueError('Vector cannot be empty')
        
        # Check for valid numeric values
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)) or val != val:  # NaN check
                raise ValueError(f'Invalid vector value at position {i}: {val}')
        
        return v
    
    @model_validator(mode='after')
    def validate_vector_length(self) -> 'EmbeddingVector':
        """Validate vector length matches dimensions."""
        if len(self.vector) != self.dimensions:
            raise ValueError(f'Vector length {len(self.vector)} does not match dimensions {self.dimensions}')
        return self
    
    def normalize(self) -> 'EmbeddingVector':
        """Return normalized vector."""
        import math
        
        # Calculate magnitude
        magnitude = math.sqrt(sum(x * x for x in self.vector))
        if magnitude == 0:
            raise ValueError('Cannot normalize zero vector')
        
        # Normalize
        normalized_vector = [x / magnitude for x in self.vector]
        
        return self.model_copy(update={'vector': normalized_vector})


class BM25Statistics(BaseModel):
    """Model for BM25 statistics with validation."""
    
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    vocabulary_size: int = Field(..., ge=0, description="Size of vocabulary")
    average_document_length: float = Field(..., gt=0.0, description="Average document length")
    collection_frequency: Dict[str, int] = Field(default_factory=dict, description="Term frequencies")
    document_frequencies: Dict[str, int] = Field(default_factory=dict, description="Document frequencies")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @field_validator('collection_frequency', 'document_frequencies')
    @classmethod
    def validate_frequency_dicts(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate frequency dictionaries."""
        for term, freq in v.items():
            if freq < 0:
                raise ValueError(f'Frequency cannot be negative for term "{term}": {freq}')
            if not term.strip():
                raise ValueError('Term cannot be empty or whitespace')
        return v
    
    @model_validator(mode='after')
    def validate_statistics_consistency(self) -> 'BM25Statistics':
        """Validate statistics consistency."""
        # Check that document frequencies don't exceed total documents
        for term, doc_freq in self.document_frequencies.items():
            if doc_freq > self.total_documents:
                raise ValueError(f'Document frequency for "{term}" ({doc_freq}) exceeds total documents ({self.total_documents})')
        
        return self


class IndexingMetrics(BaseModel):
    """Model for indexing operation metrics."""
    
    operation_id: str = Field(..., description="Unique operation identifier")
    status: IndexingStatus = Field(..., description="Operation status")
    start_time: datetime = Field(..., description="Operation start time")
    end_time: Optional[datetime] = Field(default=None, description="Operation end time")
    files_discovered: int = Field(ge=0, description="Number of files discovered")
    files_processed: int = Field(ge=0, description="Number of files processed successfully")
    files_failed: int = Field(ge=0, description="Number of files that failed processing")
    files_skipped: int = Field(ge=0, description="Number of files skipped")
    chunks_created: int = Field(ge=0, description="Number of chunks created")
    embeddings_generated: int = Field(ge=0, description="Number of embeddings generated")
    processing_errors: List[str] = Field(default_factory=list, description="Processing error messages")
    memory_usage_mb: Optional[float] = Field(default=None, ge=0.0, description="Peak memory usage in MB")
    
    @model_validator(mode='after')
    def validate_metrics_consistency(self) -> 'IndexingMetrics':
        """Validate metrics consistency."""
        # Check that processed + failed + skipped = discovered
        total_handled = self.files_processed + self.files_failed + self.files_skipped
        if total_handled > self.files_discovered:
            raise ValueError(f'Total handled files ({total_handled}) exceeds discovered files ({self.files_discovered})')
        
        # Check time consistency
        if self.end_time and self.end_time < self.start_time:
            raise ValueError('End time cannot be before start time')
        
        # Check status consistency
        if self.status == IndexingStatus.COMPLETED and self.end_time is None:
            raise ValueError('End time is required for completed operations')
        
        if self.status == IndexingStatus.FAILED and not self.processing_errors:
            raise ValueError('Processing errors are required for failed operations')
        
        return self
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def get_processing_rate(self) -> Optional[float]:
        """Get files processed per second."""
        duration = self.get_duration_seconds()
        if duration and duration > 0:
            return self.files_processed / duration
        return None


class VectorSearchQuery(BaseModel):
    """Model for vector search queries."""
    
    query_vector: List[float] = Field(..., description="Query embedding vector")
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    ef_search: Optional[int] = Field(default=None, ge=1, description="HNSW ef_search parameter")
    include_scores: bool = Field(default=True, description="Include similarity scores")
    
    @field_validator('query_vector')
    @classmethod
    def validate_query_vector(cls, v: List[float]) -> List[float]:
        """Validate query vector."""
        if not v:
            raise ValueError('Query vector cannot be empty')
        
        # Check for valid numeric values
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)) or val != val:  # NaN check
                raise ValueError(f'Invalid vector value at position {i}: {val}')
        
        return v


class BM25SearchQuery(BaseModel):
    """Model for BM25 search queries."""
    
    query_terms: List[str] = Field(..., min_length=1, description="Tokenized query terms")
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    k1: float = Field(default=1.5, ge=0.0, le=3.0, description="BM25 k1 parameter")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 b parameter")
    include_scores: bool = Field(default=True, description="Include BM25 scores")
    
    @field_validator('query_terms')
    @classmethod
    def validate_query_terms(cls, v: List[str]) -> List[str]:
        """Validate and clean query terms."""
        cleaned = []
        for term in v:
            cleaned_term = term.strip().lower()
            if cleaned_term and not cleaned_term.isspace():
                cleaned.append(cleaned_term)
        
        if not cleaned:
            raise ValueError('At least one valid query term is required')
        
        return cleaned


class HybridSearchQuery(BaseModel):
    """Model for hybrid search queries."""
    
    vector_query: VectorSearchQuery = Field(..., description="Vector search component")
    bm25_query: BM25SearchQuery = Field(..., description="BM25 search component")
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for vector search")
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for BM25 search")
    
    @model_validator(mode='after')
    def validate_weights(self) -> 'HybridSearchQuery':
        """Validate that weights sum to 1.0."""
        if abs(self.vector_weight + self.bm25_weight - 1.0) > 0.001:
            raise ValueError('vector_weight + bm25_weight must equal 1.0')
        return self
    
    @model_validator(mode='after')
    def validate_top_k_consistency(self) -> 'HybridSearchQuery':
        """Validate top_k consistency between queries."""
        if self.vector_query.top_k != self.bm25_query.top_k:
            raise ValueError('top_k must be the same for both vector and BM25 queries')
        return self


class IndexHealthMetrics(BaseModel):
    """Model for index health metrics."""
    
    total_chunks: int = Field(ge=0, description="Total number of chunks")
    total_embeddings: int = Field(ge=0, description="Total number of embeddings")
    orphaned_embeddings: int = Field(ge=0, description="Embeddings without corresponding chunks")
    missing_embeddings: int = Field(ge=0, description="Chunks without embeddings")
    corrupted_chunks: int = Field(ge=0, description="Chunks with invalid data")
    index_size_mb: float = Field(ge=0.0, description="Index size in megabytes")
    last_health_check: datetime = Field(default_factory=datetime.now, description="Last health check timestamp")
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    
    @model_validator(mode='after')
    def validate_health_metrics(self) -> 'IndexHealthMetrics':
        """Validate health metrics consistency."""
        if self.orphaned_embeddings > self.total_embeddings:
            raise ValueError('Orphaned embeddings cannot exceed total embeddings')
        
        if self.missing_embeddings > self.total_chunks:
            raise ValueError('Missing embeddings cannot exceed total chunks')
        
        return self
    
    def is_healthy(self) -> bool:
        """Check if index is healthy."""
        return (
            self.orphaned_embeddings == 0
            and self.missing_embeddings == 0
            and self.corrupted_chunks == 0
        )
    
    def get_health_score(self) -> float:
        """Get health score between 0.0 and 1.0."""
        if self.total_chunks == 0:
            return 1.0
        
        issues = self.orphaned_embeddings + self.missing_embeddings + self.corrupted_chunks
        return max(0.0, 1.0 - (issues / self.total_chunks))


class DatabaseSchemaVersion(BaseModel):
    """Model for database schema versioning."""
    
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$', description="Semantic version")
    applied_at: datetime = Field(..., description="When version was applied")
    description: str = Field(..., min_length=1, description="Version description")
    migration_scripts: List[str] = Field(default_factory=list, description="Applied migration scripts")
    checksum: Optional[str] = Field(default=None, description="Schema checksum")
    
    @field_validator('version')
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError('Version must be in format "major.minor.patch"')
        
        for part in parts:
            try:
                int(part)
            except ValueError:
                raise ValueError(f'Version component "{part}" must be a number')
        
        return v
