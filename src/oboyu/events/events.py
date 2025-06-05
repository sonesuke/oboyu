"""Core event type definitions for index state management."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol


class IndexEvent(Protocol):
    """Protocol defining the interface for all index events."""
    
    timestamp: datetime
    operation_id: str
    
    @property
    def event_type(self) -> str:
        """The type identifier for this event."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        ...


@dataclass
class BaseIndexEvent(ABC, IndexEvent):
    """Base class for all index events."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """The type identifier for this event."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type
        return data


@dataclass
class IndexingStartedEvent(BaseIndexEvent):
    """Event published when indexing operation starts."""
    
    document_count: int = 0
    total_size_bytes: int = 0
    source_path: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "indexing_started"


@dataclass
class IndexingCompletedEvent(BaseIndexEvent):
    """Event published when indexing operation completes successfully."""
    
    document_count: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "indexing_completed"


@dataclass
class IndexingFailedEvent(BaseIndexEvent):
    """Event published when indexing operation fails."""
    
    error: str = ""
    error_type: str = ""
    document_count_processed: int = 0
    duration_seconds: float = 0.0
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "indexing_failed"


@dataclass
class DocumentProcessedEvent(BaseIndexEvent):
    """Event published when a single document is processed."""
    
    document_path: str = ""
    chunks_created: int = 0
    embeddings_generated: int = 0
    processing_time_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "document_processed"


@dataclass
class DatabaseClearedEvent(BaseIndexEvent):
    """Event published when database is cleared."""
    
    records_deleted: int = 0
    tables_cleared: List[str] = field(default_factory=list)
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "database_cleared"


@dataclass
class DatabaseClearFailedEvent(BaseIndexEvent):
    """Event published when database clear operation fails."""
    
    error: str = ""
    error_type: str = ""
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "database_clear_failed"


@dataclass
class IndexCorruptionDetectedEvent(BaseIndexEvent):
    """Event published when index corruption is detected."""
    
    corruption_type: str = ""
    affected_tables: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "index_corruption_detected"


@dataclass
class IndexHealthCheckEvent(BaseIndexEvent):
    """Event published when index health check is performed."""
    
    health_status: str = "healthy"  # healthy, degraded, unhealthy
    checks_performed: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "index_health_check"


@dataclass
class EmbeddingGeneratedEvent(BaseIndexEvent):
    """Event published when embeddings are generated."""
    
    chunk_count: int = 0
    embedding_model: str = ""
    processing_time_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "embedding_generated"


@dataclass
class BM25IndexUpdatedEvent(BaseIndexEvent):
    """Event published when BM25 index is updated."""
    
    terms_indexed: int = 0
    documents_indexed: int = 0
    processing_time_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "bm25_index_updated"


@dataclass
class HNSWIndexCreatedEvent(BaseIndexEvent):
    """Event published when HNSW index is created or updated."""
    
    vector_count: int = 0
    index_parameters: Dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "hnsw_index_created"


@dataclass
class DatabaseConnectionEvent(BaseIndexEvent):
    """Event published for database connection lifecycle."""
    
    connection_status: str = ""  # connected, disconnected, failed
    database_path: str = ""
    error: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        """Return the event type identifier."""
        return "database_connection"
