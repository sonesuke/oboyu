"""Tests for event type definitions."""

import pytest
from datetime import datetime

from oboyu.events.events import (
    IndexingStartedEvent,
    IndexingCompletedEvent,
    IndexingFailedEvent,
    DocumentProcessedEvent,
    DatabaseClearedEvent,
    DatabaseClearFailedEvent,
    IndexCorruptionDetectedEvent,
    IndexHealthCheckEvent,
    EmbeddingGeneratedEvent,
    BM25IndexUpdatedEvent,
    HNSWIndexCreatedEvent,
    DatabaseConnectionEvent,
)


class TestEventTypes:
    """Test cases for event type definitions."""
    
    def test_indexing_started_event(self):
        """Test IndexingStartedEvent creation and properties."""
        event = IndexingStartedEvent(
            document_count=10,
            total_size_bytes=1024,
            source_path="/test/path"
        )
        
        assert event.event_type == "indexing_started"
        assert event.document_count == 10
        assert event.total_size_bytes == 1024
        assert event.source_path == "/test/path"
        assert isinstance(event.timestamp, datetime)
        assert len(event.operation_id) > 0
    
    def test_indexing_completed_event(self):
        """Test IndexingCompletedEvent creation and properties."""
        event = IndexingCompletedEvent(
            document_count=10,
            chunks_created=50,
            embeddings_generated=50,
            duration_seconds=5.5,
            success=True
        )
        
        assert event.event_type == "indexing_completed"
        assert event.document_count == 10
        assert event.chunks_created == 50
        assert event.embeddings_generated == 50
        assert event.duration_seconds == 5.5
        assert event.success is True
    
    def test_indexing_failed_event(self):
        """Test IndexingFailedEvent creation and properties."""
        event = IndexingFailedEvent(
            error="Test error",
            error_type="ValueError",
            document_count_processed=5,
            duration_seconds=2.5
        )
        
        assert event.event_type == "indexing_failed"
        assert event.error == "Test error"
        assert event.error_type == "ValueError"
        assert event.document_count_processed == 5
        assert event.duration_seconds == 2.5
    
    def test_document_processed_event(self):
        """Test DocumentProcessedEvent creation and properties."""
        event = DocumentProcessedEvent(
            document_path="/test/doc.txt",
            chunks_created=3,
            embeddings_generated=3,
            processing_time_seconds=1.2,
            success=True
        )
        
        assert event.event_type == "document_processed"
        assert event.document_path == "/test/doc.txt"
        assert event.chunks_created == 3
        assert event.embeddings_generated == 3
        assert event.processing_time_seconds == 1.2
        assert event.success is True
        assert event.error is None
    
    def test_database_cleared_event(self):
        """Test DatabaseClearedEvent creation and properties."""
        event = DatabaseClearedEvent(
            records_deleted=100,
            tables_cleared=["chunks", "embeddings"]
        )
        
        assert event.event_type == "database_cleared"
        assert event.records_deleted == 100
        assert event.tables_cleared == ["chunks", "embeddings"]
    
    def test_database_clear_failed_event(self):
        """Test DatabaseClearFailedEvent creation and properties."""
        event = DatabaseClearFailedEvent(
            error="Permission denied",
            error_type="PermissionError"
        )
        
        assert event.event_type == "database_clear_failed"
        assert event.error == "Permission denied"
        assert event.error_type == "PermissionError"
    
    def test_index_corruption_detected_event(self):
        """Test IndexCorruptionDetectedEvent creation and properties."""
        event = IndexCorruptionDetectedEvent(
            corruption_type="data_mismatch",
            affected_tables=["chunks", "embeddings"],
            severity="high",
            description="Chunk count doesn't match embedding count"
        )
        
        assert event.event_type == "index_corruption_detected"
        assert event.corruption_type == "data_mismatch"
        assert event.affected_tables == ["chunks", "embeddings"]
        assert event.severity == "high"
        assert event.description == "Chunk count doesn't match embedding count"
    
    def test_index_health_check_event(self):
        """Test IndexHealthCheckEvent creation and properties."""
        event = IndexHealthCheckEvent(
            health_status="healthy",
            checks_performed=["connection", "integrity"],
            issues_found=[],
            total_documents=50,
            total_chunks=200,
            total_embeddings=200
        )
        
        assert event.event_type == "index_health_check"
        assert event.health_status == "healthy"
        assert event.checks_performed == ["connection", "integrity"]
        assert event.issues_found == []
        assert event.total_documents == 50
        assert event.total_chunks == 200
        assert event.total_embeddings == 200
    
    def test_embedding_generated_event(self):
        """Test EmbeddingGeneratedEvent creation and properties."""
        event = EmbeddingGeneratedEvent(
            chunk_count=25,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            processing_time_seconds=3.5,
            success=True
        )
        
        assert event.event_type == "embedding_generated"
        assert event.chunk_count == 25
        assert event.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert event.processing_time_seconds == 3.5
        assert event.success is True
        assert event.error is None
    
    def test_bm25_index_updated_event(self):
        """Test BM25IndexUpdatedEvent creation and properties."""
        event = BM25IndexUpdatedEvent(
            terms_indexed=1500,
            documents_indexed=10,
            processing_time_seconds=2.0,
            success=True
        )
        
        assert event.event_type == "bm25_index_updated"
        assert event.terms_indexed == 1500
        assert event.documents_indexed == 10
        assert event.processing_time_seconds == 2.0
        assert event.success is True
        assert event.error is None
    
    def test_hnsw_index_created_event(self):
        """Test HNSWIndexCreatedEvent creation and properties."""
        event = HNSWIndexCreatedEvent(
            vector_count=200,
            index_parameters={"ef_construction": 200, "M": 16},
            processing_time_seconds=1.5,
            success=True
        )
        
        assert event.event_type == "hnsw_index_created"
        assert event.vector_count == 200
        assert event.index_parameters == {"ef_construction": 200, "M": 16}
        assert event.processing_time_seconds == 1.5
        assert event.success is True
        assert event.error is None
    
    def test_database_connection_event(self):
        """Test DatabaseConnectionEvent creation and properties."""
        event = DatabaseConnectionEvent(
            connection_status="connected",
            database_path="/test/db.duckdb"
        )
        
        assert event.event_type == "database_connection"
        assert event.connection_status == "connected"
        assert event.database_path == "/test/db.duckdb"
        assert event.error is None
    
    def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = IndexingStartedEvent(
            document_count=5,
            total_size_bytes=1024,
            source_path="/test"
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "indexing_started"
        assert event_dict["document_count"] == 5
        assert event_dict["total_size_bytes"] == 1024
        assert event_dict["source_path"] == "/test"
        assert "timestamp" in event_dict
        assert "operation_id" in event_dict
    
    def test_event_defaults(self):
        """Test event creation with default values."""
        event = IndexingStartedEvent()
        
        assert event.event_type == "indexing_started"
        assert event.document_count == 0
        assert event.total_size_bytes == 0
        assert event.source_path is None
        assert isinstance(event.timestamp, datetime)
        assert len(event.operation_id) > 0