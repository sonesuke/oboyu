"""State validation event handler."""

import logging
from pathlib import Path
from typing import Any, Optional

from ..events import IndexCorruptionDetectedEvent, IndexEvent
from .base import EventHandler

logger = logging.getLogger(__name__)


class IndexStateValidator(EventHandler):
    """Validates index integrity after operations."""
    
    def __init__(self, event_bus: Optional[Any] = None, database_path: Optional[Path] = None) -> None:  # noqa: ANN401
        """Initialize the state validator.
        
        Args:
            event_bus: Event bus for publishing corruption events
            database_path: Path to the database for validation

        """
        self.event_bus = event_bus
        self.database_path = database_path
    
    def handle(self, event: IndexEvent) -> None:
        """Handle an event by performing state validation.
        
        Args:
            event: The event to handle

        """
        try:
            if event.event_type == "indexing_completed":
                self._validate_indexing_integrity(event.operation_id)
            elif event.event_type == "database_cleared":
                self._ensure_clean_state(event.operation_id)
            elif event.event_type == "hnsw_index_created":
                self._validate_hnsw_index(event.operation_id)
            elif event.event_type == "bm25_index_updated":
                self._validate_bm25_index(event.operation_id)
        except Exception as e:
            logger.error(f"State validation failed for {event.event_type}: {e}")
            self._publish_corruption_event(
                operation_id=event.operation_id,
                corruption_type="validation_error",
                description=f"State validation failed: {e}",
                severity="high"
            )
    
    def _validate_indexing_integrity(self, operation_id: str) -> None:
        """Validate index integrity after indexing completion.
        
        Args:
            operation_id: The operation ID for context

        """
        if not self.database_path or not self.database_path.exists():
            logger.warning(f"Cannot validate integrity - database path not available: {self.database_path}")
            return
        
        try:
            # Import here to avoid circular dependencies
            import sqlite3
            
            with sqlite3.connect(self.database_path) as conn:
                # Check chunk count matches embedding count
                chunk_count_result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
                chunk_count = chunk_count_result[0] if chunk_count_result else 0
                
                # Check if embeddings table exists
                table_check = conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='embeddings'
                """).fetchone()
                
                if table_check:
                    embedding_count_result = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
                    embedding_count = embedding_count_result[0] if embedding_count_result else 0
                    
                    if chunk_count != embedding_count:
                        self._publish_corruption_event(
                            operation_id=operation_id,
                            corruption_type="chunk_embedding_mismatch",
                            description=f"Chunk count ({chunk_count}) doesn't match embedding count ({embedding_count})",
                            severity="high"
                        )
                        return
                
                # Check for orphaned records
                self._check_orphaned_records(conn, operation_id)
                
                # Check VSS extension status if available
                self._check_vss_extension_status(conn, operation_id)
                
                logger.debug(f"Index integrity validation passed for operation {operation_id}")
                
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            self._publish_corruption_event(
                operation_id=operation_id,
                corruption_type="database_error",
                description=f"Database integrity check failed: {e}",
                severity="medium"
            )
    
    def _ensure_clean_state(self, operation_id: str) -> None:
        """Ensure database is in clean state after clearing.
        
        Args:
            operation_id: The operation ID for context

        """
        if not self.database_path or not self.database_path.exists():
            logger.warning(f"Cannot validate clean state - database path not available: {self.database_path}")
            return
        
        try:
            import sqlite3
            
            with sqlite3.connect(self.database_path) as conn:
                # Check all main tables are empty
                tables_to_check = ['chunks', 'embeddings', 'documents', 'file_metadata']
                
                for table in tables_to_check:
                    # Check if table exists
                    table_check = conn.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name=?
                    """, (table,)).fetchone()
                    
                    if table_check:
                        count_result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                        count = count_result[0] if count_result else 0
                        
                        if count > 0:
                            self._publish_corruption_event(
                                operation_id=operation_id,
                                corruption_type="incomplete_clear",
                                description=f"Table {table} still contains {count} records after clear",
                                severity="medium"
                            )
                            return
                
                logger.debug(f"Clean state validation passed for operation {operation_id}")
                
        except Exception as e:
            logger.error(f"Clean state validation failed: {e}")
            self._publish_corruption_event(
                operation_id=operation_id,
                corruption_type="clean_state_error",
                description=f"Clean state validation failed: {e}",
                severity="medium"
            )
    
    def _validate_hnsw_index(self, operation_id: str) -> None:
        """Validate HNSW index consistency.
        
        Args:
            operation_id: The operation ID for context

        """
        if not self.database_path or not self.database_path.exists():
            return
        
        try:
            import sqlite3
            
            with sqlite3.connect(self.database_path) as conn:
                # Check if HNSW index exists and is accessible
                try:
                    # Try to query the vector search index
                    conn.execute("SELECT name FROM vss_modules").fetchall()
                    logger.debug(f"HNSW index validation passed for operation {operation_id}")
                except Exception as e:
                    logger.warning(f"HNSW index validation warning: {e}")
                    
        except Exception as e:
            logger.error(f"HNSW index validation failed: {e}")
            self._publish_corruption_event(
                operation_id=operation_id,
                corruption_type="hnsw_index_error",
                description=f"HNSW index validation failed: {e}",
                severity="medium"
            )
    
    def _validate_bm25_index(self, operation_id: str) -> None:
        """Validate BM25 index consistency.
        
        Args:
            operation_id: The operation ID for context

        """
        if not self.database_path or not self.database_path.exists():
            return
        
        try:
            import sqlite3
            
            with sqlite3.connect(self.database_path) as conn:
                # Check if BM25 tables exist and have data
                bm25_tables = ['bm25_index', 'bm25_statistics']
                
                for table in bm25_tables:
                    table_check = conn.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name=?
                    """, (table,)).fetchone()
                    
                    if table_check:
                        count_result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                        count = count_result[0] if count_result else 0
                        logger.debug(f"BM25 table {table} has {count} records")
                
                logger.debug(f"BM25 index validation passed for operation {operation_id}")
                
        except Exception as e:
            logger.error(f"BM25 index validation failed: {e}")
            self._publish_corruption_event(
                operation_id=operation_id,
                corruption_type="bm25_index_error",
                description=f"BM25 index validation failed: {e}",
                severity="medium"
            )
    
    def _check_orphaned_records(self, conn: Any, operation_id: str) -> None:  # noqa: ANN401
        """Check for orphaned records in the database.
        
        Args:
            conn: Database connection
            operation_id: The operation ID for context

        """
        try:
            # Check for embeddings without corresponding chunks
            orphaned_embeddings = conn.execute("""
                SELECT COUNT(*) FROM embeddings e
                LEFT JOIN chunks c ON e.chunk_id = c.chunk_id
                WHERE c.chunk_id IS NULL
            """).fetchone()
            
            if orphaned_embeddings and orphaned_embeddings[0] > 0:
                self._publish_corruption_event(
                    operation_id=operation_id,
                    corruption_type="orphaned_embeddings",
                    description=f"Found {orphaned_embeddings[0]} orphaned embeddings",
                    severity="medium"
                )
            
        except Exception as e:
            logger.warning(f"Could not check for orphaned records: {e}")
    
    def _check_vss_extension_status(self, conn: Any, operation_id: str) -> None:  # noqa: ANN401
        """Check VSS extension status.
        
        Args:
            conn: Database connection
            operation_id: The operation ID for context

        """
        try:
            # Check if VSS extension is loaded
            vss_modules = conn.execute("SELECT name FROM vss_modules").fetchall()
            if not vss_modules:
                logger.warning("VSS extension appears to be unavailable")
                
        except Exception as e:
            logger.debug(f"VSS extension check skipped: {e}")
    
    def _publish_corruption_event(
        self,
        operation_id: str,
        corruption_type: str,
        description: str,
        severity: str = "medium"
    ) -> None:
        """Publish an index corruption event.
        
        Args:
            operation_id: The operation ID for context
            corruption_type: Type of corruption detected
            description: Description of the corruption
            severity: Severity level (low, medium, high, critical)

        """
        if self.event_bus:
            corruption_event = IndexCorruptionDetectedEvent(
                operation_id=operation_id,
                corruption_type=corruption_type,
                description=description,
                severity=severity
            )
            self.event_bus.publish(corruption_event)
        else:
            logger.error(f"Index corruption detected but no event bus available: {description}")
