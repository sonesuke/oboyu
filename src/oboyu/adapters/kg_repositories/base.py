"""Base utilities for DuckDB Knowledge Graph repository.

This module provides common utilities and helper functions for
DuckDB-based KG repository implementation.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from duckdb import DuckDBPyConnection

from oboyu.domain.models.knowledge_graph import Entity, ProcessingStatus, Relation

logger = logging.getLogger(__name__)


class DuckDBKGRepositoryBase:
    """Base class with common utilities for DuckDB KG repository."""

    def _entity_from_row(self, row: Dict[str, Any]) -> Entity:
        """Convert database row to Entity object."""
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            definition=row["definition"],
            properties=json.loads(row["properties"]) if row["properties"] else {},
            chunk_id=row["chunk_id"],
            canonical_name=row["canonical_name"],
            merged_from=json.loads(row["merged_from"]) if row["merged_from"] else [],
            merge_confidence=row["merge_confidence"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"])
            if isinstance(row["created_at"], str)
            else (row["created_at"] if row["created_at"] else datetime.now()),
            updated_at=datetime.fromisoformat(row["updated_at"])
            if isinstance(row["updated_at"], str)
            else (row["updated_at"] if row["updated_at"] else datetime.now()),
        )

    def _relation_from_row(self, row: Dict[str, Any]) -> Relation:
        """Convert database row to Relation object."""
        return Relation(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            properties=json.loads(row["properties"]) if row["properties"] else {},
            chunk_id=row["chunk_id"],
            confidence=row["confidence"],
            created_at=datetime.fromisoformat(row["created_at"])
            if isinstance(row["created_at"], str)
            else (row["created_at"] if row["created_at"] else datetime.now()),
            updated_at=datetime.fromisoformat(row["updated_at"])
            if isinstance(row["updated_at"], str)
            else (row["updated_at"] if row["updated_at"] else datetime.now()),
        )

    def _processing_status_from_row(self, row: Dict[str, Any]) -> ProcessingStatus:
        """Convert database row to ProcessingStatus object."""
        return ProcessingStatus(
            chunk_id=row["chunk_id"],
            processed_at=datetime.fromisoformat(row["processed_at"])
            if isinstance(row["processed_at"], str)
            else (row["processed_at"] if row["processed_at"] else datetime.now()),
            processing_version=row["processing_version"],
            entity_count=row["entity_count"],
            relation_count=row["relation_count"],
            processing_time_ms=row["processing_time_ms"],
            model_used=row["model_used"],
            error_message=row["error_message"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"])
            if isinstance(row["created_at"], str)
            else (row["created_at"] if row["created_at"] else datetime.now()),
            updated_at=datetime.fromisoformat(row["updated_at"])
            if isinstance(row["updated_at"], str)
            else (row["updated_at"] if row["updated_at"] else datetime.now()),
        )

    def _get_columns_safely(self, connection: DuckDBPyConnection, operation_name: str) -> List[str]:
        """Get column descriptions safely with error handling."""
        description = connection.description
        if description is None:
            from oboyu.ports.repositories.kg_repository import RepositoryError

            raise RepositoryError("No column description available", operation_name)
        return [desc[0] for desc in description]
