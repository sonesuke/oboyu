"""Knowledge Graph repository interface.

This module defines the repository interface for storing and retrieving
knowledge graph entities and relations, following the hexagonal architecture pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from oboyu.domain.models.knowledge_graph import Entity, ProcessingStatus, Relation


class KGRepository(ABC):
    """Abstract interface for knowledge graph data persistence."""

    @abstractmethod
    async def save_entity(self, entity: Entity) -> None:
        """Save an entity to the knowledge graph.

        Args:
            entity: Entity to save

        Raises:
            RepositoryError: If save operation fails

        """

    @abstractmethod
    async def save_entities(self, entities: List[Entity]) -> None:
        """Save multiple entities to the knowledge graph.

        Args:
            entities: List of entities to save

        Raises:
            RepositoryError: If save operation fails

        """

    @abstractmethod
    async def save_relation(self, relation: Relation) -> None:
        """Save a relation to the knowledge graph.

        Args:
            relation: Relation to save

        Raises:
            RepositoryError: If save operation fails

        """

    @abstractmethod
    async def save_relations(self, relations: List[Relation]) -> None:
        """Save multiple relations to the knowledge graph.

        Args:
            relations: List of relations to save

        Raises:
            RepositoryError: If save operation fails

        """

    @abstractmethod
    async def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by its ID.

        Args:
            entity_id: Unique entity identifier

        Returns:
            Entity if found, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails

        """

    @abstractmethod
    async def get_entities_by_chunk_id(self, chunk_id: str) -> List[Entity]:
        """Retrieve all entities extracted from a specific chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            List of entities from the chunk

        Raises:
            RepositoryError: If retrieval operation fails

        """

    @abstractmethod
    async def get_relations_by_chunk_id(self, chunk_id: str) -> List[Relation]:
        """Retrieve all relations extracted from a specific chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            List of relations from the chunk

        Raises:
            RepositoryError: If retrieval operation fails

        """

    @abstractmethod
    async def get_entities_by_type(self, entity_type: str, limit: Optional[int] = None) -> List[Entity]:
        """Retrieve entities by type.

        Args:
            entity_type: Type of entities to retrieve
            limit: Optional limit on number of results

        Returns:
            List of entities of the specified type

        Raises:
            RepositoryError: If retrieval operation fails

        """

    @abstractmethod
    async def get_relations_by_type(self, relation_type: str, limit: Optional[int] = None) -> List[Relation]:
        """Retrieve relations by type.

        Args:
            relation_type: Type of relations to retrieve
            limit: Optional limit on number of results

        Returns:
            List of relations of the specified type

        Raises:
            RepositoryError: If retrieval operation fails

        """

    @abstractmethod
    async def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> List[Entity]:
        """Get neighboring entities connected through relations.

        Args:
            entity_id: Source entity ID
            max_hops: Maximum number of hops to traverse

        Returns:
            List of connected entities

        Raises:
            RepositoryError: If traversal operation fails

        """

    @abstractmethod
    async def search_entities_by_name(self, name_pattern: str, limit: Optional[int] = None) -> List[Entity]:
        """Search entities by name pattern.

        Args:
            name_pattern: Pattern to match against entity names
            limit: Optional limit on number of results

        Returns:
            List of matching entities

        Raises:
            RepositoryError: If search operation fails

        """

    @abstractmethod
    async def save_processing_status(self, status: ProcessingStatus) -> None:
        """Save processing status for a chunk.

        Args:
            status: Processing status to save

        Raises:
            RepositoryError: If save operation fails

        """

    @abstractmethod
    async def get_processing_status(self, chunk_id: str) -> Optional[ProcessingStatus]:
        """Get processing status for a chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Processing status if exists, None otherwise

        Raises:
            RepositoryError: If retrieval operation fails

        """

    @abstractmethod
    async def get_unprocessed_chunks(self, processing_version: str, limit: Optional[int] = None) -> List[str]:
        """Get chunk IDs that haven't been processed with the specified version.

        Args:
            processing_version: Version of processing pipeline
            limit: Optional limit on number of results

        Returns:
            List of unprocessed chunk IDs

        Raises:
            RepositoryError: If query operation fails

        """

    @abstractmethod
    async def delete_entities_by_chunk_id(self, chunk_id: str) -> int:
        """Delete all entities associated with a chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Number of entities deleted

        Raises:
            RepositoryError: If deletion operation fails

        """

    @abstractmethod
    async def delete_relations_by_chunk_id(self, chunk_id: str) -> int:
        """Delete all relations associated with a chunk.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Number of relations deleted

        Raises:
            RepositoryError: If deletion operation fails

        """

    @abstractmethod
    async def get_entity_count(self) -> int:
        """Get total number of entities in the knowledge graph.

        Returns:
            Total entity count

        Raises:
            RepositoryError: If count operation fails

        """

    @abstractmethod
    async def get_relation_count(self) -> int:
        """Get total number of relations in the knowledge graph.

        Returns:
            Total relation count

        Raises:
            RepositoryError: If count operation fails

        """


class RepositoryError(Exception):
    """Exception raised when repository operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        """Initialize repository error.

        Args:
            message: Error description
            operation: Optional operation that failed

        """
        super().__init__(message)
        self.operation = operation
