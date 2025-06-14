"""Entity deduplication service interface.

This module defines the port interface for EDC (Extract-Define-Canonicalize) based
entity deduplication following the research paper methodology.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from oboyu.domain.models.knowledge_graph import Entity


class EntityDeduplicationService(ABC):
    """Abstract interface for entity deduplication using EDC methodology."""

    @abstractmethod
    async def generate_entity_definition(self, entity: Entity, context: Optional[str] = None) -> str:
        """Generate a natural language definition for an entity (Define step).

        Args:
            entity: Entity to define
            context: Optional context text for better definition generation

        Returns:
            Natural language definition of the entity

        Raises:
            DeduplicationError: If definition generation fails

        """

    @abstractmethod
    async def find_similar_entities(
        self,
        entity: Entity,
        candidate_entities: List[Entity],
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Entity, float]]:
        """Find similar entities using vector similarity (Canonicalize step).

        Args:
            entity: Target entity to find matches for
            candidate_entities: Pool of entities to search in
            similarity_threshold: Minimum similarity score for matches

        Returns:
            List of (similar_entity, similarity_score) tuples ordered by similarity

        Raises:
            DeduplicationError: If similarity computation fails

        """

    @abstractmethod
    async def verify_entity_merge(
        self,
        entity1: Entity,
        entity2: Entity,
        similarity_score: float,
    ) -> Tuple[bool, float]:
        """Verify if two entities should be merged using LLM validation.

        Args:
            entity1: First entity
            entity2: Second entity
            similarity_score: Vector similarity score

        Returns:
            Tuple of (should_merge, confidence_score)

        Raises:
            DeduplicationError: If verification fails

        """

    @abstractmethod
    async def canonicalize_entity(
        self,
        entities_to_merge: List[Entity],
        merge_confidences: List[float],
    ) -> Entity:
        """Create canonical entity from multiple similar entities.

        Args:
            entities_to_merge: List of entities to merge
            merge_confidences: Confidence scores for each merge

        Returns:
            Canonical entity with merged information

        Raises:
            DeduplicationError: If canonicalization fails

        """

    @abstractmethod
    async def deduplicate_entities(
        self,
        entities: List[Entity],
        similarity_threshold: float = 0.85,
        verification_threshold: float = 0.8,
    ) -> List[Entity]:
        """Perform complete entity deduplication pipeline.

        Args:
            entities: List of entities to deduplicate
            similarity_threshold: Vector similarity threshold
            verification_threshold: LLM verification threshold

        Returns:
            Deduplicated list of canonical entities

        Raises:
            DeduplicationError: If deduplication pipeline fails

        """

    @abstractmethod
    async def normalize_entity_name(self, name: str, entity_type: str) -> str:
        """Normalize entity name for Japanese text variations.

        Args:
            name: Original entity name
            entity_type: Type of entity for context-specific normalization

        Returns:
            Normalized entity name

        Raises:
            DeduplicationError: If normalization fails

        """


class DeduplicationError(Exception):
    """Exception raised when entity deduplication operations fail."""

    def __init__(self, message: str, entity_id: Optional[str] = None) -> None:
        """Initialize deduplication error.

        Args:
            message: Error description
            entity_id: Optional entity ID where error occurred

        """
        super().__init__(message)
        self.entity_id = entity_id
