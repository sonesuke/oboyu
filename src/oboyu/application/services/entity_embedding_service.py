"""Entity Embedding Service for pre-computed entity embeddings in EDC process.

This service manages the lifecycle of entity embeddings, providing:
- Computation and storage of embeddings for entities
- Batch processing for efficient embedding generation
- Staleness detection and automatic updates
- Integration with existing embedding infrastructure
"""

import logging
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from oboyu.domain.models.knowledge_graph import Entity
from oboyu.indexer.services.embedding import EmbeddingService
from oboyu.ports.repositories.kg_repository import KGRepository

logger = logging.getLogger(__name__)


class EntityEmbeddingService:
    """Service for managing pre-computed entity embeddings.

    This service handles the complete lifecycle of entity embeddings:
    - Computing embeddings for entities using their textual descriptions
    - Storing embeddings persistently in the database
    - Detecting stale embeddings and updating them
    - Providing efficient batch operations for large-scale processing
    """

    def __init__(
        self,
        kg_repository: KGRepository,
        embedding_service: Optional[EmbeddingService] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 100,
        staleness_threshold_days: int = 30,
    ) -> None:
        """Initialize the EntityEmbeddingService.

        Args:
            kg_repository: Repository for KG entity operations
            embedding_service: Optional pre-configured embedding service
            embedding_model: Model name for embeddings (used if embedding_service is None)
            batch_size: Default batch size for processing
            staleness_threshold_days: Days after which embeddings are considered stale

        """
        self.kg_repository = kg_repository
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.staleness_threshold_days = staleness_threshold_days

        # Initialize embedding service if not provided
        if embedding_service is None:
            self.embedding_service = EmbeddingService(
                model_name=embedding_model,
                batch_size=batch_size,
                query_prefix="",  # No prefix for entity embeddings
                use_cache=True,
            )
        else:
            self.embedding_service = embedding_service

    def create_entity_description(self, entity: Entity) -> str:
        """Create a textual description for an entity to generate embeddings.

        This follows the EDC methodology by combining entity information into
        a coherent description suitable for embedding generation.

        Args:
            entity: Entity to create description for

        Returns:
            Textual description of the entity

        """
        parts = []

        # Add entity name and type
        if entity.name:
            parts.append(f"Entity: {entity.name}")

        if entity.entity_type:
            parts.append(f"Type: {entity.entity_type}")

        # Add definition if available
        if entity.definition:
            parts.append(f"Definition: {entity.definition}")

        # Add canonical name if different from name
        if entity.canonical_name and entity.canonical_name != entity.name:
            parts.append(f"Canonical Name: {entity.canonical_name}")

        # Add key properties
        if entity.properties:
            property_strings = []
            for key, value in entity.properties.items():
                if value and str(value).strip():
                    property_strings.append(f"{key}: {value}")

            if property_strings:
                parts.append(f"Properties: {', '.join(property_strings)}")

        # Join all parts with separator
        description = " | ".join(parts)

        # Fallback if no meaningful description can be created
        if not description.strip():
            description = f"Entity {entity.name or entity.id}"

        return description

    async def compute_and_store_embedding(self, entity: Entity) -> bool:
        """Compute and store embedding for a single entity.

        Args:
            entity: Entity to compute embedding for

        Returns:
            True if embedding was computed and stored successfully

        """
        try:
            # Create entity description
            description = self.create_entity_description(entity)

            # Generate embedding
            embeddings = self.embedding_service.generate_embeddings([description])

            if embeddings and len(embeddings) > 0:
                embedding_vector = embeddings[0]

                # Update entity with embedding information
                entity.embedding = embedding_vector.tolist()
                entity.embedding_model = self.embedding_model
                entity.embedding_updated_at = datetime.now()

                # Store updated entity
                await self.kg_repository.update_entity(entity)

                logger.debug(f"Computed and stored embedding for entity {entity.id}")
                return True
            else:
                logger.warning(f"Failed to generate embedding for entity {entity.id}")
                return False

        except Exception as e:
            logger.error(f"Error computing embedding for entity {entity.id}: {e}")
            return False

    async def batch_compute_embeddings(
        self,
        entities: List[Entity],
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """Compute embeddings for multiple entities in batch.

        Args:
            entities: List of entities to process
            skip_existing: Skip entities that already have embeddings
            progress_callback: Optional callback for progress updates

        Returns:
            Number of entities successfully processed

        """
        if not entities:
            return 0

        # Filter entities to process
        entities_to_process = []
        for entity in entities:
            if skip_existing and entity.embedding is not None:
                continue
            entities_to_process.append(entity)

        if not entities_to_process:
            logger.info("No entities need embedding computation")
            return 0

        logger.info(f"Computing embeddings for {len(entities_to_process)} entities")

        success_count = 0

        # Process in batches
        for i in range(0, len(entities_to_process), self.batch_size):
            batch = entities_to_process[i : i + self.batch_size]

            try:
                # Create descriptions for batch
                descriptions = [self.create_entity_description(entity) for entity in batch]

                # Generate embeddings for batch
                embeddings = self.embedding_service.generate_embeddings(descriptions)

                if len(embeddings) == len(batch):
                    # Update entities with embeddings
                    now = datetime.now()
                    for j, entity in enumerate(batch):
                        entity.embedding = embeddings[j].tolist()
                        entity.embedding_model = self.embedding_model
                        entity.embedding_updated_at = now

                    # Store updated entities
                    await self.kg_repository.batch_update_entities(batch)
                    success_count += len(batch)

                    logger.debug(f"Processed batch {i // self.batch_size + 1}, {len(batch)} entities")
                else:
                    logger.warning(f"Embedding count mismatch in batch {i // self.batch_size + 1}")

            except Exception as e:
                logger.error(f"Error processing batch {i // self.batch_size + 1}: {e}")

            # Report progress
            if progress_callback:
                progress_callback("entity_embeddings", i + len(batch), len(entities_to_process))

        logger.info(f"Successfully computed embeddings for {success_count}/{len(entities_to_process)} entities")
        return success_count

    async def get_entity_embedding(self, entity_id: str) -> Optional[NDArray[np.float32]]:
        """Get embedding for an entity by ID.

        Args:
            entity_id: ID of the entity

        Returns:
            Embedding vector or None if not available

        """
        try:
            entity = await self.kg_repository.get_entity_by_id(entity_id)
            if entity and entity.embedding:
                return np.array(entity.embedding, dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding for entity {entity_id}: {e}")
            return None

    async def update_stale_embeddings(self, max_age_days: Optional[int] = None) -> int:
        """Update embeddings that are older than the staleness threshold.

        Args:
            max_age_days: Maximum age in days (uses instance default if None)

        Returns:
            Number of embeddings updated

        """
        if max_age_days is None:
            max_age_days = self.staleness_threshold_days

        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        logger.info(f"Updating embeddings older than {max_age_days} days (before {cutoff_date})")

        try:
            # Get entities with stale embeddings
            # This assumes the repository has a method to find stale embeddings
            stale_entities = await self.kg_repository.find_entities_with_stale_embeddings(cutoff_date, self.embedding_model)

            if not stale_entities:
                logger.info("No stale embeddings found")
                return 0

            # Update stale embeddings
            updated_count = await self.batch_compute_embeddings(
                stale_entities,
                skip_existing=False,  # Force update even if embedding exists
            )

            logger.info(f"Updated {updated_count} stale embeddings")
            return updated_count

        except Exception as e:
            logger.error(f"Error updating stale embeddings: {e}")
            return 0

    async def rebuild_all_embeddings(self, entity_type: Optional[str] = None) -> int:
        """Rebuild all embeddings, optionally filtered by entity type.

        Args:
            entity_type: Optional entity type filter

        Returns:
            Number of embeddings rebuilt

        """
        logger.info(f"Rebuilding all embeddings{f' for type {entity_type}' if entity_type else ''}")

        try:
            # Get all entities
            if entity_type:
                entities = await self.kg_repository.find_entities_by_type(entity_type)
            else:
                entities = await self.kg_repository.get_all_entities()

            if not entities:
                logger.info("No entities found for embedding rebuild")
                return 0

            # Rebuild embeddings for all entities
            rebuilt_count = await self.batch_compute_embeddings(
                entities,
                skip_existing=False,  # Force rebuild even if embedding exists
            )

            logger.info(f"Rebuilt {rebuilt_count} embeddings")
            return rebuilt_count

        except Exception as e:
            logger.error(f"Error rebuilding embeddings: {e}")
            return 0

    async def get_embedding_statistics(self) -> dict:
        """Get statistics about entity embeddings.

        Returns:
            Dictionary with embedding statistics

        """
        try:
            stats = await self.kg_repository.get_embedding_statistics()

            # Add model information
            stats["current_model"] = self.embedding_model
            stats["embedding_dimensions"] = self.embedding_service.dimensions

            return stats

        except Exception as e:
            logger.error(f"Error getting embedding statistics: {e}")
            return {
                "error": str(e),
                "current_model": self.embedding_model,
                "embedding_dimensions": self.embedding_service.dimensions,
            }
