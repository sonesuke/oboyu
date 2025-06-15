"""Knowledge Graph service for orchestrating extraction and storage.

This module provides the main service class for building and managing
the Property Graph Index, including batch processing and delta updates.
"""

import logging
from typing import Any, List, Optional

from oboyu.domain.models.knowledge_graph import Entity, KnowledgeGraphExtraction, ProcessingStatus
from oboyu.ports.repositories.kg_repository import KGRepository, RepositoryError
from oboyu.ports.services.entity_deduplication_service import EntityDeduplicationService
from oboyu.ports.services.kg_extraction_service import KGExtractionService

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service for building and managing knowledge graphs from text chunks."""

    def __init__(
        self,
        kg_repository: KGRepository,
        extraction_service: KGExtractionService,
        deduplication_service: Optional[EntityDeduplicationService] = None,
        processing_version: str = "1.0.0",
        confidence_threshold: float = 0.7,
        enable_deduplication: bool = True,
    ) -> None:
        """Initialize the Knowledge Graph service.

        Args:
            kg_repository: Repository for storing KG data
            extraction_service: Service for LLM-based extraction
            deduplication_service: Optional service for entity deduplication
            processing_version: Version of processing pipeline
            confidence_threshold: Minimum confidence for storing extractions
            enable_deduplication: Whether to enable entity deduplication

        """
        self.kg_repository = kg_repository
        self.extraction_service = extraction_service
        self.deduplication_service = deduplication_service
        self.processing_version = processing_version
        self.confidence_threshold = confidence_threshold
        self.enable_deduplication = enable_deduplication

    async def build_knowledge_graph(
        self,
        chunk_texts_and_ids: List[tuple[str, str]],
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[ProcessingStatus]:
        """Build knowledge graph from text chunks.

        Args:
            chunk_texts_and_ids: List of (text, chunk_id) tuples
            entity_types: Entity types to extract
            relation_types: Relation types to extract
            batch_size: Batch size for processing

        Returns:
            List of processing status results

        Raises:
            ServiceError: If processing fails

        """
        if not chunk_texts_and_ids:
            return []

        logger.info(f"Building knowledge graph from {len(chunk_texts_and_ids)} chunks")
        processing_results = []

        # Process in batches
        for i in range(0, len(chunk_texts_and_ids), batch_size):
            batch = chunk_texts_and_ids[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(chunk_texts_and_ids) + batch_size - 1) // batch_size}")

            batch_results = await self._process_batch(batch, entity_types, relation_types)
            processing_results.extend(batch_results)

        return processing_results

    async def _process_batch(
        self,
        batch: List[tuple[str, str]],
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> List[ProcessingStatus]:
        """Process a batch of chunks."""
        results = []

        for text, chunk_id in batch:
            try:
                # Extract knowledge graph
                extraction = await self.extraction_service.extract_knowledge_graph(text, chunk_id, entity_types=entity_types, relation_types=relation_types)

                # Process the extraction
                status = await self._process_extraction(extraction)
                results.append(status)

            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_id}: {e}")
                # Create error status
                error_status = ProcessingStatus(
                    chunk_id=chunk_id,
                    processing_version=self.processing_version,
                    entity_count=0,
                    relation_count=0,
                    error_message=str(e),
                    status="error",
                )
                try:
                    await self.kg_repository.save_processing_status(error_status)
                except RepositoryError as repo_error:
                    logger.error(f"Failed to save error status: {repo_error}")

                results.append(error_status)

        return results

    async def _process_extraction(self, extraction: KnowledgeGraphExtraction) -> ProcessingStatus:
        """Process a single extraction result."""
        try:
            # Filter by confidence threshold
            high_confidence_entities = [entity for entity in extraction.entities if entity.confidence >= self.confidence_threshold]
            high_confidence_relations = [relation for relation in extraction.relations if relation.confidence >= self.confidence_threshold]

            logger.info(
                f"Chunk {extraction.chunk_id}: Extracted {len(extraction.entities)} entities "
                f"({len(high_confidence_entities)} high confidence) and {len(extraction.relations)} relations "
                f"({len(high_confidence_relations)} high confidence)"
            )

            # Apply entity deduplication if enabled
            if self.enable_deduplication and self.deduplication_service and high_confidence_entities:
                try:
                    # Get existing entities for deduplication comparison
                    existing_entities = []
                    for entity in high_confidence_entities:
                        similar_existing = await self.kg_repository.get_entities_by_type(entity.entity_type, limit=100)
                        existing_entities.extend(similar_existing)

                    # Deduplicate new entities against existing ones
                    all_entities = high_confidence_entities + existing_entities
                    deduplicated_entities = await self.deduplication_service.deduplicate_entities(
                        all_entities, similarity_threshold=0.85, verification_threshold=0.8
                    )

                    # Filter to get only the newly processed entities (with updated canonical info)
                    new_chunk_entities = [
                        e for e in deduplicated_entities if e.chunk_id == extraction.chunk_id or e.id in [ne.id for ne in high_confidence_entities]
                    ]

                    high_confidence_entities = new_chunk_entities
                    logger.info(f"Deduplication reduced entities from {len(all_entities)} to {len(deduplicated_entities)}")

                except Exception as e:
                    logger.warning(f"Entity deduplication failed, proceeding without: {e}")

            # Save entities first, then relations (to ensure foreign key constraints)
            if high_confidence_entities:
                await self.kg_repository.save_entities(high_confidence_entities)
                logger.info(f"Saved {len(high_confidence_entities)} entities")

            if high_confidence_relations:
                # Verify all referenced entities exist before saving relations
                entity_ids = {e.id for e in high_confidence_entities}
                valid_relations = []

                for relation in high_confidence_relations:
                    if relation.source_id in entity_ids and relation.target_id in entity_ids:
                        valid_relations.append(relation)
                    else:
                        logger.warning(f"Skipping relation with missing entity IDs: {relation.source_id} -> {relation.target_id}")

                if valid_relations:
                    await self.kg_repository.save_relations(valid_relations)
                    logger.info(f"Saved {len(valid_relations)} relations (skipped {len(high_confidence_relations) - len(valid_relations)})")
                else:
                    logger.info("No valid relations to save")

            # Create processing status
            status = ProcessingStatus(
                chunk_id=extraction.chunk_id,
                processing_version=self.processing_version,
                entity_count=len(high_confidence_entities),
                relation_count=len(high_confidence_relations),
                processing_time_ms=extraction.processing_time_ms,
                model_used=extraction.model_used,
                status="completed",
            )

            # Save processing status
            await self.kg_repository.save_processing_status(status)

            return status

        except Exception as e:
            logger.error(f"Failed to process extraction for chunk {extraction.chunk_id}: {e}")
            error_status = ProcessingStatus(
                chunk_id=extraction.chunk_id,
                processing_version=self.processing_version,
                entity_count=0,
                relation_count=0,
                error_message=str(e),
                status="error",
            )
            try:
                await self.kg_repository.save_processing_status(error_status)
            except RepositoryError as repo_error:
                logger.error(f"Failed to save error status: {repo_error}")

            return error_status

    async def update_knowledge_graph_delta(
        self,
        chunk_id: str,
        text: str,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> ProcessingStatus:
        """Update knowledge graph for a single chunk (delta update).

        Args:
            chunk_id: Chunk identifier
            text: Chunk text content
            entity_types: Entity types to extract
            relation_types: Relation types to extract

        Returns:
            Processing status result

        Raises:
            ServiceError: If update fails

        """
        try:
            # Check if already processed with current version
            existing_status = await self.kg_repository.get_processing_status(chunk_id)
            if existing_status and existing_status.processing_version == self.processing_version and existing_status.status == "completed":
                logger.info(f"Chunk {chunk_id} already processed with version {self.processing_version}")
                return existing_status

            # Delete existing entities and relations for this chunk
            await self.kg_repository.delete_entities_by_chunk_id(chunk_id)
            await self.kg_repository.delete_relations_by_chunk_id(chunk_id)

            # Extract new knowledge graph
            extraction = await self.extraction_service.extract_knowledge_graph(text, chunk_id, entity_types=entity_types, relation_types=relation_types)

            # Process the extraction
            return await self._process_extraction(extraction)

        except Exception as e:
            logger.error(f"Failed to update knowledge graph for chunk {chunk_id}: {e}")
            raise ServiceError(f"Delta update failed: {e}")

    async def get_unprocessed_chunks(self, limit: Optional[int] = None) -> List[str]:
        """Get chunk IDs that need processing.

        Args:
            limit: Optional limit on number of results

        Returns:
            List of unprocessed chunk IDs

        """
        try:
            return await self.kg_repository.get_unprocessed_chunks(self.processing_version, limit)
        except RepositoryError as e:
            logger.error(f"Failed to get unprocessed chunks: {e}")
            raise ServiceError(f"Failed to get unprocessed chunks: {e}")

    async def get_knowledge_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph.

        Returns:
            Dictionary with entity count, relation count, etc.

        """
        try:
            entity_count = await self.kg_repository.get_entity_count()
            relation_count = await self.kg_repository.get_relation_count()

            return {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "processing_version": self.processing_version,
            }
        except RepositoryError as e:
            logger.error(f"Failed to get KG stats: {e}")
            raise ServiceError(f"Failed to get statistics: {e}")

    async def deduplicate_all_entities(
        self,
        entity_type: Optional[str] = None,
        similarity_threshold: float = 0.85,
        verification_threshold: float = 0.8,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Perform global entity deduplication across all stored entities.

        Args:
            entity_type: Optional entity type to deduplicate (if None, all types)
            similarity_threshold: Vector similarity threshold
            verification_threshold: LLM verification threshold
            batch_size: Processing batch size

        Returns:
            Dictionary with deduplication statistics

        Raises:
            ServiceError: If deduplication fails

        """
        if not self.deduplication_service:
            raise ServiceError("Deduplication service not configured")

        try:
            logger.info(f"Starting global entity deduplication for type: {entity_type or 'all'}")

            if entity_type:
                # Deduplicate specific entity type
                entities = await self.kg_repository.get_entities_by_type(entity_type)
                entity_types_to_process = [entity_type]
            else:
                # Get all entity types
                all_entities = []
                entity_types_to_process = ["PERSON", "COMPANY", "ORGANIZATION", "PRODUCT", "LOCATION", "EVENT"]

                for etype in entity_types_to_process:
                    type_entities = await self.kg_repository.get_entities_by_type(etype)
                    all_entities.extend(type_entities)

                entities = all_entities

            original_count = len(entities)
            if original_count == 0:
                return {"original_count": 0, "deduplicated_count": 0, "merged_count": 0}

            # Process entities in batches for large datasets
            all_deduplicated = []
            total_merged = 0

            for i in range(0, len(entities), batch_size):
                batch = entities[i : i + batch_size]
                logger.info(f"Deduplicating batch {i // batch_size + 1}/{(len(entities) + batch_size - 1) // batch_size}")

                deduplicated_batch = await self.deduplication_service.deduplicate_entities(batch, similarity_threshold, verification_threshold)

                batch_merged = len(batch) - len(deduplicated_batch)
                total_merged += batch_merged

                all_deduplicated.extend(deduplicated_batch)

            # Save deduplicated entities back to database
            if entity_type:
                # Delete old entities of this type and save new ones
                for entity in entities:
                    await self.kg_repository.delete_entities_by_chunk_id(entity.chunk_id or "")

            await self.kg_repository.save_entities(all_deduplicated)

            final_count = len(all_deduplicated)

            logger.info(f"Global deduplication complete: {original_count} → {final_count} entities ({total_merged} merged)")

            return {
                "original_count": original_count,
                "deduplicated_count": final_count,
                "merged_count": total_merged,
            }

        except Exception as e:
            logger.error(f"Global entity deduplication failed: {e}")
            raise ServiceError(f"Global deduplication failed: {e}")

    async def find_duplicate_entities(
        self,
        entity_name: str,
        entity_type: Optional[str] = None,
        similarity_threshold: float = 0.85,
    ) -> List[tuple[str, str, float]]:
        """Find potential duplicate entities for a given entity name.

        Args:
            entity_name: Name of entity to find duplicates for
            entity_type: Optional entity type filter
            similarity_threshold: Minimum similarity score

        Returns:
            List of (entity_id, entity_name, similarity_score) tuples

        Raises:
            ServiceError: If search fails

        """
        if not self.deduplication_service:
            raise ServiceError("Deduplication service not configured")

        try:
            # Create temporary entity for comparison
            temp_entity = Entity(name=entity_name, entity_type=entity_type or "UNKNOWN")

            # Get candidate entities
            if entity_type:
                candidates = await self.kg_repository.get_entities_by_type(entity_type)
            else:
                # Search across all types
                candidates = []
                for etype in ["PERSON", "COMPANY", "ORGANIZATION", "PRODUCT", "LOCATION"]:
                    type_entities = await self.kg_repository.get_entities_by_type(etype, limit=50)
                    candidates.extend(type_entities)

            # Find similar entities
            similar_entities = await self.deduplication_service.find_similar_entities(temp_entity, candidates, similarity_threshold)

            # Format results
            results = [(entity.id, entity.name, similarity_score) for entity, similarity_score in similar_entities]

            logger.info(f"Found {len(results)} potential duplicates for '{entity_name}'")
            return results

        except Exception as e:
            logger.error(f"Duplicate search failed for '{entity_name}': {e}")
            raise ServiceError(f"Duplicate search failed: {e}")


class ServiceError(Exception):
    """Exception raised when Knowledge Graph service operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        """Initialize service error.

        Args:
            message: Error description
            operation: Optional operation that failed

        """
        super().__init__(message)
        self.operation = operation
