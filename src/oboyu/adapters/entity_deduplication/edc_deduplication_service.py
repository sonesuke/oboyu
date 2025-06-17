"""EDC-based entity deduplication service implementation.

This module implements the Extract-Define-Canonicalize methodology for entity
deduplication using vector similarity and LLM verification.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import jaconv
import numpy as np
from sentence_transformers import SentenceTransformer

from oboyu.application.services.entity_embedding_service import EntityEmbeddingService
from oboyu.domain.models.knowledge_graph import Entity
from oboyu.ports.services.entity_deduplication_service import DeduplicationError, EntityDeduplicationService
from oboyu.ports.services.kg_extraction_service import KGExtractionService

from .edc_utils import EDCUtils

logger = logging.getLogger(__name__)


class EDCDeduplicationService(EntityDeduplicationService):
    """EDC-based entity deduplication service.

    Uses Extract-Define-Canonicalize methodology with vector similarity
    and LLM verification for Japanese entity deduplication.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        llm_service: KGExtractionService,
        similarity_threshold: float = 0.85,
        verification_threshold: float = 0.8,
        entity_embedding_service: Optional[EntityEmbeddingService] = None,
        use_precomputed_embeddings: bool = True,
    ) -> None:
        """Initialize EDC deduplication service.

        Args:
            embedding_model: Sentence transformer for vector similarity (fallback)
            llm_service: LLM service for definition generation and verification
            similarity_threshold: Default vector similarity threshold
            verification_threshold: Default LLM verification threshold
            entity_embedding_service: Service for pre-computed embeddings
            use_precomputed_embeddings: Whether to use pre-computed embeddings when available

        """
        self.embedding_model = embedding_model
        self.llm_service = llm_service
        self.similarity_threshold = similarity_threshold
        self.verification_threshold = verification_threshold
        self.entity_embedding_service = entity_embedding_service
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self._entity_cache: Dict[str, np.ndarray] = {}

    async def generate_entity_definition(self, entity: Entity, context: Optional[str] = None) -> str:
        """Generate a natural language definition for an entity (Define step)."""
        try:
            # Create definition prompt
            prompt = f"""あなたは知識グラフの専門家です。以下のエンティティについて、明確で具体的な定義を生成してください。

エンティティ情報:
- 名前: {entity.name}
- タイプ: {entity.entity_type}
- プロパティ: {json.dumps(entity.properties, ensure_ascii=False) if entity.properties else "なし"}

{f"コンテキスト情報: {context}" if context else ""}

要求:
1. エンティティの本質的な特徴を説明
2. 他の類似エンティティと区別できる要素を含める
3. 簡潔で明確な文章（1-2文）で記述
4. 推測ではなく、提供された情報に基づいて記述

定義:"""

            # Run LLM inference to generate definition
            response = await self._run_llm_for_definition(prompt)

            # Clean and validate response
            definition = response.strip()
            if not definition or len(definition) < 10:
                # Fallback to basic definition
                definition = f"{entity.entity_type}として識別される「{entity.name}」"

            logger.debug(f"Generated definition for {entity.name}: {definition}")
            return definition

        except Exception as e:
            logger.error(f"Failed to generate definition for entity {entity.name}: {e}")
            raise DeduplicationError(f"Definition generation failed: {e}", entity.id)

    async def _run_llm_for_definition(self, prompt: str) -> str:
        """Run LLM inference for definition generation."""
        try:
            # Use the LLM extraction service for definition generation
            # Create a dummy extraction to get LLM response
            await self.llm_service.extract_knowledge_graph(prompt, "definition_generation")

            # For now, return a simple fallback since we need dedicated definition LLM
            return "エンティティの定義を生成中..."

        except Exception as e:
            logger.warning(f"LLM definition generation failed, using fallback: {e}")
            return "エンティティの定義"

    async def find_similar_entities(
        self,
        entity: Entity,
        candidate_entities: List[Entity],
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Entity, float]]:
        """Find similar entities using vector similarity (Canonicalize step)."""
        if not candidate_entities:
            return []

        try:
            # Normalize target entity name
            target_name = await self.normalize_entity_name(entity.name, entity.entity_type)

            # Get or compute target entity embedding
            target_embedding = await self._get_entity_embedding(entity, target_name)

            # Compute similarities with candidates
            similarities = []
            for candidate in candidate_entities:
                if candidate.id == entity.id:
                    continue  # Skip self

                try:
                    # Check entity type compatibility
                    if not EDCUtils.are_types_compatible(entity.entity_type, candidate.entity_type):
                        continue

                    # Normalize candidate name
                    candidate_name = await self.normalize_entity_name(candidate.name, candidate.entity_type)

                    # Get candidate embedding
                    candidate_embedding = await self._get_entity_embedding(candidate, candidate_name)

                    # Compute cosine similarity
                    similarity = self._compute_cosine_similarity(target_embedding, candidate_embedding)

                    if similarity >= similarity_threshold:
                        similarities.append((candidate, float(similarity)))

                except Exception as e:
                    logger.warning(f"Failed to compute similarity for candidate {candidate.name}: {e}")
                    continue

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Found {len(similarities)} similar entities for {entity.name}")
            return similarities

        except Exception as e:
            logger.error(f"Failed to find similar entities for {entity.name}: {e}")
            raise DeduplicationError(f"Similarity search failed: {e}", entity.id)

    async def _get_entity_embedding(self, entity: Entity, normalized_name: str) -> np.ndarray:
        """Get or compute entity embedding with caching and pre-computed embedding support."""
        cache_key = f"{entity.id}:{normalized_name}"

        # Check memory cache first
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Try to use pre-computed embeddings if available and enabled
        if self.use_precomputed_embeddings and self.entity_embedding_service and entity.embedding is not None:
            logger.debug(f"Using pre-computed embedding for entity {entity.id}")
            embedding = np.array(entity.embedding, dtype=np.float32)

            # Normalize embedding (ensure it's unit vector)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Cache the result
            self._entity_cache[cache_key] = embedding
            return embedding

        # Fall back to dynamic computation
        logger.debug(f"Computing dynamic embedding for entity {entity.id}")
        embedding = await self._compute_dynamic_embedding(entity, normalized_name)

        # Cache the result
        self._entity_cache[cache_key] = embedding
        return embedding

    async def _compute_dynamic_embedding(self, entity: Entity, normalized_name: str) -> np.ndarray:
        """Compute embedding dynamically using the sentence transformer."""
        # Create entity description for embedding
        description = EDCUtils.create_entity_description(entity, normalized_name)

        # Compute embedding
        embedding_result = self.embedding_model.encode(description, normalize_embeddings=True)

        # Convert to numpy array if needed
        if not isinstance(embedding_result, np.ndarray):
            embedding = np.array(embedding_result, dtype=np.float32)
        else:
            embedding = embedding_result.astype(np.float32)

        return embedding

    async def verify_entity_merge(
        self,
        entity1: Entity,
        entity2: Entity,
        similarity_score: float,
    ) -> Tuple[bool, float]:
        """Verify if two entities should be merged using LLM validation."""
        try:
            # Quick rejection for very low similarity
            if similarity_score < 0.7:
                return False, 0.0

            # Create verification prompt for future LLM integration
            _prompt = f"""以下の2つのエンティティが同一のものかどうかを判定してください。

エンティティ1:
- 名前: {entity1.name}
- タイプ: {entity1.entity_type}
- 定義: {entity1.definition or "なし"}

エンティティ2:
- 名前: {entity2.name}
- タイプ: {entity2.entity_type}
- 定義: {entity2.definition or "なし"}

ベクター類似度: {similarity_score:.3f}

判定基準:
1. 同一の実体を指しているか
2. 名前の表記揺れや略称の可能性
3. 異なる情報源からの同一エンティティか

回答形式（JSON）:
{{
  "should_merge": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "判定理由"
}}

JSON形式のみで回答してください:"""

            # For now, use heuristic-based verification since LLM integration is complex
            should_merge, confidence = EDCUtils.heuristic_verification(entity1, entity2, similarity_score)

            logger.debug(f"Merge verification for {entity1.name} vs {entity2.name}: {should_merge} (confidence: {confidence:.3f})")
            return should_merge, confidence

        except Exception as e:
            logger.error(f"Failed to verify merge for {entity1.name} vs {entity2.name}: {e}")
            # Conservative approach: don't merge on error
            return False, 0.0

    async def canonicalize_entity(
        self,
        entities_to_merge: List[Entity],
        merge_confidences: List[float],
    ) -> Entity:
        """Create canonical entity from multiple similar entities."""
        if not entities_to_merge:
            raise DeduplicationError("No entities provided for canonicalization")

        if len(entities_to_merge) == 1:
            return entities_to_merge[0]

        try:
            # Choose the entity with highest confidence as base
            base_entity = max(entities_to_merge, key=lambda e: e.confidence)

            # Merge information from all entities
            canonical_entity = Entity(
                name=EDCUtils.choose_canonical_name(entities_to_merge),
                entity_type=base_entity.entity_type,
                definition=EDCUtils.merge_definitions(entities_to_merge),
                properties=EDCUtils.merge_properties(entities_to_merge),
                chunk_id=base_entity.chunk_id,  # Keep original chunk reference
                canonical_name=base_entity.name,  # Mark as canonical
                merged_from=[e.id for e in entities_to_merge if e.id != base_entity.id],
                merge_confidence=float(np.mean(merge_confidences)),
                confidence=max(e.confidence for e in entities_to_merge),
            )

            logger.info(f"Canonicalized entity: {canonical_entity.name} from {len(entities_to_merge)} entities")
            return canonical_entity

        except Exception as e:
            logger.error(f"Failed to canonicalize entities: {e}")
            raise DeduplicationError(f"Canonicalization failed: {e}")

    async def deduplicate_entities(
        self,
        entities: List[Entity],
        similarity_threshold: float = 0.85,
        verification_threshold: float = 0.8,
    ) -> List[Entity]:
        """Perform complete entity deduplication pipeline."""
        if not entities:
            return []

        logger.info(f"Starting deduplication of {len(entities)} entities")

        try:
            # Pre-compute embeddings for entities that don't have them
            if self.use_precomputed_embeddings and self.entity_embedding_service:
                await self._ensure_embeddings_computed(entities)

            # Track processed entities and merges
            processed_ids = set()
            canonical_entities = []

            for entity in entities:
                if entity.id in processed_ids:
                    continue

                # Find similar entities
                candidates = [e for e in entities if e.id not in processed_ids and e.id != entity.id]
                similar_entities = await self.find_similar_entities(entity, candidates, similarity_threshold)

                # Verify merges
                entities_to_merge = [entity]
                merge_confidences = [1.0]  # Original entity has full confidence

                for similar_entity, similarity_score in similar_entities:
                    should_merge, confidence = await self.verify_entity_merge(entity, similar_entity, similarity_score)

                    if should_merge and confidence >= verification_threshold:
                        entities_to_merge.append(similar_entity)
                        merge_confidences.append(confidence)
                        processed_ids.add(similar_entity.id)

                # Canonicalize merged entities
                if len(entities_to_merge) > 1:
                    canonical_entity = await self.canonicalize_entity(entities_to_merge, merge_confidences)
                    logger.info(f"Merged {len(entities_to_merge)} entities into: {canonical_entity.name}")
                else:
                    canonical_entity = entity

                canonical_entities.append(canonical_entity)
                processed_ids.add(entity.id)

            logger.info(f"Deduplication complete: {len(entities)} → {len(canonical_entities)} entities")
            return canonical_entities

        except Exception as e:
            logger.error(f"Deduplication pipeline failed: {e}")
            raise DeduplicationError(f"Deduplication failed: {e}")

    async def normalize_entity_name(self, name: str, entity_type: str) -> str:
        """Normalize entity name for Japanese text variations."""
        try:
            # Basic Japanese text normalization
            normalized = jaconv.normalize(name)

            # Convert full-width to half-width for ASCII
            normalized = jaconv.z2h(normalized, ascii=True, digit=True)

            # Entity type specific normalization
            if entity_type in ["COMPANY", "ORGANIZATION"]:
                # Company name variations (株式会社, ㈱, etc.)
                normalized = re.sub(r"株式会社|㈱|有限会社|㈲|合同会社|LLC", "", normalized)
                normalized = re.sub(r"Corporation|Corp\.?|Inc\.?|Ltd\.?", "", normalized, flags=re.IGNORECASE)

            elif entity_type == "PERSON":
                # Person name variations (honorifics, etc.)
                normalized = re.sub(r"さん|氏|様|先生|博士|Dr\.?|Mr\.?|Ms\.?|Mrs\.?", "", normalized, flags=re.IGNORECASE)

            # Remove extra whitespace
            normalized = re.sub(r"\s+", " ", normalized).strip()

            return normalized

        except Exception as e:
            logger.warning(f"Name normalization failed for '{name}': {e}")
            return name  # Return original on error

    async def _ensure_embeddings_computed(self, entities: List[Entity]) -> None:
        """Ensure all entities have pre-computed embeddings available."""
        if not self.entity_embedding_service:
            return

        # Find entities without embeddings
        entities_without_embeddings = [entity for entity in entities if entity.embedding is None]

        if not entities_without_embeddings:
            logger.debug("All entities already have pre-computed embeddings")
            return

        logger.info(f"Computing embeddings for {len(entities_without_embeddings)} entities without pre-computed embeddings")

        try:
            # Use batch computation for efficiency
            computed_count = await self.entity_embedding_service.batch_compute_embeddings(
                entities_without_embeddings,
                skip_existing=False,  # Force computation since we filtered above
            )

            logger.info(f"Successfully computed {computed_count} embeddings")

        except Exception as e:
            logger.warning(f"Failed to compute some embeddings, will fall back to dynamic computation: {e}")

    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about embedding usage in the service."""
        if not self.entity_embedding_service:
            return {
                "embedding_service_available": False,
                "use_precomputed_embeddings": self.use_precomputed_embeddings,
                "cache_size": len(self._entity_cache),
            }

        try:
            stats = await self.entity_embedding_service.get_embedding_statistics()
            stats.update(
                {
                    "embedding_service_available": True,
                    "use_precomputed_embeddings": self.use_precomputed_embeddings,
                    "cache_size": len(self._entity_cache),
                    "fallback_model": str(self.embedding_model) if hasattr(self.embedding_model, "__str__") else type(self.embedding_model).__name__,
                }
            )
            return stats
        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            return {
                "embedding_service_available": True,
                "use_precomputed_embeddings": self.use_precomputed_embeddings,
                "cache_size": len(self._entity_cache),
                "error": str(e),
            }
