"""Query expansion utilities for GraphRAG.

This module provides utilities for expanding queries with knowledge graph entities
and computing relevance scores.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from oboyu.domain.models.knowledge_graph import Entity
from oboyu.ports.repositories.kg_repository import KGRepository
from oboyu.ports.services.graphrag_service import GraphRAGError

logger = logging.getLogger(__name__)


class QueryExpansionHelper:
    """Helper class for query expansion operations."""

    def __init__(
        self,
        kg_repository: KGRepository,
        embedding_model: SentenceTransformer,
    ) -> None:
        """Initialize query expansion helper.

        Args:
            kg_repository: Knowledge graph repository
            embedding_model: Sentence transformer for semantic similarity

        """
        self.kg_repository = kg_repository
        self.embedding_model = embedding_model

    def extract_entity_candidates(self, query: str) -> List[str]:
        """Extract potential entity names from query text."""
        candidates = []

        # Split query into terms and phrases
        terms = query.split()

        # Add individual terms (for simple entity names)
        candidates.extend(terms)

        # Add 2-grams and 3-grams (for compound entity names)
        for i in range(len(terms) - 1):
            candidates.append(" ".join(terms[i : i + 2]))

        for i in range(len(terms) - 2):
            candidates.append(" ".join(terms[i : i + 3]))

        # Clean and filter candidates
        cleaned_candidates = []
        for candidate in candidates:
            # Remove common stop words and short terms
            if len(candidate) >= 2 and candidate.lower() not in {"の", "は", "が", "を", "に", "で", "と", "から", "まで", "について", "に関して"}:
                cleaned_candidates.append(candidate.strip())

        return list(set(cleaned_candidates))  # Remove duplicates

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings."""
        try:
            # Use sentence transformer for semantic similarity
            embeddings = self.embedding_model.encode([text1, text2], normalize_embeddings=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            return max(0.0, similarity)  # Ensure non-negative
        except Exception:
            # Fallback to character-based similarity
            return len(set(text1.lower()) & set(text2.lower())) / max(len(set(text1.lower())), len(set(text2.lower())))

    async def compute_entity_relevance_scores(
        self,
        query: str,
        entities: List[Entity],
        centrality_scores: Dict[str, float] = None,
        use_centrality: bool = True,
        use_semantic_similarity: bool = True,
    ) -> List[Tuple[Entity, float]]:
        """Compute relevance scores for entities given a query."""
        try:
            scored_entities = []

            if centrality_scores is None:
                centrality_scores = {}

            for entity in entities:
                score = 0.0

                # Semantic similarity component
                if use_semantic_similarity:
                    name_similarity = self.compute_text_similarity(query, entity.name)
                    score += name_similarity * 0.6

                    if entity.definition:
                        def_similarity = self.compute_text_similarity(query, entity.definition)
                        score += def_similarity * 0.2

                # Centrality component
                if use_centrality and entity.id in centrality_scores:
                    # Normalize centrality score
                    centrality_component = min(0.3, centrality_scores[entity.id] / 10.0)
                    score += centrality_component

                # Entity confidence component
                score += entity.confidence * 0.1

                scored_entities.append((entity, score))

            # Sort by relevance score
            scored_entities.sort(key=lambda x: x[1], reverse=True)
            return scored_entities

        except Exception as e:
            logger.error(f"Failed to compute entity relevance scores: {e}")
            raise GraphRAGError(f"Entity relevance scoring failed: {e}", "compute_entity_relevance_scores")

    def compute_chunk_relevance(self, query: str, chunk: Dict[str, Any], entities: List[Entity]) -> float:
        """Compute relevance score for a chunk given query and entities."""
        try:
            base_score = 0.0

            # Semantic similarity with content
            content_similarity = self.compute_text_similarity(query, chunk["content"][:500])
            base_score += content_similarity * 0.4

            # Entity relevance boost
            entity_boost = 0.0
            for entity_info in chunk["related_entities"]:
                entity_boost += entity_info["confidence"] * 0.1

            # Relation relevance boost
            relation_boost = 0.0
            for relation_info in chunk["related_relations"]:
                relation_boost += relation_info["confidence"] * 0.05

            # Combine scores
            total_score = base_score + min(entity_boost, 0.3) + min(relation_boost, 0.2)
            return min(1.0, total_score)

        except Exception:
            return 0.1  # Fallback score

    async def get_entity_relations(self, entity_id: str) -> List[Any]:
        """Get relations involving the specified entity."""
        try:
            # Get relations where entity is source or target
            source_relations = await self.kg_repository.get_relations_by_type("", limit=100)
            entity_relations = []

            for relation in source_relations:
                if relation.source_id == entity_id or relation.target_id == entity_id:
                    entity_relations.append(relation)

            return entity_relations
        except Exception as e:
            logger.warning(f"Failed to get relations for entity {entity_id}: {e}")
            return []
